# [IA] ChatGPT – 24/11/2026 – testes automatizados da Etapa 1

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.etapa1 as etapa1

from src.etapa1 import (
    calcular_hash,
    ler_manifesto,
    conferir_integridade,
    inicio_ano_epidemiologico,
    gerar_semanas_esperadas,
    ler_dados,
    conferir_calendario,
    contar_transicoes_validas,
    analisar_municipio,
)


# Configurações dos testes
RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data" / "raw"

SERRA = "3205002"
VILA_VELHA = "3205200"


def criar_csv_temporario(tmp_path, dados):
    # Cria um CSV artificial sem alterar os dados reais.
    arquivo = tmp_path / "municipio_teste.csv"

    dados.to_csv(arquivo, index=False)

    return arquivo


def criar_dados_calendario(indices, niveis):
    # Cria registros artificiais usando semanas do calendário.
    calendario = gerar_semanas_esperadas()

    linhas = calendario.iloc[indices]

    return pd.DataFrame({
        "data_iniSE": linhas["data"].dt.strftime("%Y-%m-%d").tolist(),
        "SE": linhas["SE"].tolist(),
        "nivel": niveis,
        "versao_modelo": ["teste"] * len(indices),
    })


# --------------------------------------------------
# TESTES DO CALENDÁRIO EPIDEMIOLÓGICO
# --------------------------------------------------

def test_total_semanas_esperadas():
    # Verifica o total de 574 semanas entre 2015 e 2025.
    calendario = gerar_semanas_esperadas()

    assert len(calendario) == 574

    assert calendario["SE"].nunique() == 574

    assert calendario["data"].nunique() == 574


def test_calendario_anos_2015_2025():
    # Verifica se todos os anos estão presentes.
    calendario = gerar_semanas_esperadas()

    anos = set(calendario["ano"])

    assert anos == set(range(2015, 2026))


def test_semanas_53():
    # Verifica as semanas 53 dos anos 2020 e 2025.
    calendario = gerar_semanas_esperadas()

    semanas = set(calendario["SE"])

    assert 202053 in semanas
    assert 202553 in semanas

    assert len(calendario[calendario["ano"] == 2020]) == 53
    assert len(calendario[calendario["ano"] == 2025]) == 53


def test_mudanca_ano_epidemiologico():
    # Confere a continuidade nas mudanças de ano.
    calendario = gerar_semanas_esperadas()

    datas = calendario["data"].sort_values()

    diferencas = datas.diff().dropna()

    assert (
        diferencas == pd.Timedelta(days=7)
    ).all()

    # Confere o início epidemiológico de 2021.
    inicio_2021 = inicio_ano_epidemiologico(2021)

    assert inicio_2021 == date(2021, 1, 3)

    # Confere a passagem da semana 53 para a semana 1.
    semana_53 = calendario[
        calendario["SE"] == 202053
    ]["data"].iloc[0]

    semana_1 = calendario[
        calendario["SE"] == 202101
    ]["data"].iloc[0]

    assert semana_1 - semana_53 == pd.Timedelta(days=7)


def test_inicio_ano_epidemiologico():
    # Verifica datas conhecidas do calendário implementado.
    assert inicio_ano_epidemiologico(2015) == date(2015, 1, 4)
    assert inicio_ano_epidemiologico(2020) == date(2019, 12, 29)
    assert inicio_ano_epidemiologico(2021) == date(2021, 1, 3)
    assert inicio_ano_epidemiologico(2025) == date(2024, 12, 29)


# --------------------------------------------------
# TESTES DE SEMANAS AUSENTES E DUPLICADAS
# --------------------------------------------------

def test_identificar_semana_ausente(tmp_path):
    # Remove uma semana de uma sequência artificial.
    calendario = gerar_semanas_esperadas().iloc[:5]

    dados = pd.DataFrame({
        "data_iniSE": calendario["data"].dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
        "SE": calendario["SE"].tolist(),
        "nivel": [1, 2, 3, 4, 1],
        "versao_modelo": ["teste"] * 5,
    })

    # Remove a terceira semana.
    dados = dados.drop(index=2)

    arquivo = criar_csv_temporario(tmp_path, dados)

    resultado = analisar_municipio(
        arquivo,
        calendario
    )

    assert resultado["semanas_esperadas"] == 5
    assert resultado["semanas_observadas"] == 4
    assert resultado["semanas_ausentes"] == 1
    assert resultado["duplicatas"] == 0

    # Apenas dois pares consecutivos permanecem.
    assert resultado["transicoes_validas"] == 2


def test_identificar_semana_duplicada(tmp_path):
    # Cria uma sequência com uma semana repetida.
    dados = criar_dados_calendario(
        [0, 1, 1, 2, 3],
        [1, 2, 2, 3, 4]
    )

    arquivo = criar_csv_temporario(tmp_path, dados)

    calendario = gerar_semanas_esperadas().iloc[:4]

    resultado = analisar_municipio(
        arquivo,
        calendario
    )

    assert resultado["semanas_esperadas"] == 4
    assert resultado["semanas_observadas"] == 4
    assert resultado["duplicatas"] == 1

    # A semana duplicada não participa das transições.
    assert resultado["transicoes_validas"] == 1


# --------------------------------------------------
# TESTES DA CONTAGEM DOS NÍVEIS
# --------------------------------------------------

def test_contagem_niveis(tmp_path):
    # Verifica a contagem dos quatro níveis.
    dados = criar_dados_calendario(
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 1, 2, 2, 3, 3, 4, 4]
    )

    arquivo = criar_csv_temporario(tmp_path, dados)

    calendario = gerar_semanas_esperadas().iloc[:8]

    resultado = analisar_municipio(
        arquivo,
        calendario
    )

    assert resultado["nivel_1"] == 2
    assert resultado["nivel_2"] == 2
    assert resultado["nivel_3"] == 2
    assert resultado["nivel_4"] == 2

    assert resultado["semanas_nivel_valido"] == 8

    soma = (
        resultado["nivel_1"]
        + resultado["nivel_2"]
        + resultado["nivel_3"]
        + resultado["nivel_4"]
    )

    assert soma == resultado["semanas_nivel_valido"]


# --------------------------------------------------
# TESTES DAS TRANSIÇÕES
# --------------------------------------------------

def test_transicoes_semanas_consecutivas():
    # Quatro semanas consecutivas geram três transições.
    dados = criar_dados_calendario(
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    )

    dados["data_iniSE"] = pd.to_datetime(
        dados["data_iniSE"]
    )

    resultado = contar_transicoes_validas(dados)

    assert resultado == 3


def test_transicoes_com_lacuna():
    # Uma semana ausente interrompe as transições.
    dados = criar_dados_calendario(
        [0, 1, 3, 4],
        [1, 2, 3, 4]
    )

    dados["data_iniSE"] = pd.to_datetime(
        dados["data_iniSE"]
    )

    resultado = contar_transicoes_validas(dados)

    assert resultado == 2


@pytest.mark.parametrize("nivel_invalido", [
    None,
    0,
    5,
    -1,
    np.nan,
])
def test_transicoes_niveis_invalidos(nivel_invalido):
    # Níveis inválidos interrompem as transições.
    dados = criar_dados_calendario(
        [0, 1, 2, 3, 4],
        [1, 2, nivel_invalido, 3, 4]
    )

    dados["data_iniSE"] = pd.to_datetime(
        dados["data_iniSE"]
    )

    resultado = contar_transicoes_validas(dados)

    assert resultado == 2


def test_transicoes_semanas_duplicadas():
    # Semanas duplicadas são excluídas das transições.
    dados = criar_dados_calendario(
        [0, 1, 1, 2, 3],
        [1, 2, 2, 3, 4]
    )

    dados["data_iniSE"] = pd.to_datetime(
        dados["data_iniSE"]
    )

    resultado = contar_transicoes_validas(dados)

    # Apenas a transição da semana 2 para a semana 3 é válida.
    assert resultado == 1


# --------------------------------------------------
# TESTES DE DATAS E CÓDIGOS SE
# --------------------------------------------------

def test_rejeitar_data_invalida(tmp_path):
    # Uma data inexistente deve ser rejeitada.
    dados = pd.DataFrame({
        "data_iniSE": ["2020-02-30"],
        "SE": [202009],
        "nivel": [1],
        "versao_modelo": ["teste"],
    })

    arquivo = criar_csv_temporario(tmp_path, dados)

    with pytest.raises(ValueError, match="Datas inválidas"):
        ler_dados(arquivo)


def test_rejeitar_data_fora_calendario():
    # Uma data fora do calendário deve ser rejeitada.
    dados = pd.DataFrame({
        "data_iniSE": pd.to_datetime(["2020-01-01"]),
        "SE": [202001],
        "nivel": [1],
        "versao_modelo": ["teste"],
    })

    calendario = gerar_semanas_esperadas()

    with pytest.raises(
        ValueError,
        match="Datas fora do calendário"
    ):
        conferir_calendario(
            dados,
            calendario,
            "teste.csv"
        )


def test_rejeitar_codigo_se_incorreto():
    # Uma data correta com SE incorreta deve ser rejeitada.
    calendario = gerar_semanas_esperadas()

    primeira = calendario.iloc[0]

    dados = pd.DataFrame({
        "data_iniSE": [primeira["data"]],
        "SE": [999999],
        "nivel": [1],
        "versao_modelo": ["teste"],
    })

    with pytest.raises(
        ValueError,
        match="Inconsistência entre SE"
    ):
        conferir_calendario(
            dados,
            calendario,
            "teste.csv"
        )


# --------------------------------------------------
# TESTES DE INTEGRIDADE DOS ARQUIVOS
# --------------------------------------------------

def test_calcular_hash(tmp_path):
    # O hash deve ser igual para arquivos com conteúdo igual.
    arquivo1 = tmp_path / "arquivo1.csv"
    arquivo2 = tmp_path / "arquivo2.csv"

    conteudo = b"SE,nivel\n202001,1\n"

    arquivo1.write_bytes(conteudo)
    arquivo2.write_bytes(conteudo)

    assert calcular_hash(arquivo1) == calcular_hash(arquivo2)

    # Uma alteração no conteúdo deve modificar o hash.
    arquivo2.write_bytes(b"SE,nivel\n202001,2\n")

    assert calcular_hash(arquivo1) != calcular_hash(arquivo2)


def test_ler_manifesto(tmp_path, monkeypatch):
    # Verifica a leitura de um manifesto temporário.
    manifesto = tmp_path / "MANIFEST.txt"

    manifesto.write_text(
        "Data e hora: 2026-11-23T10:00:00-03:00\n"
        "Arquivo: data/raw/3205002.csv\n"
        "URL: https://exemplo.test/dados.csv\n"
        "Linhas: 574\n"
        "SHA-256: abc123\n\n",
        encoding="utf-8"
    )

    monkeypatch.setattr(
        etapa1,
        "MANIFESTO",
        manifesto
    )

    registros = ler_manifesto()

    assert "3205002.csv" in registros

    assert registros["3205002.csv"]["Linhas"] == "574"

    assert registros["3205002.csv"]["SHA-256"] == "abc123"


def test_detectar_hash_diferente(tmp_path, monkeypatch):
    # Cria 78 arquivos artificiais para testar o manifesto.
    pasta = tmp_path / "raw"
    pasta.mkdir()

    manifesto = tmp_path / "MANIFEST.txt"

    arquivos = []
    registros = []

    for numero in range(78):
        codigo = f"{3200000 + numero:07d}"
        arquivo = pasta / f"{codigo}.csv"

        arquivo.write_text(
            "SE,nivel\n202001,1\n",
            encoding="utf-8"
        )

        arquivos.append(arquivo)

        hash_original = calcular_hash(arquivo)

        registros.append(
            f"Data e hora: 2026-11-23T10:00:00-03:00\n"
            f"Arquivo: data/raw/{arquivo.name}\n"
            f"URL: https://exemplo.test/{arquivo.name}\n"
            f"Linhas: 1\n"
            f"SHA-256: {hash_original}\n"
        )

    manifesto.write_text(
        "\n\n".join(registros) + "\n\n",
        encoding="utf-8"
    )

    monkeypatch.setattr(
        etapa1,
        "MANIFESTO",
        manifesto
    )

    # Altera um arquivo depois do registro do hash.
    arquivos[0].write_text(
        "SE,nivel\n202001,4\n",
        encoding="utf-8"
    )

    with pytest.raises(
        ValueError,
        match="Falha de integridade"
    ):
        conferir_integridade(arquivos)


# --------------------------------------------------
# TESTES DE INTEGRAÇÃO COM OS DADOS REAIS
# --------------------------------------------------

@pytest.mark.parametrize(
    "codigo",
    [SERRA, VILA_VELHA]
)
def test_auditoria_municipios_reais(codigo):
    # Lê os arquivos reais sem realizar modificações.
    arquivo = PASTA_DADOS / f"{codigo}.csv"

    assert arquivo.exists(), (
        f"Arquivo real não encontrado: {arquivo}"
    )

    calendario = gerar_semanas_esperadas()

    resultado = analisar_municipio(
        arquivo,
        calendario
    )

    # Valores já conferidos para Serra e Vila Velha.
    assert resultado["semanas_esperadas"] == 574

    assert resultado["semanas_observadas"] == 574

    assert resultado["semanas_ausentes"] == 0

    assert resultado["duplicatas"] == 0

    assert resultado["transicoes_validas"] == 573

    # Confere a soma dos níveis.
    soma_niveis = (
        resultado["nivel_1"]
        + resultado["nivel_2"]
        + resultado["nivel_3"]
        + resultado["nivel_4"]
    )

    assert soma_niveis == resultado["semanas_nivel_valido"]

    # Confere se a quantidade de semanas válidas é coerente.
    assert resultado["semanas_nivel_valido"] <= 574


# [/IA]