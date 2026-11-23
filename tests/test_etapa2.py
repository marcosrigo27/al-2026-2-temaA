# [IA] ChatGPT – 23/11/2026 – testes automatizados da Etapa 2

import numpy as np
import pandas as pd
import pytest

import src.etapa2 as etapa2

from src.etapa2 import (
    extrair_transicoes,
    validar_matriz,
    calcular_resultados,
    executar,
)

from src.markov import (
    matriz_contagens,
    matriz_transicao,
    verificar_estocastica,
    distribuicao_estacionaria,
)


TOLERANCIA = 1e-12


def test_transicoes_semanas_consecutivas(tmp_path):
    # Verifica transições entre semanas separadas por 7 dias.

    dados = pd.DataFrame({
        "data_iniSE": [
            "2020-01-06",
            "2020-01-13",
            "2020-01-20",
            "2020-01-27",
        ],
        "SE": [
            202002,
            202003,
            202004,
            202005,
        ],
        "nivel": [1, 2, 3, 4],
    })

    arquivo = tmp_path / "municipio.csv"
    dados.to_csv(arquivo, index=False)

    transicoes = extrair_transicoes(arquivo)

    esperado = [
        (1, 2),
        (2, 3),
        (3, 4),
    ]

    assert transicoes == esperado


def test_semanas_ausentes_interrompem_transicoes(tmp_path):
    # Uma lacuna de 14 dias não pode gerar transição.

    dados = pd.DataFrame({
        "data_iniSE": [
            "2020-01-06",
            "2020-01-13",
            "2020-01-27",
            "2020-02-03",
        ],
        "SE": [
            202002,
            202003,
            202005,
            202006,
        ],
        "nivel": [1, 2, 3, 4],
    })

    arquivo = tmp_path / "municipio.csv"
    dados.to_csv(arquivo, index=False)

    transicoes = extrair_transicoes(arquivo)

    esperado = [
        (1, 2),
        (3, 4),
    ]

    assert transicoes == esperado

    # A passagem de 2 para 3 atravessa uma lacuna.
    assert (2, 3) not in transicoes


def test_niveis_ausentes_sao_ignorados(tmp_path):
    # Registros sem nível interrompem a sequência.

    dados = pd.DataFrame({
        "data_iniSE": [
            "2020-01-06",
            "2020-01-13",
            "2020-01-20",
            "2020-01-27",
            "2020-02-03",
        ],
        "SE": [
            202002,
            202003,
            202004,
            202005,
            202006,
        ],
        "nivel": [1, 2, None, 3, 4],
    })

    arquivo = tmp_path / "municipio.csv"
    dados.to_csv(arquivo, index=False)

    transicoes = extrair_transicoes(arquivo)

    esperado = [
        (1, 2),
        (3, 4),
    ]

    assert transicoes == esperado


@pytest.mark.parametrize("nivel_invalido", [0, 5, -1, "invalido"])
def test_niveis_invalidos_sao_ignorados(
    tmp_path,
    nivel_invalido
):
    # Níveis fora de 1 a 4 não geram transições.

    dados = pd.DataFrame({
        "data_iniSE": [
            "2020-01-06",
            "2020-01-13",
            "2020-01-20",
            "2020-01-27",
            "2020-02-03",
        ],
        "SE": [
            202002,
            202003,
            202004,
            202005,
            202006,
        ],
        "nivel": [1, 2, nivel_invalido, 3, 4],
    })

    arquivo = tmp_path / "municipio.csv"
    dados.to_csv(arquivo, index=False)

    transicoes = extrair_transicoes(arquivo)

    assert transicoes == [
        (1, 2),
        (3, 4),
    ]


def test_semanas_duplicadas(tmp_path):
    # Verifica se semanas duplicadas são detectadas.

    dados = pd.DataFrame({
        "data_iniSE": [
            "2020-01-06",
            "2020-01-13",
            "2020-01-13",
            "2020-01-20",
        ],
        "SE": [
            202002,
            202003,
            202003,
            202004,
        ],
        "nivel": [1, 2, 3, 4],
    })

    arquivo = tmp_path / "municipio.csv"
    dados.to_csv(arquivo, index=False)

    with pytest.raises(ValueError, match="duplicadas"):
        extrair_transicoes(arquivo)


def test_matriz_contagens_valores_esperados():
    # Verifica os valores da matriz de contagens.

    transicoes = [
        (1, 1),
        (1, 2),
        (1, 2),
        (2, 3),
        (2, 3),
        (3, 4),
        (4, 1),
    ]

    esperado = np.array([
        [1, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])

    C = matriz_contagens(transicoes)

    np.testing.assert_array_equal(C, esperado)

    assert C.sum() == len(transicoes)


def test_probabilidades_estocasticas():
    # Verifica probabilidades não negativas e linhas somando 1.

    C = np.array([
        [9, 1, 0, 0],
        [2, 6, 2, 0],
        [0, 2, 7, 1],
        [0, 0, 3, 7],
    ])

    P = matriz_transicao(C)

    assert verificar_estocastica(P)

    assert np.all(P >= 0)

    np.testing.assert_allclose(
        P.sum(axis=1),
        np.ones(4),
        atol=TOLERANCIA,
        rtol=0,
    )

    # Também utiliza a validação da Etapa 2.
    validar_matriz(P)


def test_erros_padrao():
    # Verifica a fórmula do erro-padrão das probabilidades.

    C = np.array([
        [9, 1, 0, 0],
        [2, 6, 2, 0],
        [0, 2, 7, 1],
        [0, 0, 3, 7],
    ])

    (
        P,
        erros,
        P_suave,
        pi,
        pi_suave,
        distancia
    ) = calcular_resultados(C, int(C.sum()))

    # Cada linha possui 10 transições.
    esperado = np.sqrt(
        P * (1 - P) / 10
    )

    np.testing.assert_allclose(
        erros,
        esperado,
        atol=TOLERANCIA,
        rtol=0,
    )

    # Confere especificamente a probabilidade 0,9.
    erro_esperado = np.sqrt(
        0.9 * (1 - 0.9) / 10
    )

    assert np.isclose(
        erros[0, 0],
        erro_esperado,
        atol=TOLERANCIA,
        rtol=0,
    )


def test_suavizacao_laplace():
    # Verifica a suavização com alfa igual a 0,5.

    C = np.array([
        [9, 1, 0, 0],
        [2, 6, 2, 0],
        [0, 2, 7, 1],
        [0, 0, 3, 7],
    ])

    (
        P,
        erros,
        P_suave,
        pi,
        pi_suave,
        distancia
    ) = calcular_resultados(C, int(C.sum()))

    alfa = 0.5

    total_por_estado = C.sum(axis=1, keepdims=True)

    esperado = (
        (C + alfa) /
        (total_por_estado + 4 * alfa)
    )

    np.testing.assert_allclose(
        P_suave,
        esperado,
        atol=TOLERANCIA,
        rtol=0,
    )

    assert verificar_estocastica(P_suave)

    # Uma transição não observada recebe probabilidade positiva.
    assert P_suave[0, 2] > 0


def test_soma_contagens_78_municipios(
    tmp_path,
    monkeypatch
):
    # Cria 78 arquivos temporários para testar a agregação.
    # Nenhum arquivo real de data/raw é alterado.

    pasta_dados = tmp_path / "raw"
    pasta_tabelas = tmp_path / "tabelas"
    pasta_figuras = tmp_path / "figuras"

    pasta_dados.mkdir()

    monkeypatch.setattr(
        etapa2,
        "PASTA_DADOS",
        pasta_dados
    )

    monkeypatch.setattr(
        etapa2,
        "PASTA_TABELAS",
        pasta_tabelas
    )

    monkeypatch.setattr(
        etapa2,
        "PASTA_FIGURAS",
        pasta_figuras
    )

    # Evita criar figuras durante este teste de agregação.
    monkeypatch.setattr(
        etapa2,
        "salvar_mapa_calor",
        lambda nome, P: None
    )

    dados = pd.DataFrame({
        "data_iniSE": [
            "2020-01-06",
            "2020-01-13",
            "2020-01-20",
            "2020-01-27",
            "2020-02-03",
        ],
        "SE": [
            202002,
            202003,
            202004,
            202005,
            202006,
        ],
        "nivel": [1, 2, 3, 4, 1],
    })

    codigos = [
        etapa2.SERRA,
        etapa2.VILA_VELHA,
    ]

    # Completa a lista com outros 76 códigos de teste.
    for numero in range(76):
        codigos.append(str(3200000 + numero))

    # Cria os 78 arquivos CSV temporários.
    for codigo in codigos:
        arquivo = pasta_dados / f"{codigo}.csv"
        dados.to_csv(arquivo, index=False)

    assert len(list(pasta_dados.glob("*.csv"))) == 78

    # Executa a agregação usando a função existente.
    executar()

    # Lê a matriz agregada gerada pela Etapa 2.
    agregado = pd.read_csv(
        pasta_tabelas / "e2_es_agregado_contagens.csv",
        index_col=0
    ).to_numpy()

    # Soma independentemente as matrizes dos 78 arquivos.
    soma_esperada = np.zeros((4, 4), dtype=int)

    total_transicoes = 0

    for arquivo in pasta_dados.glob("*.csv"):

        transicoes = extrair_transicoes(arquivo)

        C = matriz_contagens(transicoes)

        soma_esperada += C

        total_transicoes += len(transicoes)

    np.testing.assert_array_equal(
        agregado,
        soma_esperada
    )

    assert agregado.sum() == total_transicoes

    # Cada arquivo possui quatro transições válidas.
    assert total_transicoes == 78 * 4


def test_distribuicao_estacionaria_norma_l1():
    # Verifica a diferença entre as distribuições estacionárias.

    C = np.array([
        [9, 1, 0, 0],
        [2, 6, 2, 0],
        [0, 2, 7, 1],
        [0, 0, 3, 7],
    ])

    (
        P,
        erros,
        P_suave,
        pi,
        pi_suave,
        distancia
    ) = calcular_resultados(C, int(C.sum()))

    # Calcula novamente as distribuições com markov.py.
    pi_original = distribuicao_estacionaria(P)

    pi_laplace = distribuicao_estacionaria(P_suave)

    # Calcula a norma L1 independentemente.
    esperado = np.linalg.norm(
        pi_original - pi_laplace,
        ord=1
    )

    np.testing.assert_allclose(
        pi,
        pi_original,
        atol=TOLERANCIA,
        rtol=0,
    )

    np.testing.assert_allclose(
        pi_suave,
        pi_laplace,
        atol=TOLERANCIA,
        rtol=0,
    )

    assert np.isclose(
        distancia,
        esperado,
        atol=TOLERANCIA,
        rtol=0,
    )

    assert distancia >= 0


def test_soma_contagens_incorreta():
    # Verifica se o programa detecta divergência nas contagens.

    C = np.array([
        [9, 1, 0, 0],
        [2, 6, 2, 0],
        [0, 2, 7, 1],
        [0, 0, 3, 7],
    ])

    total_incorreto = int(C.sum()) + 1

    with pytest.raises(
        ValueError,
        match="Soma das contagens"
    ):
        calcular_resultados(C, total_incorreto)
# [/IA]

def test_transicao_entre_anos_com_semana_53(tmp_path):
    # Verifica a transição entre a semana 53 de 2020 e a semana 1 de 2021.

    dados = pd.DataFrame({
        "data_iniSE": [
            "2020-12-20",
            "2020-12-27",
            "2021-01-03",
        ],
        "SE": [
            202052,
            202053,
            202101,
        ],
        "nivel": [1, 2, 3],
    })

    arquivo = tmp_path / "municipio.csv"
    dados.to_csv(arquivo, index=False)

    transicoes = extrair_transicoes(arquivo)

    assert transicoes == [
        (1, 2),
        (2, 3),
    ]