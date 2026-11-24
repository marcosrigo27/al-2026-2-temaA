# [IA] ChatGPT – 24/11/2026 – testes automatizados da Etapa 3

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

import src.etapa3 as etapa3

from src.etapa3 import (
    carregar_matriz,
    estacionaria_sistema,
    estacionaria_potencias,
    estacionaria_autovetor,
    comparar_distribuicoes,
    calcular_residuo,
    verificar_primitividade,
    calcular_lambda2,
    frequencia_empirica,
    analisar_localidade,
)

from src.markov import verificar_estocastica


# Caminhos do projeto
RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data" / "raw"
PASTA_TABELAS = RAIZ / "results" / "tabelas"

# Tolerâncias
TOLERANCIA = 1e-10
TOLERANCIA_REFERENCIA = 1e-6
TOLERANCIA_POTENCIAS = 1e-12

# Matrizes artificiais usadas nos testes unitários
P_ARTIFICIAL = np.array([
    [0.7, 0.1, 0.1, 0.1],
    [0.1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
    [0.1, 0.1, 0.1, 0.7],
])

P_IDENTIDADE = np.eye(4)

# Matriz de referência fornecida pelo professor
P_REFERENCIA = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])

PI_REFERENCIA = np.array([
    0.498970,
    0.205954,
    0.174125,
    0.120951,
])

AUTOVALORES_REFERENCIA = np.array([
    1.0,
    0.869971,
    0.618816,
    0.461213,
])

# Distribuições iniciais
DISTRIBUICOES_INICIAIS = {
    "e1": np.array([1.0, 0.0, 0.0, 0.0]),
    "e4": np.array([0.0, 0.0, 0.0, 1.0]),
    "uniforme": np.array([0.25, 0.25, 0.25, 0.25]),
}

# Resultados reais já obtidos na Etapa 3
RESULTADOS_REAIS = {
    "serra": {
        "codigo": "3205002",
        "pi": [0.537522, 0.223386, 0.068063, 0.171030],
        "lambda2": 0.879408,
        "menor_k": 2,
    },
    "vila_velha": {
        "codigo": "3205200",
        "pi": [0.493892, 0.226876, 0.075044, 0.204188],
        "lambda2": 0.881074,
        "menor_k": 2,
    },
    "es_agregado": {
        "codigo": None,
        "pi": [0.774102, 0.097554, 0.010252, 0.118092],
        "lambda2": 0.885841,
        "menor_k": 2,
    },
}


# --------------------------------------------------
# TESTES DE LEITURA DAS MATRIZES
# --------------------------------------------------

def test_carregar_matriz_temporaria(tmp_path, monkeypatch):
    # Cria uma matriz CSV artificial para testar a leitura.

    monkeypatch.setattr(
        etapa3,
        "PASTA_TABELAS",
        tmp_path
    )

    tabela = pd.DataFrame(
        P_ARTIFICIAL,
        index=[
            "origem_1",
            "origem_2",
            "origem_3",
            "origem_4",
        ],
        columns=[
            "destino_1",
            "destino_2",
            "destino_3",
            "destino_4",
        ],
    )

    arquivo = tmp_path / "e2_teste_probabilidades.csv"

    tabela.to_csv(arquivo)

    P = carregar_matriz("teste")

    assert P.shape == (4, 4)

    np.testing.assert_allclose(
        P,
        P_ARTIFICIAL,
        atol=TOLERANCIA,
        rtol=0
    )

    assert verificar_estocastica(P)


def test_carregar_matriz_inexistente(tmp_path, monkeypatch):
    # Verifica a rejeição de um arquivo inexistente.

    monkeypatch.setattr(
        etapa3,
        "PASTA_TABELAS",
        tmp_path
    )

    with pytest.raises(FileNotFoundError):
        carregar_matriz("inexistente")


# --------------------------------------------------
# TESTES DO SISTEMA LINEAR
# --------------------------------------------------

def test_estacionaria_sistema():
    # Uma matriz simétrica e estocástica possui pi uniforme.

    esperado = np.array([
        0.25,
        0.25,
        0.25,
        0.25,
    ])

    pi = estacionaria_sistema(P_ARTIFICIAL)

    np.testing.assert_allclose(
        pi,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_estacionaria_sistema_referencia():
    # Verifica a distribuição da Tabela 1 do professor.

    pi = estacionaria_sistema(P_REFERENCIA)

    np.testing.assert_allclose(
        pi,
        PI_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DO MÉTODO DAS POTÊNCIAS
# --------------------------------------------------

@pytest.mark.parametrize(
    "nome,distribuicao",
    list(DISTRIBUICOES_INICIAIS.items())
)
def test_estacionaria_potencias(nome, distribuicao):
    # Testa a convergência a partir de e1, e4 e uniforme.

    esperado = np.array([
        0.25,
        0.25,
        0.25,
        0.25,
    ])

    pi, iteracoes, diferenca = estacionaria_potencias(
        P_ARTIFICIAL,
        distribuicao
    )

    np.testing.assert_allclose(
        pi,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )

    assert iteracoes >= 1

    assert diferenca < TOLERANCIA_POTENCIAS


@pytest.mark.parametrize(
    "nome,distribuicao",
    list(DISTRIBUICOES_INICIAIS.items())
)
def test_potencias_referencia(nome, distribuicao):
    # Verifica a convergência para a matriz do professor.

    pi, iteracoes, diferenca = estacionaria_potencias(
        P_REFERENCIA,
        distribuicao
    )

    np.testing.assert_allclose(
        pi,
        PI_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )

    assert iteracoes >= 1

    assert diferenca < TOLERANCIA_POTENCIAS


def test_criterio_parada_potencias():
    # Confere o critério de parada usando a norma L1.

    inicial = np.array([
        1.0, 0.0, 0.0, 0.0
    ])

    pi, iteracoes, diferenca = estacionaria_potencias(
        P_ARTIFICIAL,
        inicial
    )

    assert diferenca < 1e-12

    # Reproduz as iterações para conferir o valor retornado.
    atual = inicial.copy()

    for _ in range(iteracoes - 1):
        atual = atual @ P_ARTIFICIAL

    proxima = atual @ P_ARTIFICIAL

    diferenca_calculada = np.linalg.norm(
        proxima - atual,
        ord=1
    )

    np.testing.assert_allclose(
        pi,
        proxima,
        atol=1e-14,
        rtol=0
    )

    assert np.isclose(
        diferenca,
        diferenca_calculada,
        atol=1e-14,
        rtol=0
    )

    assert diferenca_calculada < 1e-12


@pytest.mark.parametrize(
    "distribuicao",
    [
        [1, 0, 0],
        [-0.1, 0.5, 0.3, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        [0, 0, 0, 0],
    ]
)
def test_potencias_rejeita_distribuicoes_invalidas(distribuicao):
    # Verifica a rejeição de distribuições inválidas.

    with pytest.raises(ValueError):
        estacionaria_potencias(
            P_ARTIFICIAL,
            distribuicao
        )


# --------------------------------------------------
# TESTES DO MÉTODO DOS AUTOVETORES
# --------------------------------------------------

def test_estacionaria_autovetor():
    # Confere a distribuição estacionária por autovetor.

    esperado = np.array([
        0.25,
        0.25,
        0.25,
        0.25,
    ])

    pi = estacionaria_autovetor(P_ARTIFICIAL)

    np.testing.assert_allclose(
        pi,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_estacionaria_autovetor_referencia():
    # Verifica o resultado da Tabela 1.

    pi = estacionaria_autovetor(P_REFERENCIA)

    np.testing.assert_allclose(
        pi,
        PI_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


def test_autovetor_independente():
    # Confere o autovetor diretamente com SciPy.

    pi_etapa3 = estacionaria_autovetor(P_REFERENCIA)

    autovalores, autovetores = linalg.eig(
        P_REFERENCIA.T
    )

    indice = np.argmin(
        np.abs(autovalores - 1)
    )

    vetor = np.real(
        autovetores[:, indice]
    )

    pi_scipy = vetor / vetor.sum()

    np.testing.assert_allclose(
        pi_etapa3,
        pi_scipy,
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DA CONCORDÂNCIA DOS MÉTODOS
# --------------------------------------------------

def test_concordancia_tres_metodos():
    # Compara os três métodos para a matriz de referência.

    pi_sistema = estacionaria_sistema(P_REFERENCIA)

    pi_potencias, _, _ = estacionaria_potencias(
        P_REFERENCIA,
        DISTRIBUICOES_INICIAIS["e1"]
    )

    pi_autovetor = estacionaria_autovetor(P_REFERENCIA)

    comparacao = comparar_distribuicoes(
        pi_sistema,
        pi_potencias,
        pi_autovetor
    )

    assert comparacao["maior_diferenca"] <= 1e-10

    assert comparacao["sistema_potencias"] <= 1e-10

    assert comparacao["sistema_autovetor"] <= 1e-10

    assert comparacao["potencias_autovetor"] <= 1e-10


def test_rejeitar_metodos_divergentes():
    # Verifica se diferenças superiores à tolerância são rejeitadas.

    pi_sistema = np.array([
        0.25, 0.25, 0.25, 0.25
    ])

    pi_potencias = np.array([
        0.30, 0.20, 0.25, 0.25
    ])

    pi_autovetor = pi_sistema.copy()

    with pytest.raises(ValueError):
        comparar_distribuicoes(
            pi_sistema,
            pi_potencias,
            pi_autovetor
        )


# --------------------------------------------------
# TESTES DO RESÍDUO E NORMALIZAÇÃO
# --------------------------------------------------

def test_residuo_estacionario():
    # Confere a condição pi * P = pi.

    pi = estacionaria_sistema(P_REFERENCIA)

    residuo = calcular_residuo(
        P_REFERENCIA,
        pi
    )

    assert residuo <= 1e-10

    np.testing.assert_allclose(
        pi @ P_REFERENCIA,
        pi,
        atol=TOLERANCIA,
        rtol=0
    )


def test_normalizacao_distribuicao():
    # Verifica se as probabilidades somam 1.

    pi = estacionaria_sistema(P_REFERENCIA)

    assert np.all(pi >= 0)

    assert np.isclose(
        pi.sum(),
        1.0,
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DA PRIMITIVIDADE
# --------------------------------------------------

def test_matriz_primitiva():
    # Uma matriz com todas as entradas positivas é primitiva.

    resultado = verificar_primitividade(
        P_ARTIFICIAL
    )

    assert resultado["primitiva"] is True

    assert resultado["menor_k"] == 1

    assert resultado["limite_wielandt"] == 10

    assert resultado["dentro_limite"] is True


def test_matriz_nao_primitiva():
    # A matriz identidade não é primitiva.

    resultado = verificar_primitividade(
        P_IDENTIDADE
    )

    assert resultado["primitiva"] is False

    assert resultado["menor_k"] is None

    assert resultado["limite_wielandt"] == 10


def test_primitividade_referencia():
    # A matriz de referência possui menor k igual a 2.

    resultado = verificar_primitividade(
        P_REFERENCIA
    )

    assert resultado["primitiva"] is True

    assert resultado["menor_k"] == 2

    assert resultado["limite_wielandt"] == 10

    # P possui entradas iguais a zero.
    assert np.any(P_REFERENCIA == 0)

    # P ao quadrado possui todas as entradas positivas.
    P2 = P_REFERENCIA @ P_REFERENCIA

    assert np.all(P2 > 0)


def test_limite_wielandt():
    # Verifica o limite para quatro estados.

    n = 4

    esperado = (n - 1) ** 2 + 1

    resultado = verificar_primitividade(
        P_REFERENCIA
    )

    assert esperado == 10

    assert resultado["limite_wielandt"] == esperado

    assert resultado["menor_k"] <= esperado


# --------------------------------------------------
# TESTES DOS AUTOVALORES
# --------------------------------------------------

def test_lambda2_artificial():
    # Verifica o segundo maior módulo dos autovalores.

    lambda2 = calcular_lambda2(
        P_ARTIFICIAL
    )

    # A matriz artificial possui autovalores 1 e 0,6.
    assert np.isclose(
        lambda2,
        0.6,
        atol=TOLERANCIA,
        rtol=0
    )


def test_lambda2_referencia():
    # Verifica os autovalores da Tabela 1.

    lambda2 = calcular_lambda2(
        P_REFERENCIA
    )

    assert np.isclose(
        lambda2,
        0.869971,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


def test_autovalores_referencia():
    # Confere o espectro completo da matriz de referência.

    autovalores = linalg.eigvals(
        P_REFERENCIA
    )

    autovalores = np.sort(
        np.abs(autovalores)
    )[::-1]

    np.testing.assert_allclose(
        autovalores,
        AUTOVALORES_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DAS FREQUÊNCIAS EMPÍRICAS
# --------------------------------------------------

def test_frequencia_empirica_temporaria(
    tmp_path,
    monkeypatch
):
    # Cria um CSV artificial para conferir as frequências.

    monkeypatch.setattr(
        etapa3,
        "PASTA_DADOS",
        tmp_path
    )

    calendario = etapa3.gerar_semanas_esperadas().iloc[:8]

    dados = pd.DataFrame({
        "data_iniSE": calendario["data"].dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
        "SE": calendario["SE"].tolist(),
        "nivel": [1, 1, 1, 2, 2, 3, 4, 4],
        "versao_modelo": ["teste"] * 8,
    })

    dados.to_csv(
        tmp_path / "3205002.csv",
        index=False
    )

    frequencias, contagens, total = frequencia_empirica(
        ["3205002"]
    )

    esperado = np.array([
        3 / 8,
        2 / 8,
        1 / 8,
        2 / 8,
    ])

    np.testing.assert_allclose(
        frequencias,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )

    np.testing.assert_array_equal(
        contagens,
        [3, 2, 1, 2]
    )

    assert total == 8


@pytest.mark.parametrize(
    "nome",
    ["serra", "vila_velha", "es_agregado"]
)
def test_frequencias_empiricas_reais(nome):
    # Consulta os CSVs reais somente para leitura.

    codigo = RESULTADOS_REAIS[nome]["codigo"]

    if codigo is None:
        codigos = None
    else:
        codigos = [codigo]

    frequencias, contagens, total = frequencia_empirica(
        codigos
    )

    assert frequencias.shape == (4,)

    assert np.all(frequencias >= 0)

    assert np.isclose(
        frequencias.sum(),
        1,
        atol=TOLERANCIA,
        rtol=0
    )

    assert contagens.sum() == total

    assert total > 0

    # Calcula a frequência novamente pelas contagens.
    esperado = contagens / total

    np.testing.assert_allclose(
        frequencias,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DE INTEGRAÇÃO COM AS MATRIZES REAIS
# --------------------------------------------------

@pytest.mark.parametrize(
    "nome",
    ["serra", "vila_velha", "es_agregado"]
)
def test_carregar_matrizes_reais(nome):
    # Verifica os arquivos produzidos na Etapa 2.

    P = carregar_matriz(nome)

    assert P.shape == (4, 4)

    assert verificar_estocastica(P)

    assert np.all(P >= 0)

    np.testing.assert_allclose(
        P.sum(axis=1),
        np.ones(4),
        atol=1e-12,
        rtol=0
    )


@pytest.mark.parametrize(
    "nome",
    ["serra", "vila_velha", "es_agregado"]
)
def test_resultados_estacionarios_reais(nome):
    # Compara os resultados com os valores já obtidos.

    P = carregar_matriz(nome)

    esperado = RESULTADOS_REAIS[nome]

    pi = estacionaria_sistema(P)

    np.testing.assert_allclose(
        pi,
        esperado["pi"],
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )

    assert np.isclose(
        pi.sum(),
        1,
        atol=TOLERANCIA,
        rtol=0
    )

    assert calcular_residuo(P, pi) <= 1e-10


@pytest.mark.parametrize(
    "nome",
    ["serra", "vila_velha", "es_agregado"]
)
def test_lambda2_real(nome):
    # Verifica o segundo maior módulo dos autovalores.

    P = carregar_matriz(nome)

    lambda2 = calcular_lambda2(P)

    esperado = RESULTADOS_REAIS[nome]["lambda2"]

    assert np.isclose(
        lambda2,
        esperado,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


@pytest.mark.parametrize(
    "nome",
    ["serra", "vila_velha", "es_agregado"]
)
def test_primitividade_real(nome):
    # Verifica o menor k das matrizes reais.

    P = carregar_matriz(nome)

    resultado = verificar_primitividade(P)

    esperado = RESULTADOS_REAIS[nome]["menor_k"]

    assert resultado["primitiva"] is True

    assert resultado["menor_k"] == esperado

    assert resultado["limite_wielandt"] == 10

    # Confere diretamente que P^k é positiva.
    potencia = np.linalg.matrix_power(
        P,
        esperado
    )

    assert np.all(potencia > 0)


@pytest.mark.parametrize(
    "nome",
    ["serra", "vila_velha", "es_agregado"]
)
def test_concordancia_metodos_reais(nome):
    # Verifica os três métodos nas matrizes reais.

    P = carregar_matriz(nome)

    pi_sistema = estacionaria_sistema(P)

    pi_autovetor = estacionaria_autovetor(P)

    for distribuicao in DISTRIBUICOES_INICIAIS.values():

        pi_potencias, iteracoes, diferenca = (
            estacionaria_potencias(
                P,
                distribuicao
            )
        )

        comparacao = comparar_distribuicoes(
            pi_sistema,
            pi_potencias,
            pi_autovetor
        )

        assert comparacao["maior_diferenca"] <= 1e-10

        assert diferenca < 1e-12

        assert iteracoes >= 1


@pytest.mark.parametrize(
    "nome",
    ["serra", "vila_velha", "es_agregado"]
)
def test_analisar_localidade_real(nome):
    # Executa a análise completa da localidade.

    codigo = RESULTADOS_REAIS[nome]["codigo"]

    if codigo is None:
        codigos = None
    else:
        codigos = [codigo]

    (
        resumo,
        distribuicoes,
        potencias,
        comparacoes
    ) = analisar_localidade(
        nome,
        codigos
    )

    assert resumo["localidade"] == nome

    assert resumo["residuo_l1"] <= 1e-10

    assert resumo["primitiva"] is True

    assert resumo["menor_k"] == 2

    assert resumo["limite_wielandt"] == 10

    assert len(distribuicoes) == 4

    assert len(potencias) == 3

    assert len(comparacoes) == 3

    for comparacao in comparacoes:
        assert comparacao["maior_diferenca"] <= 1e-10


# [/IA]