import numpy as np
import pytest

from src.markov import (
    matriz_contagens,
    matriz_transicao,
    verificar_estocastica,
    distribuicao_estacionaria,
    calcular_autovalores,
)


# [IA] ChatGPT – 23/11/2026 – testes automatizados

# Matriz fictícia de referência fornecida pelo professor.
P_REFERENCIA = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])

TOLERANCIA = 1e-6


def test_matriz_contagens():
   #Verifica a contagem das transições entre os níveis
    transicoes = [
        (1, 1),
        (1, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ]

    esperado = np.array([
        [1, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])

    resultado = matriz_contagens(transicoes)

    np.testing.assert_array_equal(resultado, esperado)


def test_matriz_transicao_estocastica():
    #Verifica se a matriz de transição é estocástica
    contagens = np.array([
        [9, 1, 0, 0],
        [2, 6, 2, 0],
        [0, 2, 7, 1],
        [0, 0, 3, 7],
    ])

    P = matriz_transicao(contagens)

    assert verificar_estocastica(P)
    assert np.all(P >= 0)

    np.testing.assert_allclose(
        P.sum(axis=1),
        np.ones(4),
        atol=1e-12,
        rtol=0,
    )

    np.testing.assert_allclose(
        P[0],
        [0.9, 0.1, 0, 0],
        atol=TOLERANCIA,
        rtol=0,
    )


@pytest.mark.parametrize("valor", [-1, 1.5])
def test_matriz_transicao_rejeita_contagens_invalidas(valor):
    #Verifica a rejeição de contagens negativas e decimais
    contagens = np.eye(4, dtype=int)

    if isinstance(valor, float):
        contagens = contagens.astype(float)

    contagens[0, 0] = valor

    with pytest.raises(ValueError):
        matriz_transicao(contagens)


def test_matriz_transicao_rejeita_estado_sem_observacoes():
    #Verifica a rejeição de estados sem transições
    contagens = np.eye(4, dtype=int)

    # O nível 3 não possui transições observadas
    contagens[2] = 0

    with pytest.raises(
        ValueError,
        match="Estados sem transições observadas"
    ):
        matriz_transicao(contagens)


def test_matriz_referencia_estocastica():
    #Verifica se a matriz de referência é estocástica
    assert verificar_estocastica(P_REFERENCIA)


def test_distribuicao_estacionaria_referencia():
    #Compara a distribuição com os valores do professor.
    esperado = np.array([
        0.498970,
        0.205954,
        0.174125,
        0.120951,
    ])

    pi = distribuicao_estacionaria(P_REFERENCIA)

    np.testing.assert_allclose(
        pi,
        esperado,
        atol=TOLERANCIA,
        rtol=0,
    )

    # Verifica a condição pi * P = pi
    np.testing.assert_allclose(
        pi @ P_REFERENCIA,
        pi,
        atol=TOLERANCIA,
        rtol=0,
    )

    # Verifica se as probabilidades somam 1
    np.testing.assert_allclose(
        pi.sum(),
        1,
        atol=TOLERANCIA,
        rtol=0,
    )


def test_autovalores_referencia():
    #Compara os autovalores com os valores do professor.
    esperado = np.array([
        1.0,
        0.869971,
        0.618816,
        0.461213,
    ])

    obtidos = calcular_autovalores(P_REFERENCIA)

    np.testing.assert_allclose(
        obtidos,
        esperado,
        atol=TOLERANCIA,
        rtol=0,
    )


def test_distribuicao_estacionaria_nao_unica():
    #Verifica a rejeição de uma distribuição não única.
    P = np.eye(4)

    with pytest.raises(ValueError, match="não é única"):
        distribuicao_estacionaria(P)



# [/IA]