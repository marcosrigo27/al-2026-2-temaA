import numpy as np
from scipy import linalg


# [IA] ChatGPT – 23/11/2026 – funções matemáticas

def matriz_contagens(transicoes):
    #Constrói a matriz de contagens dos níveis 1 a 4.
    C = np.zeros((4, 4), dtype=int)

    for origem, destino in transicoes:
        if (
            origem not in (1, 2, 3, 4)
            or destino not in (1, 2, 3, 4)
        ):
            raise ValueError("Os níveis devem estar entre 1 e 4.")

        C[origem - 1, destino - 1] += 1

    return C


def matriz_transicao(C):
    #Calcula a matriz de probabilidades de transição.
    C = np.asarray(C)

    if C.shape != (4, 4):
        raise ValueError("A matriz de contagens deve ser 4x4.")

    # Verifica se as contagens são números inteiros
    if not np.issubdtype(C.dtype, np.integer):
        raise ValueError("As contagens devem ser números inteiros.")

    # Verifica se existem contagens negativas
    if np.any(C < 0):
        raise ValueError("As contagens não podem ser negativas.")

    # Calcula o total de transições de cada estado
    somas = C.sum(axis=1)

    # Não é possível estimar probabilidades sem observações
    if np.any(somas == 0):
        estados = np.where(somas == 0)[0] + 1

        raise ValueError(
            f"Estados sem transições observadas: {estados.tolist()}. "
            "Não é possível estimar suas probabilidades."
        )

    # Divide cada contagem pelo total da respectiva linha
    P = C / somas[:, np.newaxis]

    return P


def verificar_estocastica(P, tolerancia=1e-12):
    #Verifica se a matriz é estocástica.
    P = np.asarray(P)

    if P.shape != (4, 4):
        return False

    # Verifica se todos os valores são finitos
    if not np.all(np.isfinite(P)):
        return False

    # Verifica valores não negativos
    valores_validos = np.all(P >= -tolerancia)

    # Verifica se cada linha soma 1
    somas_validas = np.allclose(
        P.sum(axis=1),
        1,
        atol=tolerancia,
        rtol=0
    )

    return bool(valores_validos and somas_validas)


def distribuicao_estacionaria(P):
    #Calcula a distribuição estacionária por sistema linear
    P = np.asarray(P, dtype=float)

    if not verificar_estocastica(P):
        raise ValueError("A matriz P não é estocástica.")

    # Monta o sistema (P.T - I) * pi = 0
    A = P.T - np.eye(4)
    b = np.zeros(4)

    # Verifica se a distribuição estacionária é única
    posto = np.linalg.matrix_rank(A, tol=1e-10)

    if posto < 3:
        raise ValueError(
            "A distribuição estacionária não é única. "
            "A matriz possui múltiplas soluções estacionárias."
        )

    # Substitui a última equação pela normalização
    A[-1, :] = 1
    b[-1] = 1

    # Resolve o sistema linear
    pi = linalg.solve(A, b)

    # Confere se os valores são probabilidades válidas
    if np.any(pi < -1e-10):
        raise ValueError("Distribuição estacionária inválida.")

    # Confere se pi * P = pi
    if not np.allclose(pi @ P, pi, atol=1e-10, rtol=0):
        raise ValueError(
            "A distribuição não satisfaz a condição estacionária."
        )

    # Confere se a soma é igual a 1
    if not np.isclose(pi.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError(
            "A soma das probabilidades estacionárias não é 1."
        )

    return pi


def calcular_autovalores(P):
    #Calcula os autovalores da matriz de transição
    P = np.asarray(P, dtype=float)

    if not verificar_estocastica(P):
        raise ValueError("A matriz P não é estocástica.")

    # Calcula os autovalores
    autovalores = linalg.eigvals(P)

    # Ordena os autovalores pelo módulo, do maior ao menor
    indices = np.argsort(-np.abs(autovalores))

    return autovalores[indices]


# [/IA]