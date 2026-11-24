# [IA] ChatGPT – 24/11/2026 – Etapa 4: espectro, diagonalização e convergência

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

from src.etapa3 import (
    carregar_matriz,
    estacionaria_sistema,
    calcular_lambda2,
)


# Caminhos do projeto
RAIZ = Path(__file__).resolve().parent.parent
PASTA_TABELAS = RAIZ / "results" / "tabelas"
PASTA_FIGURAS = RAIZ / "results" / "figuras"

# Localidades analisadas
LOCALIDADES = [
    "serra",
    "vila_velha",
    "es_agregado",
]

NOMES = {
    "serra": "Serra",
    "vila_velha": "Vila Velha",
    "es_agregado": "Espírito Santo agregado",
}

# Configurações
TOLERANCIA = 1e-10
HORIZONTE = 52
MAX_SEMANAS_GRAFICO = 100

# Estilos diferentes para impressão em preto e branco
ESTILOS = [
    ("-", "o"),
    ("--", "s"),
    ("-.", "^"),
    (":", "D"),
]


def calcular_espectro(P):
    # Calcula os autovalores e autovetores à direita.
    autovalores, autovetores = linalg.eig(P)

    # Ordena os autovalores pelo módulo.
    indices = np.argsort(-np.abs(autovalores))

    autovalores = autovalores[indices]
    autovetores = autovetores[:, indices]

    # Confere se a soma dos autovalores é igual ao traço.
    soma_autovalores = np.sum(autovalores)
    traco = np.trace(P)

    erro_traco = abs(soma_autovalores - traco)

    if erro_traco > TOLERANCIA:
        raise ValueError(
            "A soma dos autovalores não coincide com o traço."
        )

    # Confere se o produto dos autovalores é igual ao determinante.
    produto_autovalores = np.prod(autovalores)
    determinante = linalg.det(P)

    erro_determinante = abs(
        produto_autovalores - determinante
    )

    if erro_determinante > TOLERANCIA:
        raise ValueError(
            "O produto dos autovalores não coincide "
            "com o determinante."
        )

    # Calcula o segundo maior módulo.
    lambda2 = float(
        np.sort(np.abs(autovalores))[-2]
    )

    # Confere o resultado com a função da Etapa 3.
    lambda2_etapa3 = calcular_lambda2(P)

    if not np.isclose(
        lambda2,
        lambda2_etapa3,
        atol=TOLERANCIA,
        rtol=0
    ):
        raise ValueError(
            "O valor de lambda2 não concorda com a Etapa 3."
        )

    return {
        "autovalores": autovalores,
        "autovetores": autovetores,
        "lambda2": lambda2,
        "traco": traco,
        "soma_autovalores": soma_autovalores,
        "erro_traco": float(erro_traco),
        "determinante": determinante,
        "produto_autovalores": produto_autovalores,
        "erro_determinante": float(erro_determinante),
    }


def calcular_meia_vida(lambda2):
    # Calcula a meia-vida usando o segundo maior módulo.
    if not np.isfinite(lambda2):
        raise ValueError(
            "O valor de lambda2 não é finito."
        )

    if lambda2 < 0 or lambda2 > 1:
        raise ValueError(
            "O módulo de lambda2 deve estar entre 0 e 1."
        )

    # Se lambda2 for zero, a memória desaparece rapidamente.
    if lambda2 == 0:
        return 0.0

    # Se lambda2 for 1, não existe decaimento exponencial.
    if lambda2 == 1:
        return float("inf")

    meia_vida = (
        np.log(0.5) / np.log(lambda2)
    )

    return float(meia_vida)


def diagonalizar_matriz(P, espectro=None):
    # Calcula P = V * Lambda * V inversa.
    if espectro is None:
        espectro = calcular_espectro(P)

    autovalores = espectro["autovalores"]
    V = espectro["autovetores"]

    Lambda = np.diag(autovalores)

    # Verifica se a matriz de autovetores é invertível.
    posto = np.linalg.matrix_rank(V)

    if posto != P.shape[0]:
        raise ValueError(
            "A matriz não é diagonalizável: "
            "os autovetores não são linearmente independentes."
        )

    # Calcula o número de condicionamento na norma 1.
    condicao = np.linalg.cond(V, p=1)

    if not np.isfinite(condicao):
        raise ValueError(
            "A matriz de autovetores possui "
            "condicionamento inválido."
        )

    # Matrizes muito mal condicionadas não são seguras.
    if condicao * np.finfo(float).eps > TOLERANCIA:
        raise ValueError(
            "A diagonalização não é numericamente confiável "
            "devido ao condicionamento elevado."
        )

    V_inversa = linalg.inv(V)

    # Reconstrói a matriz original.
    reconstruida_complexa = (
        V @ Lambda @ V_inversa
    )

    # Compara a reconstrução antes de remover partes imaginárias.
    erro_reconstrucao = np.linalg.norm(
        reconstruida_complexa - P,
        ord=1
    )

    if erro_reconstrucao > TOLERANCIA:
        raise ValueError(
            "A reconstrução por diagonalização "
            "não reproduziu a matriz original."
        )

    reconstruida = converter_real(
        reconstruida_complexa,
        "Reconstrução da matriz"
    )

    return {
        "V": V,
        "Lambda": Lambda,
        "V_inversa": V_inversa,
        "reconstruida": reconstruida,
        "condicao": float(np.real(condicao)),
        "erro_reconstrucao": float(erro_reconstrucao),
    }


def converter_real(matriz, descricao):
    # Remove apenas partes imaginárias causadas por arredondamento.
    matriz = np.asarray(matriz)

    if np.iscomplexobj(matriz):

        maior_imaginario = np.max(
            np.abs(np.imag(matriz))
        )

        if maior_imaginario > TOLERANCIA:
            raise ValueError(
                f"{descricao} possui parte imaginária "
                "numericamente significativa."
            )

        matriz = np.real(matriz)

    return np.asarray(matriz, dtype=float)


def potencia_quadrados(P, expoente):
    # Calcula P^n por quadrados sucessivos.
    if not isinstance(expoente, (int, np.integer)):
        raise ValueError(
            "O expoente deve ser inteiro."
        )

    if expoente < 0:
        raise ValueError(
            "O expoente deve ser não negativo."
        )

    n = P.shape[0]

    resultado = np.eye(n)
    base = np.asarray(P, dtype=float).copy()

    potencia = int(expoente)

    # Implementação explícita da exponenciação binária.
    while potencia > 0:

        if potencia % 2 == 1:
            resultado = resultado @ base

        base = base @ base

        potencia //= 2

    return resultado


def potencia_diagonalizacao(diagonalizacao, expoente):
    # Calcula P^n usando V * Lambda^n * V inversa.
    if not isinstance(expoente, (int, np.integer)):
        raise ValueError(
            "O expoente deve ser inteiro."
        )

    if expoente < 0:
        raise ValueError(
            "O expoente deve ser não negativo."
        )

    V = diagonalizacao["V"]
    Lambda = diagonalizacao["Lambda"]
    V_inversa = diagonalizacao["V_inversa"]

    # Eleva os elementos da diagonal ao expoente.
    autovalores = np.diag(Lambda)

    Lambda_potencia = np.diag(
        autovalores ** expoente
    )

    potencia_complexa = (
        V @ Lambda_potencia @ V_inversa
    )

    return converter_real(
        potencia_complexa,
        f"Potência P^{expoente}"
    )


def comparar_potencias(P, diagonalizacao, expoente=52):
    # Calcula P^52 por dois métodos independentes.
    P_quadrados = potencia_quadrados(
        P,
        expoente
    )

    P_diagonalizacao = potencia_diagonalizacao(
        diagonalizacao,
        expoente
    )

    # Calcula a norma matricial 1 da diferença.
    diferenca = np.linalg.norm(
        P_quadrados - P_diagonalizacao,
        ord=1
    )

    if diferenca > TOLERANCIA:
        raise ValueError(
            "Os métodos de cálculo de P^52 "
            "não concordam dentro da tolerância."
        )

    return {
        "quadrados": P_quadrados,
        "diagonalizacao": P_diagonalizacao,
        "diferenca_l1": float(diferenca),
    }


def calcular_convergencia(P, pi, max_semanas=100):
    # Calcula ||ei * P^n - pi||1 para os quatro estados.
    if max_semanas < 0:
        raise ValueError(
            "O número de semanas não pode ser negativo."
        )

    n_estados = P.shape[0]

    # A matriz identidade contém os quatro vetores iniciais.
    distribuicoes = np.eye(n_estados)

    registros = []

    for semana in range(max_semanas + 1):

        for estado in range(n_estados):

            distribuicao = distribuicoes[estado]

            distancia = np.linalg.norm(
                distribuicao - pi,
                ord=1
            )

            registros.append({
                "semana": semana,
                "estado_inicial": estado + 1,
                "distancia_l1": float(distancia),
            })

        # Avança todas as distribuições uma semana.
        distribuicoes = distribuicoes @ P

    return pd.DataFrame(registros)


def gerar_grafico_convergencia(nome, convergencia, lambda2):
    # Cria o gráfico de convergência em escala logarítmica.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a curva de referência C * |lambda2|^n.
    distancias_iniciais = convergencia[
        convergencia["semana"] == 0
    ]["distancia_l1"]

    C = float(distancias_iniciais.max())

    semanas = np.arange(
        int(convergencia["semana"].max()) + 1
    )

    referencia = C * (lambda2 ** semanas)

    for indice, estado in enumerate(range(1, 5)):

        dados_estado = convergencia[
            convergencia["estado_inicial"] == estado
        ]

        estilo, marcador = ESTILOS[indice]

        # Valores não positivos não podem aparecer no eixo log.
        valores = dados_estado["distancia_l1"].to_numpy()

        valores = np.where(
            valores > 0,
            valores,
            np.nan
        )

        ax.semilogy(
            dados_estado["semana"],
            valores,
            linestyle=estilo,
            marker=marcador,
            markevery=10,
            markersize=4,
            linewidth=1.5,
            color="black",
            label=f"Estado inicial e{estado}",
        )

    # Evita representar valores iguais a zero.
    referencia = np.where(
        referencia > 0,
        referencia,
        np.nan
    )

    ax.semilogy(
        semanas,
        referencia,
        linestyle=(0, (5, 2, 1, 2)),
        linewidth=2,
        color="gray",
        label="Referência C|λ₂|ⁿ",
    )

    ax.set_title(
        f"Convergência da cadeia de Markov - {NOMES[nome]}"
    )

    ax.set_xlabel("Semanas (n)")
    ax.set_ylabel("Distância L1 até a distribuição estacionária")

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.4
    )

    ax.legend()

    fig.tight_layout()

    caminho = (
        PASTA_FIGURAS /
        f"e4_{nome}_convergencia.png"
    )

    fig.savefig(
        caminho,
        dpi=300
    )

    plt.close(fig)


def salvar_matriz(nome, descricao, matriz):
    # Salva matrizes reais ou complexas em CSV.
    matriz = np.asarray(matriz)

    linhas = [
        f"linha_{i + 1}"
        for i in range(matriz.shape[0])
    ]

    colunas = [
        f"coluna_{i + 1}"
        for i in range(matriz.shape[1])
    ]

    tabela = pd.DataFrame(
        matriz,
        index=linhas,
        columns=colunas
    )

    tabela.to_csv(
        PASTA_TABELAS / f"e4_{nome}_{descricao}.csv",
        float_format="%.15g"
    )


def salvar_autovalores(nome, autovalores):
    # Registra os autovalores com partes real e imaginária.
    registros = []

    for indice, valor in enumerate(autovalores):

        registros.append({
            "localidade": nome,
            "indice": indice + 1,
            "parte_real": float(np.real(valor)),
            "parte_imaginaria": float(np.imag(valor)),
            "modulo": float(abs(valor)),
        })

    pd.DataFrame(registros).to_csv(
        PASTA_TABELAS / f"e4_{nome}_autovalores.csv",
        index=False,
        float_format="%.15g"
    )


def analisar_localidade(nome):
    # Executa todos os cálculos da Etapa 4.
    P = carregar_matriz(nome)

    # Calcula a distribuição estacionária.
    pi = estacionaria_sistema(P)

    # Calcula espectro e lambda2.
    espectro = calcular_espectro(P)

    lambda2 = espectro["lambda2"]

    # Calcula a meia-vida.
    meia_vida = calcular_meia_vida(lambda2)

    # Realiza a diagonalização.
    diagonalizacao = diagonalizar_matriz(
        P,
        espectro
    )

    # Calcula P^52 pelos dois métodos.
    potencias = comparar_potencias(
        P,
        diagonalizacao,
        HORIZONTE
    )

    # Calcula a convergência para os quatro estados.
    convergencia = calcular_convergencia(
        P,
        pi,
        MAX_SEMANAS_GRAFICO
    )

    # Salva os autovalores.
    salvar_autovalores(
        nome,
        espectro["autovalores"]
    )

    # Salva as matrizes da diagonalização.
    salvar_matriz(
        nome,
        "V",
        diagonalizacao["V"]
    )

    salvar_matriz(
        nome,
        "Lambda",
        diagonalizacao["Lambda"]
    )

    salvar_matriz(
        nome,
        "V_inversa",
        diagonalizacao["V_inversa"]
    )

    salvar_matriz(
        nome,
        "reconstrucao",
        diagonalizacao["reconstruida"]
    )

    # Salva P^52 pelos dois métodos.
    salvar_matriz(
        nome,
        "P52_quadrados",
        potencias["quadrados"]
    )

    salvar_matriz(
        nome,
        "P52_diagonalizacao",
        potencias["diagonalizacao"]
    )

    # Salva os dados da convergência.
    convergencia.to_csv(
        PASTA_TABELAS / f"e4_{nome}_convergencia.csv",
        index=False,
        float_format="%.15g"
    )

    # Gera o gráfico.
    gerar_grafico_convergencia(
        nome,
        convergencia,
        lambda2
    )

    resumo = {
        "localidade": nome,
        "lambda2_modulo": lambda2,
        "meia_vida_semanas": meia_vida,
        "condicao_V": diagonalizacao["condicao"],
        "erro_reconstrucao_l1": diagonalizacao[
            "erro_reconstrucao"
        ],
        "diferenca_P52_l1": potencias["diferenca_l1"],
        "erro_traco": espectro["erro_traco"],
        "erro_determinante": espectro[
            "erro_determinante"
        ],
        "verificacao_traco": (
            espectro["erro_traco"] <= TOLERANCIA
        ),
        "verificacao_determinante": (
            espectro["erro_determinante"] <= TOLERANCIA
        ),
    }

    # Imprime um resumo organizado.
    print()
    print("=" * 65)
    print(f"ETAPA 4 - {NOMES[nome].upper()}")
    print("=" * 65)

    print()
    print("Autovalores:")
    for valor in espectro["autovalores"]:
        print(f"  {valor}")

    print()
    print(f"|lambda2|: {lambda2:.6f}")
    print(f"Meia-vida: {meia_vida:.6f} semanas")

    print()
    print("Diagonalização:")
    print(
        f"  Condicionamento de V: "
        f"{diagonalizacao['condicao']:.6e}"
    )
    print(
        f"  Erro de reconstrução: "
        f"{diagonalizacao['erro_reconstrucao']:.12e}"
    )

    print()
    print("Comparação de P^52:")
    print(
        f"  Diferença na norma matricial 1: "
        f"{potencias['diferenca_l1']:.12e}"
    )

    print()
    print("Verificações:")
    print(
        f"  Soma dos autovalores = traço: "
        f"{resumo['verificacao_traco']}"
    )
    print(
        f"  Produto dos autovalores = determinante: "
        f"{resumo['verificacao_determinante']}"
    )

    return resumo


def executar():
    # Executa a Etapa 4 para Serra, Vila Velha e ES.
    PASTA_TABELAS.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_FIGURAS.mkdir(
        parents=True,
        exist_ok=True
    )

    resumos = []

    for nome in LOCALIDADES:
        resultado = analisar_localidade(nome)
        resumos.append(resultado)

    tabela_resumo = pd.DataFrame(resumos)

    tabela_resumo.to_csv(
        PASTA_TABELAS / "e4_resumo.csv",
        index=False,
        float_format="%.15g"
    )

    # Compara as localidades pela meia-vida.
    comparacao = tabela_resumo[
        [
            "localidade",
            "lambda2_modulo",
            "meia_vida_semanas",
            "condicao_V",
        ]
    ].copy()

    comparacao = comparacao.sort_values(
        "meia_vida_semanas"
    )

    comparacao.to_csv(
        PASTA_TABELAS / "e4_comparacao_localidades.csv",
        index=False,
        float_format="%.15g"
    )

    print()
    print("=" * 65)
    print("COMPARAÇÃO ENTRE AS LOCALIDADES")
    print("=" * 65)

    print()
    print(
        comparacao.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()
    print("Interpretação:")
    print(
        "Quanto maior |lambda2|, mais lenta tende a ser "
        "a convergência da cadeia."
    )
    print(
        "Quanto maior a meia-vida, mais tempo o estado "
        "inicial influencia a distribuição futura."
    )

    print()
    print("=" * 65)
    print("ETAPA 4 CONCLUÍDA")
    print("=" * 65)

    print()
    print("Resultados salvos em:")
    print("results/tabelas/")
    print("results/figuras/")


if __name__ == "__main__":
    executar()


# [/IA]