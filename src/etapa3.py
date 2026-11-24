# [IA] ChatGPT – 24/11/2026 – Etapa 3: distribuição estacionária e primitividade

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

from src.markov import (
    verificar_estocastica,
    distribuicao_estacionaria,
    calcular_autovalores,
)

from src.etapa1 import (
    ler_dados,
    gerar_semanas_esperadas,
    conferir_calendario,
)


# Caminhos do projeto
RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data" / "raw"
PASTA_TABELAS = RAIZ / "results" / "tabelas"

# Municípios escolhidos
SERRA = "3205002"
VILA_VELHA = "3205200"

# Configurações
NIVEIS = [1, 2, 3, 4]

TOLERANCIA_POTENCIAS = 1e-12
TOLERANCIA_COMPARACAO = 1e-10
MAX_ITERACOES = 1000000

# Distribuições iniciais
DISTRIBUICOES_INICIAIS = {
    "e1": np.array([1.0, 0.0, 0.0, 0.0]),
    "e4": np.array([0.0, 0.0, 0.0, 1.0]),
    "uniforme": np.array([0.25, 0.25, 0.25, 0.25]),
}


def carregar_matriz(nome):
    # Lê a matriz de probabilidades gerada na Etapa 2.
    caminho = PASTA_TABELAS / f"e2_{nome}_probabilidades.csv"

    if not caminho.exists():
        raise FileNotFoundError(
            f"Matriz de transição não encontrada: {caminho}"
        )

    tabela = pd.read_csv(
        caminho,
        index_col=0
    )

    # Confere a identificação das linhas e colunas.
    linhas = [f"origem_{i}" for i in NIVEIS]
    colunas = [f"destino_{i}" for i in NIVEIS]

    if list(tabela.index) != linhas:
        raise ValueError(
            f"Linhas incorretas na matriz de {nome}."
        )

    if list(tabela.columns) != colunas:
        raise ValueError(
            f"Colunas incorretas na matriz de {nome}."
        )

    P = tabela.to_numpy(dtype=float)

    # Confere se a matriz é estocástica.
    if not verificar_estocastica(P):
        raise ValueError(
            f"A matriz de {nome} não é estocástica."
        )

    return P


def estacionaria_sistema(P):
    # Método 1: resolve o sistema linear com normalização.
    pi = distribuicao_estacionaria(P)

    return pi


def estacionaria_potencias(P, distribuicao_inicial):
    # Método 2: calcula pi(k+1) = pi(k) * P.
    pi_atual = np.asarray(
        distribuicao_inicial,
        dtype=float
    ).copy()

    if pi_atual.shape != (4,):
        raise ValueError(
            "A distribuição inicial deve possuir quatro valores."
        )

    if np.any(pi_atual < 0):
        raise ValueError(
            "A distribuição inicial possui valores negativos."
        )

    if not np.isclose(
        pi_atual.sum(),
        1,
        atol=1e-12,
        rtol=0
    ):
        raise ValueError(
            "A distribuição inicial deve somar 1."
        )

    # Executa as iterações explicitamente.
    for iteracao in range(1, MAX_ITERACOES + 1):

        pi_proxima = pi_atual @ P

        # Calcula a distância L1 entre iterações consecutivas.
        diferenca = np.linalg.norm(
            pi_proxima - pi_atual,
            ord=1
        )

        if diferenca < TOLERANCIA_POTENCIAS:

            # Confere se a distribuição é estacionária.
            residuo = np.linalg.norm(
                pi_proxima @ P - pi_proxima,
                ord=1
            )

            if residuo > TOLERANCIA_COMPARACAO:
                raise ValueError(
                    "O método das potências não atingiu "
                    "uma distribuição estacionária válida."
                )

            return pi_proxima, iteracao, diferenca

        pi_atual = pi_proxima

    raise RuntimeError(
        "O método das potências não convergiu "
        f"após {MAX_ITERACOES} iterações."
    )


def estacionaria_autovetor(P):
    # Método 3: calcula os autovetores de P transposta.
    autovalores, autovetores = linalg.eig(P.T)

    # Localiza o autovalor mais próximo de 1.
    indice = np.argmin(
        np.abs(autovalores - 1)
    )

    autovalor = autovalores[indice]

    if not np.isclose(
        autovalor,
        1,
        atol=1e-10,
        rtol=0
    ):
        raise ValueError(
            "Não foi encontrado autovalor igual a 1."
        )

    # Obtém o autovetor associado ao autovalor 1.
    vetor = autovetores[:, indice]

    # Remove pequenos erros imaginários numéricos.
    vetor = np.real_if_close(
        vetor,
        tol=1000
    )

    if np.iscomplexobj(vetor):
        raise ValueError(
            "O autovetor estacionário apresentou "
            "valores complexos inesperados."
        )

    vetor = np.asarray(vetor, dtype=float)

    # O sinal do autovetor pode ser invertido.
    if vetor.sum() < 0:
        vetor = -vetor

    soma = vetor.sum()

    if np.isclose(soma, 0, atol=1e-12):
        raise ValueError(
            "Não foi possível normalizar o autovetor."
        )

    # Normaliza para que a soma seja igual a 1.
    pi = vetor / soma

    if np.any(pi < -1e-10):
        raise ValueError(
            "O autovetor gerou probabilidades negativas."
        )

    # Corrige apenas pequenos erros numéricos negativos.
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()

    # Confere a condição estacionária.
    residuo = np.linalg.norm(
        pi @ P - pi,
        ord=1
    )

    if residuo > TOLERANCIA_COMPARACAO:
        raise ValueError(
            "O autovetor não satisfaz pi * P = pi."
        )

    return pi


def comparar_distribuicoes(pi_sistema, pi_potencias, pi_autovetor):
    # Compara os três métodos usando a norma L1.
    diferenca_sistema_potencias = np.linalg.norm(
        pi_sistema - pi_potencias,
        ord=1
    )

    diferenca_sistema_autovetor = np.linalg.norm(
        pi_sistema - pi_autovetor,
        ord=1
    )

    diferenca_potencias_autovetor = np.linalg.norm(
        pi_potencias - pi_autovetor,
        ord=1
    )

    maior_diferenca = max(
        diferenca_sistema_potencias,
        diferenca_sistema_autovetor,
        diferenca_potencias_autovetor
    )

    # Exige concordância entre os três métodos.
    if maior_diferenca > TOLERANCIA_COMPARACAO:
        raise ValueError(
            "Os métodos de distribuição estacionária "
            "não concordam dentro da tolerância de 1e-10."
        )

    return {
        "sistema_potencias": diferenca_sistema_potencias,
        "sistema_autovetor": diferenca_sistema_autovetor,
        "potencias_autovetor": diferenca_potencias_autovetor,
        "maior_diferenca": maior_diferenca,
    }


def calcular_residuo(P, pi):
    # Calcula a norma L1 de pi * P - pi.
    residuo = np.linalg.norm(
        pi @ P - pi,
        ord=1
    )

    return float(residuo)


def verificar_primitividade(P):
    # Procura o menor k para o qual P^k é positiva.
    n = P.shape[0]

    # Limite de Wielandt: (n - 1)^2 + 1.
    limite_wielandt = (n - 1) ** 2 + 1

    potencia = np.eye(n)

    for k in range(1, limite_wielandt + 1):

        # Calcula as potências por multiplicações sucessivas.
        potencia = potencia @ P

        # Verifica se todos os elementos são positivos.
        if np.all(potencia > 0):

            return {
                "primitiva": True,
                "menor_k": k,
                "limite_wielandt": limite_wielandt,
                "dentro_limite": k <= limite_wielandt,
            }

    # Se nenhuma potência até o limite for positiva,
    # a matriz não é primitiva.
    return {
        "primitiva": False,
        "menor_k": None,
        "limite_wielandt": limite_wielandt,
        "dentro_limite": False,
    }


def calcular_lambda2(P):
    # Calcula os autovalores da matriz.
    autovalores = calcular_autovalores(P)

    # Ordena os módulos em ordem decrescente.
    modulos = np.sort(
        np.abs(autovalores)
    )[::-1]

    # O primeiro módulo corresponde ao autovalor 1.
    lambda2 = float(modulos[1])

    return lambda2


def frequencia_empirica(codigos=None):
    # Calcula a frequência de ocupação de cada nível.
    arquivos = sorted(PASTA_DADOS.glob("*.csv"))

    if codigos is not None:

        arquivos = [
            PASTA_DADOS / f"{codigo}.csv"
            for codigo in codigos
        ]

    else:
        # O agregado deve utilizar os 78 municípios.
        if len(arquivos) != 78:
            raise ValueError(
                f"Esperados 78 arquivos CSV. "
                f"Encontrados: {len(arquivos)}."
            )

        for arquivo in arquivos:
            codigo = arquivo.stem

            if (
                len(codigo) != 7
                or not codigo.isdigit()
                or not codigo.startswith("32")
            ):
                raise ValueError(
                    f"Código IBGE inválido: {codigo}"
                )

    calendario = gerar_semanas_esperadas()

    contagens = np.zeros(4, dtype=int)

    total_validos = 0

    for arquivo in arquivos:

        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {arquivo}"
            )

        # Reaproveita a leitura da Etapa 1.
        dados = ler_dados(arquivo)

        # Confere a coerência do calendário epidemiológico.
        conferir_calendario(
            dados,
            calendario,
            arquivo.name
        )

        # Exclui semanas duplicadas da análise.
        duplicadas = dados["data_iniSE"].duplicated(
            keep=False
        )

        dados = dados.loc[~duplicadas].copy()

        # Considera somente níveis válidos.
        dados = dados[
            dados["nivel"].isin(NIVEIS)
        ]

        # Conta as ocorrências de cada nível.
        for nivel in NIVEIS:

            quantidade = int(
                (dados["nivel"] == nivel).sum()
            )

            contagens[nivel - 1] += quantidade

            total_validos += quantidade

    if total_validos == 0:
        raise ValueError(
            "Não existem semanas com níveis válidos."
        )

    # Converte contagens em frequências relativas.
    frequencias = contagens / total_validos

    if not np.isclose(
        frequencias.sum(),
        1,
        atol=1e-12,
        rtol=0
    ):
        raise ValueError(
            "As frequências empíricas não somam 1."
        )

    return frequencias, contagens, total_validos


def analisar_localidade(nome, codigos=None):
    # Executa os cálculos da Etapa 3 para uma localidade.
    print()
    print("=" * 60)
    print(f"ANÁLISE: {nome.upper()}")
    print("=" * 60)

    P = carregar_matriz(nome)

    # Método 1: sistema linear.
    pi_sistema = estacionaria_sistema(P)

    # Método 3: autovetor à esquerda.
    pi_autovetor = estacionaria_autovetor(P)

    resultados_potencias = []
    resultados_comparacoes = []

    # Método 2: três distribuições iniciais.
    for origem, distribuicao in DISTRIBUICOES_INICIAIS.items():

        pi_potencias, iteracoes, diferenca = estacionaria_potencias(
            P,
            distribuicao
        )

        comparacao = comparar_distribuicoes(
            pi_sistema,
            pi_potencias,
            pi_autovetor
        )

        residuo_potencias = calcular_residuo(
            P,
            pi_potencias
        )

        resultados_potencias.append({
            "localidade": nome,
            "distribuicao_inicial": origem,
            "iteracoes": iteracoes,
            "distancia_final_l1": diferenca,
            "residuo_l1": residuo_potencias,
            "pi_1": pi_potencias[0],
            "pi_2": pi_potencias[1],
            "pi_3": pi_potencias[2],
            "pi_4": pi_potencias[3],
        })

        resultados_comparacoes.append({
            "localidade": nome,
            "distribuicao_inicial": origem,
            **comparacao,
        })

    # Calcula o resíduo do sistema linear.
    residuo = calcular_residuo(
        P,
        pi_sistema
    )

    if residuo > TOLERANCIA_COMPARACAO:
        raise ValueError(
            f"Resíduo estacionário elevado em {nome}."
        )

    # Verifica a primitividade.
    primitividade = verificar_primitividade(P)

    # Calcula o segundo maior módulo dos autovalores.
    lambda2 = calcular_lambda2(P)

    # Calcula a frequência empírica dos níveis.
    frequencias, contagens, total_validos = frequencia_empirica(
        codigos
    )

    # Calcula a diferença entre frequência e distribuição estacionária.
    diferenca_empirica = np.linalg.norm(
        pi_sistema - frequencias,
        ord=1
    )

    # Registra as distribuições estacionárias.
    distribuicoes = []

    for indice, nivel in enumerate(NIVEIS):

        distribuicoes.append({
            "localidade": nome,
            "nivel": nivel,
            "pi_sistema": pi_sistema[indice],
            "pi_autovetor": pi_autovetor[indice],
            "frequencia_empirica": frequencias[indice],
            "contagem_empirica": contagens[indice],
            "diferenca_absoluta": abs(
                pi_sistema[indice] - frequencias[indice]
            ),
        })

    resumo = {
        "localidade": nome,
        "residuo_l1": residuo,
        "lambda2_modulo": lambda2,
        "primitiva": primitividade["primitiva"],
        "menor_k": primitividade["menor_k"],
        "limite_wielandt": primitividade["limite_wielandt"],
        "dentro_limite": primitividade["dentro_limite"],
        "semanas_validas": total_validos,
        "diferenca_empirica_l1": diferenca_empirica,
    }

    # Imprime os resultados no terminal.
    print()
    print("Distribuição estacionária:")
    print(np.round(pi_sistema, 6))

    print()
    print("Resíduo ||piP - pi||1:")
    print(f"{residuo:.12e}")

    print()
    print("Método das potências:")

    for resultado in resultados_potencias:

        print(
            f"  {resultado['distribuicao_inicial']}: "
            f"{resultado['iteracoes']} iterações"
        )

    print()
    print(f"|lambda2| = {lambda2:.6f}")

    print()
    print("Primitividade:")
    print(f"  Primitiva: {primitividade['primitiva']}")
    print(f"  Menor k: {primitividade['menor_k']}")
    print(
        f"  Limite de Wielandt: "
        f"{primitividade['limite_wielandt']}"
    )

    print()
    print("Frequência empírica:")
    print(np.round(frequencias, 6))

    print()
    print(
        "Diferença estacionária x empírica (L1): "
        f"{diferenca_empirica:.6f}"
    )

    print()
    print("Concordância entre os três métodos: OK")

    return (
        resumo,
        distribuicoes,
        resultados_potencias,
        resultados_comparacoes,
    )


def executar():
    # Executa a Etapa 3 para as três matrizes.
    PASTA_TABELAS.mkdir(
        parents=True,
        exist_ok=True
    )

    localidades = [
        ("serra", [SERRA]),
        ("vila_velha", [VILA_VELHA]),
        ("es_agregado", None),
    ]

    resumos = []
    distribuicoes = []
    potencias = []
    comparacoes = []

    for nome, codigos in localidades:

        (
            resumo,
            dist,
            pot,
            comp
        ) = analisar_localidade(
            nome,
            codigos
        )

        resumos.append(resumo)
        distribuicoes.extend(dist)
        potencias.extend(pot)
        comparacoes.extend(comp)

    # Salva o resumo geral.
    pd.DataFrame(resumos).to_csv(
        PASTA_TABELAS / "e3_resumo.csv",
        index=False,
        float_format="%.12g"
    )

    # Salva as distribuições e frequências empíricas.
    pd.DataFrame(distribuicoes).to_csv(
        PASTA_TABELAS / "e3_distribuicoes.csv",
        index=False,
        float_format="%.12g"
    )

    # Salva os resultados do método das potências.
    pd.DataFrame(potencias).to_csv(
        PASTA_TABELAS / "e3_potencias.csv",
        index=False,
        float_format="%.12g"
    )

    # Salva as comparações entre os métodos.
    pd.DataFrame(comparacoes).to_csv(
        PASTA_TABELAS / "e3_comparacoes.csv",
        index=False,
        float_format="%.12g"
    )

    print()
    print("=" * 60)
    print("ETAPA 3 CONCLUÍDA")
    print("=" * 60)

    print()
    print("Arquivos gerados:")
    print("results/tabelas/e3_resumo.csv")
    print("results/tabelas/e3_distribuicoes.csv")
    print("results/tabelas/e3_potencias.csv")
    print("results/tabelas/e3_comparacoes.csv")


if __name__ == "__main__":
    executar()


# [/IA]