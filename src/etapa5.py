# [IA] ChatGPT – 26/11/2026 – Etapa 5: permanência e primeira passagem

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

from src.etapa1 import (
    ler_dados,
    gerar_semanas_esperadas,
    conferir_calendario,
)

from src.etapa3 import (
    carregar_matriz,
    estacionaria_sistema,
)


# Caminhos do projeto
RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data" / "raw"
PASTA_TABELAS = RAIZ / "results" / "tabelas"
PASTA_FIGURAS = RAIZ / "results" / "figuras"

# Municípios
SERRA = "3205002"
VILA_VELHA = "3205200"

LOCALIDADES = {
    "serra": [SERRA],
    "vila_velha": [VILA_VELHA],
    "es_agregado": None,
}

NOMES = {
    "serra": "Serra",
    "vila_velha": "Vila Velha",
    "es_agregado": "Espírito Santo agregado",
}

NIVEIS = [1, 2, 3, 4]

TOLERANCIA = 1e-10

# Matriz de referência do professor
P_REFERENCIA = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])

# Valores de conferência da Tabela 1
M14_REFERENCIA = 46.873065
M41_REFERENCIA = 17.035647


def permanencia_teorica(P):
    # Calcula 1 / (1 - pii) para os quatro níveis.

    permanencias = []

    for i in range(4):

        pii = P[i, i]

        if pii >= 1:
            permanencia = np.inf
        else:
            permanencia = 1 / (1 - pii)

        permanencias.append(permanencia)

    return np.array(permanencias)


def preparar_segmentos(caminho, calendario):
    # Lê os dados originais sem modificá-los.
    dados = ler_dados(caminho)

    conferir_calendario(
        dados,
        calendario,
        caminho.name
    )

    dados = dados.sort_values(
        "data_iniSE"
    ).reset_index(drop=True)

    # Identifica todas as ocorrências de semanas duplicadas.
    duplicadas = dados["data_iniSE"].duplicated(
        keep=False
    )

    # Remove registros ambíguos sem unir as semanas vizinhas.
    dados = dados.loc[~duplicadas].copy()

    dados = dados.sort_values(
        "data_iniSE"
    ).reset_index(drop=True)

    segmentos = []
    segmento_atual = []

    data_anterior = None

    for _, linha in dados.iterrows():

        data = linha["data_iniSE"]
        nivel = linha["nivel"]

        # Níveis ausentes ou inválidos interrompem a sequência.
        if nivel not in NIVEIS:

            if segmento_atual:
                segmentos.append(segmento_atual)

            segmento_atual = []
            data_anterior = None

            continue

        nivel = int(nivel)

        # Somente semanas consecutivas pertencem ao mesmo segmento.
        consecutiva = (
            data_anterior is not None
            and data - data_anterior == pd.Timedelta(days=7)
        )

        if not consecutiva:

            if segmento_atual:
                segmentos.append(segmento_atual)

            segmento_atual = []

        segmento_atual.append(nivel)
        data_anterior = data

    if segmento_atual:
        segmentos.append(segmento_atual)

    return segmentos


def obter_arquivos(codigos):
    # Seleciona os municípios utilizados na análise.

    if codigos is None:

        arquivos = sorted(
            PASTA_DADOS.glob("*.csv")
        )

        if len(arquivos) != 78:
            raise ValueError(
                "O ES agregado exige os 78 arquivos CSV."
            )

    else:

        arquivos = [
            PASTA_DADOS / f"{codigo}.csv"
            for codigo in codigos
        ]

    for arquivo in arquivos:

        if not arquivo.is_file():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {arquivo}"
            )

    return arquivos


def carregar_segmentos(codigos):
    # Reúne os segmentos válidos dos municípios.

    calendario = gerar_semanas_esperadas()

    arquivos = obter_arquivos(codigos)

    segmentos = []

    for arquivo in arquivos:

        segmentos_municipio = preparar_segmentos(
            arquivo,
            calendario
        )

        # Não conecta séries de municípios diferentes.
        segmentos.extend(segmentos_municipio)

    return segmentos


def duracoes_observadas(segmentos):
    # Identifica períodos completos em cada nível.
    # Exclui períodos censurados no início, fim ou lacuna.

    duracoes = {
        nivel: []
        for nivel in NIVEIS
    }

    censuradas = {
        nivel: 0
        for nivel in NIVEIS
    }

    for segmento in segmentos:

        inicio = 0
        tamanho = len(segmento)

        while inicio < tamanho:

            nivel = segmento[inicio]
            fim = inicio + 1

            # Conta semanas consecutivas no mesmo nível.
            while (
                fim < tamanho
                and segmento[fim] == nivel
            ):
                fim += 1

            duracao = fim - inicio

            # Uma permanência só é completa se possuir
            # entrada e saída observadas no mesmo segmento.
            censurada = (
                inicio == 0
                or fim == tamanho
            )

            if censurada:
                censuradas[nivel] += 1
            else:
                duracoes[nivel].append(duracao)

            inicio = fim

    return duracoes, censuradas


def calcular_permanencias(segmentos, P):
    # Compara permanências teóricas e observadas.

    teoricas = permanencia_teorica(P)

    duracoes, censuradas = duracoes_observadas(
        segmentos
    )

    resultados = []

    for nivel in NIVEIS:

        observadas = np.array(
            duracoes[nivel],
            dtype=float
        )

        quantidade = len(observadas)

        if quantidade > 0:
            media = float(np.mean(observadas))
            desvio = float(np.std(observadas))
            diferenca = media - teoricas[nivel - 1]
        else:
            media = np.nan
            desvio = np.nan
            diferenca = np.nan

        resultados.append({
            "nivel": nivel,
            "p_ii": P[nivel - 1, nivel - 1],
            "permanencia_teorica": teoricas[nivel - 1],
            "permanencia_empirica": media,
            "desvio_empirico": desvio,
            "quantidade_periodos_completos": quantidade,
            "periodos_censurados": censuradas[nivel],
            "diferenca_empirica_teorica": diferenca,
        })

    return pd.DataFrame(resultados), duracoes


def primeira_passagem_sistemas(P, pi):
    # Calcula M resolvendo quatro sistemas lineares 3x3.

    M = np.zeros((4, 4))

    identidade = np.eye(4)

    for destino in range(4):

        # Seleciona os três estados diferentes do destino.
        transientes = [
            i for i in range(4)
            if i != destino
        ]

        Q = P[np.ix_(transientes, transientes)]

        A = identidade[:3, :3] - Q

        b = np.ones(3)

        # Resolve o sistema dos tempos até o destino.
        tempos = linalg.solve(A, b)

        for indice, origem in enumerate(transientes):

            M[origem, destino] = tempos[indice]

        # Tempo médio de retorno ao próprio estado.
        M[destino, destino] = 1 / pi[destino]

    return M


def verificar_kac(P, pi, M):
    # Confere a fórmula de Kac por uma equação independente.

    registros = []

    for j in range(4):

        # Tempo de retorno usando a condição no primeiro passo.
        retorno = 1.0

        for k in range(4):

            if k != j:
                retorno += P[j, k] * M[k, j]

        kac = 1 / pi[j]

        diferenca = abs(retorno - kac)

        if diferenca > TOLERANCIA:
            raise ValueError(
                f"Fórmula de Kac não satisfeita no nível {j + 1}."
            )

        registros.append({
            "nivel": j + 1,
            "retorno_primeiro_passo": retorno,
            "retorno_kac": kac,
            "diferenca_absoluta": diferenca,
            "verificacao": True,
        })

    return pd.DataFrame(registros)


def primeira_passagem_fundamental(P, pi):
    # Calcula a matriz fundamental Z.
    # Z = (I - P + 1*pi)^-1

    n = P.shape[0]

    identidade = np.eye(n)

    matriz = (
        identidade
        - P
        + np.outer(np.ones(n), pi)
    )

    # Resolve o sistema em vez de calcular a inversa diretamente.
    Z = linalg.solve(
        matriz,
        identidade
    )

    M = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            if i == j:
                M[i, j] = 1 / pi[j]
            else:
                M[i, j] = (
                    Z[j, j] - Z[i, j]
                ) / pi[j]

    return M, Z

# Função corrigida pelos alunos
def comparar_matrizes_passagem(M_sistemas, M_fundamental):
    # Registra as discrepâncias entre os dois métodos independentes.

    diferencas = np.abs(M_sistemas - M_fundamental)

    maior_diferenca = float(np.max(diferencas))

    norma_l1 = float(
        np.linalg.norm(
            M_sistemas - M_fundamental,
            ord=1
        )
    )

    escala = max(
        float(np.max(np.abs(M_sistemas))),
        float(np.max(np.abs(M_fundamental))),
        1.0
    )

    erro_relativo = maior_diferenca / escala

    # Não interrompe a análise por diferenças de arredondamento.
    # A discrepância absoluta e relativa será registrada.
    if erro_relativo > 1e-10:
        raise ValueError(
            "Os métodos apresentam divergência relativa "
            "superior à tolerância numérica."
        )

    return {
        "maior_diferenca": maior_diferenca,
        "norma_matricial_l1": norma_l1,
        "erro_relativo": erro_relativo,
    }


def tempos_empiricos_passagem(segmentos, origem, destino):
    # Calcula o tempo observado até atingir o destino.
    # Cada ocorrência da origem inicia uma observação.
    # Episódios sem chegada ao destino são censurados.

    tempos = []
    censurados = 0

    for segmento in segmentos:

        tamanho = len(segmento)

        for inicio in range(tamanho):

            if segmento[inicio] != origem:
                continue

            encontrou = False

            for fim in range(inicio + 1, tamanho):

                if segmento[fim] == destino:

                    tempos.append(fim - inicio)

                    encontrou = True
                    break

            if not encontrou:
                censurados += 1

    return tempos, censurados


def comparar_primeira_passagem_empirica(segmentos, M):
    # Compara os tempos de 1 para 4 e de 4 para 1.

    pares = [
        (1, 4),
        (4, 1),
    ]

    registros = []

    for origem, destino in pares:

        tempos, censurados = tempos_empiricos_passagem(
            segmentos,
            origem,
            destino
        )

        if tempos:
            media_empirica = float(np.mean(tempos))
        else:
            media_empirica = np.nan

        media_teorica = M[origem - 1, destino - 1]

        registros.append({
            "origem": origem,
            "destino": destino,
            "tempo_teorico": media_teorica,
            "tempo_empirico": media_empirica,
            "observacoes_completas": len(tempos),
            "observacoes_censuradas": censurados,
            "diferenca_empirica_teorica": (
                media_empirica - media_teorica
            ),
        })

    return pd.DataFrame(registros)


def grafico_nivel4(nome, duracoes, P):
    # Compara as durações observadas com a distribuição geométrica.

    observadas = duracoes[4]

    if not observadas:
        print(
            f"Sem períodos completos no nível 4 em {nome}. "
            "Histograma não gerado."
        )
        return False

    p44 = P[3, 3]

    prob_saida = 1 - p44

    if prob_saida <= 0:
        raise ValueError(
            "Não é possível calcular a geométrica com p44 = 1."
        )

    max_duracao = max(observadas)

    semanas = np.arange(
        1,
        max_duracao + 1
    )

    # Probabilidade geométrica de permanecer k semanas.
    probabilidades = (
        prob_saida
        * p44 ** (semanas - 1)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # Histograma com frequência relativa.
    pesos = np.ones(len(observadas)) / len(observadas)

    ax.hist(
        observadas,
        bins=np.arange(0.5, max_duracao + 1.5, 1),
        weights=pesos,
        color="lightgray",
        edgecolor="black",
        label="Durações observadas",
    )

    # Distribuição geométrica prevista pelo modelo.
    ax.plot(
        semanas,
        probabilidades,
        color="black",
        linestyle="--",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Distribuição geométrica",
    )

    ax.set_title(
        f"Permanência no alerta 4 - {NOMES[nome]}"
    )

    ax.set_xlabel(
        "Duração do período (semanas)"
    )

    ax.set_ylabel(
        "Probabilidade / frequência relativa"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PASTA_FIGURAS / f"e5_{nome}_nivel4_geometrica.png",
        dpi=300
    )

    plt.close(fig)

    return True


def salvar_matriz(nome, descricao, matriz):
    # Salva matrizes com identificação dos níveis.

    tabela = pd.DataFrame(
        matriz,
        index=[f"origem_{i}" for i in NIVEIS],
        columns=[f"destino_{i}" for i in NIVEIS],
    )

    tabela.to_csv(
        PASTA_TABELAS / f"e5_{nome}_{descricao}.csv",
        float_format="%.15g"
    )


def conferir_referencia():
    # Confere os valores da Tabela 1 do professor.

    pi = estacionaria_sistema(
        P_REFERENCIA
    )

    M_sistemas = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    M_fundamental, Z = primeira_passagem_fundamental(
        P_REFERENCIA,
        pi
    )

    verificacao = comparar_matrizes_passagem(
        M_sistemas,
        M_fundamental
    )

    kac = verificar_kac(
        P_REFERENCIA,
        pi,
        M_sistemas
    )

    m14 = M_sistemas[0, 3]
    m41 = M_sistemas[3, 0]

    erro_m14 = abs(
        m14 - M14_REFERENCIA
    )

    erro_m41 = abs(
        m41 - M41_REFERENCIA
    )

    # Valores da tabela estão arredondados a seis casas.
    if erro_m14 > 1e-6 or erro_m41 > 1e-6:
        raise ValueError(
            "Os valores de primeira passagem da Tabela 1 "
            "não foram reproduzidos."
        )

    salvar_matriz(
        "referencia",
        "M_sistemas",
        M_sistemas
    )

    salvar_matriz(
        "referencia",
        "M_fundamental",
        M_fundamental
    )

    salvar_matriz(
        "referencia",
        "Z",
        Z
    )

    kac.to_csv(
        PASTA_TABELAS / "e5_referencia_kac.csv",
        index=False
    )

    resultado = pd.DataFrame([
        {
            "grandeza": "m14",
            "valor_calculado": m14,
            "valor_referencia": M14_REFERENCIA,
            "erro_absoluto": erro_m14,
        },
        {
            "grandeza": "m41",
            "valor_calculado": m41,
            "valor_referencia": M41_REFERENCIA,
            "erro_absoluto": erro_m41,
        },
    ])

    resultado.to_csv(
        PASTA_TABELAS / "e5_conferencia_tabela1.csv",
        index=False,
        float_format="%.15g"
    )

    print()
    print("CONFERÊNCIA DA TABELA 1")
    print(f"m14 calculado: {m14:.6f}")
    print(f"m41 calculado: {m41:.6f}")
    print(
        "Maior discrepância entre métodos: "
        f"{verificacao['maior_diferenca']:.12e}"
    )


def analisar_localidade(nome, codigos):
    # Executa os cálculos da Etapa 5 para uma localidade.

    P = carregar_matriz(nome)

    somas_linhas = P.sum(axis=1, keepdims=True)
    P = P / somas_linhas

    pi = estacionaria_sistema(P)

    segmentos = carregar_segmentos(codigos)

    # Calcula as permanências teóricas e empíricas.
    permanencias, duracoes = calcular_permanencias(
        segmentos,
        P
    )

    # Primeiro método: quatro sistemas lineares 3x3.
    M_sistemas = primeira_passagem_sistemas(
        P,
        pi
    )

    # Verifica a fórmula de Kac.
    kac = verificar_kac(
        P,
        pi,
        M_sistemas
    )

    # Segundo método: matriz fundamental.
    M_fundamental, Z = primeira_passagem_fundamental(
        P,
        pi
    )

    # Compara os métodos independentes.
    comparacao = comparar_matrizes_passagem(
        M_sistemas,
        M_fundamental
    )

    # Compara tempos de primeira passagem teóricos e empíricos.
    passagem_empirica = comparar_primeira_passagem_empirica(
        segmentos,
        M_sistemas
    )

    # Salva os resultados.
    permanencias.insert(
        0,
        "localidade",
        nome
    )

    permanencias.to_csv(
        PASTA_TABELAS / f"e5_{nome}_permanencias.csv",
        index=False,
        float_format="%.15g"
    )

    # Salva as durações individuais dos períodos completos.
    registros_duracoes = []

    for nivel, valores in duracoes.items():

        for duracao in valores:

            registros_duracoes.append({
                "localidade": nome,
                "nivel": nivel,
                "duracao_semanas": duracao,
            })

    pd.DataFrame(
        registros_duracoes,
        columns=[
            "localidade",
            "nivel",
            "duracao_semanas",
        ]
    ).to_csv(
        PASTA_TABELAS / f"e5_{nome}_duracoes.csv",
        index=False
    )

    salvar_matriz(
        nome,
        "M_sistemas",
        M_sistemas
    )

    salvar_matriz(
        nome,
        "M_fundamental",
        M_fundamental
    )

    salvar_matriz(
        nome,
        "Z",
        Z
    )

    salvar_matriz(
        nome,
        "diferencas_M",
        np.abs(M_sistemas - M_fundamental)
    )

    kac.to_csv(
        PASTA_TABELAS / f"e5_{nome}_kac.csv",
        index=False,
        float_format="%.15g"
    )

    passagem_empirica.insert(
        0,
        "localidade",
        nome
    )

    passagem_empirica.to_csv(
        PASTA_TABELAS / f"e5_{nome}_primeira_passagem_empirica.csv",
        index=False,
        float_format="%.15g"
    )

    # Gera o histograma do nível vermelho.
    grafico_gerado = grafico_nivel4(
        nome,
        duracoes,
        P
    )

    resumo = {
        "localidade": nome,
        "m14": M_sistemas[0, 3],
        "m41": M_sistemas[3, 0],
        "maior_diferenca_M": comparacao["maior_diferenca"],
        "norma_matricial_l1": comparacao["norma_matricial_l1"],
        "maior_erro_kac": kac["diferenca_absoluta"].max(),
        "periodos_completos_nivel4": len(duracoes[4]),
        "grafico_nivel4_gerado": grafico_gerado,
    }

    # Imprime o resumo da localidade.
    print()
    print("=" * 65)
    print(f"ETAPA 5 - {NOMES[nome].upper()}")
    print("=" * 65)

    print()
    print("Permanências médias:")
    print(
        permanencias[
            [
                "nivel",
                "permanencia_teorica",
                "permanencia_empirica",
                "quantidade_periodos_completos",
            ]
        ].to_string(index=False)
    )

    print()
    print("Matriz de primeira passagem:")
    print(
        np.round(M_sistemas, 6)
    )

    print()
    print(f"m14: {M_sistemas[0, 3]:.6f} semanas")
    print(f"m41: {M_sistemas[3, 0]:.6f} semanas")

    print()
    print(
        "Maior diferença entre métodos: "
        f"{comparacao['maior_diferenca']:.12e}"
    )

    print(
        "Maior erro da fórmula de Kac: "
        f"{resumo['maior_erro_kac']:.12e}"
    )

    print()
    print("Primeira passagem teórica x empírica:")
    print(
        passagem_empirica.to_string(index=False)
    )

    return resumo


def executar():
    # Executa a Etapa 5 para as três localidades.

    PASTA_TABELAS.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_FIGURAS.mkdir(
        parents=True,
        exist_ok=True
    )

    # Confere primeiro a matriz de referência do professor.
    conferir_referencia()

    resumos = []

    for nome, codigos in LOCALIDADES.items():

        resultado = analisar_localidade(
            nome,
            codigos
        )

        resumos.append(resultado)

    pd.DataFrame(resumos).to_csv(
        PASTA_TABELAS / "e5_resumo.csv",
        index=False,
        float_format="%.15g"
    )

    print()
    print("=" * 65)
    print("ETAPA 5 CONCLUÍDA")
    print("=" * 65)

    print("Resultados salvos em:")
    print("results/tabelas/")
    print("results/figuras/")


if __name__ == "__main__":
    executar()


# [/IA]