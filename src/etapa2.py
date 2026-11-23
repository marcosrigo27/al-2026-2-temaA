# [IA] ChatGPT – 23/11/2026 – Etapa 2: matrizes e suavização

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.markov import (
    matriz_contagens,
    matriz_transicao,
    verificar_estocastica,
    distribuicao_estacionaria,
)


RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data" / "raw"
PASTA_TABELAS = RAIZ / "results" / "tabelas"
PASTA_FIGURAS = RAIZ / "results" / "figuras"

SERRA = "3205002"
VILA_VELHA = "3205200"
ALFA = 0.5
TOLERANCIA = 1e-12
NIVEIS = [1, 2, 3, 4]


def extrair_transicoes(caminho):
    # Usa a data inicial da semana para conferir a continuidade temporal.
    dados = pd.read_csv(caminho, dtype={"SE": "string"})

    exigidas = {"data_iniSE", "SE", "nivel"}

    if not exigidas.issubset(dados.columns):
        raise ValueError(
            f"Colunas ausentes em {caminho.name}: "
            f"{exigidas - set(dados.columns)}"
        )

    dados["data_iniSE"] = pd.to_datetime(
        dados["data_iniSE"],
        errors="coerce"
    )

    if dados["data_iniSE"].isna().any():
        raise ValueError(
            f"Data epidemiológica inválida em {caminho.name}."
        )

    # Não escolhe arbitrariamente registros duplicados.
    if (
        dados["data_iniSE"].duplicated().any()
        or dados["SE"].duplicated().any()
    ):
        raise ValueError(
            f"Semanas duplicadas em {caminho.name}; "
            "revisar os dados brutos."
        )

    dados = dados.sort_values("data_iniSE").reset_index(drop=True)

    niveis = pd.to_numeric(dados["nivel"], errors="coerce")
    datas = dados["data_iniSE"]

    # Uma transição só vale se as datas estiverem separadas por 7 dias.
    consecutivas = datas.diff().eq(pd.Timedelta(days=7))

    niveis_validos = niveis.isin(NIVEIS)

    pares_validos = (
        consecutivas
        & niveis_validos
        & niveis_validos.shift(1, fill_value=False)
    )

    origens = niveis.shift(1)[pares_validos].astype(int)
    destinos = niveis[pares_validos].astype(int)

    transicoes = list(zip(origens, destinos))

    return transicoes


def validar_matriz(P):
    # Verifica não negatividade e soma de cada linha igual a 1.
    if not verificar_estocastica(P, tolerancia=TOLERANCIA):
        raise ValueError(
            "A matriz de probabilidades não é estocástica."
        )

    if np.any(P < -TOLERANCIA):
        raise ValueError(
            "A matriz contém probabilidades negativas."
        )

    if not np.allclose(
        P.sum(axis=1),
        1.0,
        atol=TOLERANCIA,
        rtol=0
    ):
        raise ValueError(
            "As linhas da matriz não somam 1."
        )


def calcular_resultados(C, total_transicoes):
    # Confere se todas as transições foram contabilizadas.
    if int(C.sum()) != total_transicoes:
        raise ValueError(
            "Soma das contagens diferente das transições válidas."
        )

    # A função original rejeita linhas sem observações.
    P = matriz_transicao(C)

    validar_matriz(P)

    # Calcula os erros-padrão das probabilidades.
    total_por_estado = C.sum(axis=1, keepdims=True)

    erros = np.sqrt(
        P * (1 - P) / total_por_estado
    )

    # Aplica suavização de Laplace com alfa = 0,5.
    P_suavizada = (
        (C + ALFA) / (total_por_estado + 4 * ALFA)
    )

    validar_matriz(P_suavizada)

    # Calcula as distribuições estacionárias.
    pi = distribuicao_estacionaria(P)

    pi_suavizada = distribuicao_estacionaria(P_suavizada)

    # Calcula a diferença entre as distribuições pela norma L1.
    diferenca_l1 = float(
        np.sum(np.abs(pi - pi_suavizada))
    )

    return (
        P,
        erros,
        P_suavizada,
        pi,
        pi_suavizada,
        diferenca_l1
    )


def salvar_matriz(nome, sufixo, matriz):
    # As linhas representam a origem e as colunas o destino.
    tabela = pd.DataFrame(
        matriz,
        index=[f"origem_{i}" for i in NIVEIS],
        columns=[f"destino_{i}" for i in NIVEIS],
    )

    tabela.to_csv(
        PASTA_TABELAS / f"e2_{nome}_{sufixo}.csv",
        float_format="%.12g"
    )


def salvar_mapa_calor(nome, P):
    # Cria o mapa de calor em escala de cinza.
    fig, ax = plt.subplots(figsize=(6, 5))

    imagem = ax.imshow(
        P,
        cmap="Greys",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(range(4), NIVEIS)
    ax.set_yticks(range(4), NIVEIS)

    ax.set_xlabel("Nível de destino")
    ax.set_ylabel("Nível de origem")

    ax.set_title(
        f"Transições de alerta – {nome.replace('_', ' ').title()}"
    )

    # Mostra os valores das probabilidades em cada célula.
    for i in range(4):
        for j in range(4):

            cor = "white" if P[i, j] >= 0.55 else "black"

            ax.text(
                j,
                i,
                f"{P[i, j]:.3f}",
                ha="center",
                va="center",
                color=cor
            )

    fig.colorbar(
        imagem,
        ax=ax,
        label="Probabilidade"
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_FIGURAS / f"e2_{nome}_mapa_calor.png",
        dpi=200
    )

    plt.close(fig)


def executar():
    # Confere se existem os 78 arquivos CSV dos municípios.
    arquivos = sorted(PASTA_DADOS.glob("*.csv"))

    if len(arquivos) != 78:
        raise ValueError(
            f"Esperados 78 arquivos CSV; "
            f"encontrados {len(arquivos)}."
        )

    # Confere os municípios escolhidos.
    if not (PASTA_DADOS / f"{SERRA}.csv").is_file():
        raise FileNotFoundError(
            "CSV de Serra não encontrado."
        )

    if not (PASTA_DADOS / f"{VILA_VELHA}.csv").is_file():
        raise FileNotFoundError(
            "CSV de Vila Velha não encontrado."
        )

    # Cria as pastas de resultados.
    PASTA_TABELAS.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_FIGURAS.mkdir(
        parents=True,
        exist_ok=True
    )

    contagens = {}
    totais = {}

    agregado = np.zeros((4, 4), dtype=int)
    total_agregado = 0

    # Percorre os arquivos dos 78 municípios.
    for arquivo in arquivos:

        transicoes = extrair_transicoes(arquivo)

        C = matriz_contagens(transicoes)

        # Confere a quantidade de transições.
        if int(C.sum()) != len(transicoes):
            raise ValueError(
                f"Contagens inconsistentes em {arquivo.name}."
            )

        # Soma as contagens para formar o ES agregado.
        agregado += C

        total_agregado += len(transicoes)

        # Guarda separadamente Serra e Vila Velha.
        if arquivo.stem in (SERRA, VILA_VELHA):

            nome = (
                "serra"
                if arquivo.stem == SERRA
                else "vila_velha"
            )

            contagens[nome] = C
            totais[nome] = len(transicoes)

    # Adiciona os resultados agregados do Espírito Santo.
    contagens["es_agregado"] = agregado
    totais["es_agregado"] = total_agregado

    resumo = []
    distribuicoes = []

    # Calcula e salva os resultados das três localidades.
    for nome, C in contagens.items():

        (
            P,
            erros,
            P_suave,
            pi,
            pi_suave,
            distancia
        ) = calcular_resultados(
            C,
            totais[nome]
        )

        # Salva as matrizes em CSV.
        salvar_matriz(
            nome,
            "contagens",
            C
        )

        salvar_matriz(
            nome,
            "probabilidades",
            P
        )

        salvar_matriz(
            nome,
            "erros_padrao",
            erros
        )

        salvar_matriz(
            nome,
            "laplace",
            P_suave
        )

        # Gera o mapa de calor.
        salvar_mapa_calor(nome, P)

        # Registra o resumo dos resultados.
        resumo.append({
            "localidade": nome,
            "transicoes_validas": totais[nome],
            "soma_contagens": int(C.sum()),
            "alfa": ALFA,
            "diferenca_l1": distancia,
        })

        # Registra as distribuições estacionárias.
        for nivel, original, suave in zip(
            NIVEIS,
            pi,
            pi_suave
        ):

            distribuicoes.append({
                "localidade": nome,
                "nivel": nivel,
                "pi_original": original,
                "pi_laplace": suave,
            })

    # Salva o resumo geral.
    pd.DataFrame(resumo).to_csv(
        PASTA_TABELAS / "e2_resumo.csv",
        index=False
    )

    # Salva as distribuições estacionárias.
    pd.DataFrame(distribuicoes).to_csv(
        PASTA_TABELAS / "e2_distribuicoes_estacionarias.csv",
        index=False
    )

    print(
        "Etapa 2 concluída: CSVs e mapas de calor "
        "salvos em results/."
    )


if __name__ == "__main__":
    executar()


# [/IA]