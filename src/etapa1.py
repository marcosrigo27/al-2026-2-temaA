# [IA] ChatGPT – 24/11/2026 – Etapa 1: auditoria dos dados

import hashlib
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Caminhos do projeto
RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data" / "raw"
MANIFESTO = RAIZ / "data" / "MANIFEST.txt"

PASTA_RESULTADOS = RAIZ / "results"
PASTA_FIGURAS = PASTA_RESULTADOS / "figuras"

# Configurações
SERRA = "3205002"

ANO_INICIAL = 2015
ANO_FINAL = 2025

NIVEIS = [1, 2, 3, 4]

COLUNAS = [
    "data_iniSE",
    "SE",
    "nivel",
    "versao_modelo",
]


def calcular_hash(caminho):
    # Calcula o SHA-256 do arquivo original.
    sha256 = hashlib.sha256()

    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(65536), b""):
            sha256.update(bloco)

    return sha256.hexdigest()


def ler_manifesto():
    # Lê os registros de integridade dos arquivos.
    if not MANIFESTO.exists():
        raise FileNotFoundError(
            "Arquivo data/MANIFEST.txt não encontrado."
        )

    texto = MANIFESTO.read_text(encoding="utf-8")

    registros = {}
    registro = {}

    # Inclui uma linha vazia ao final para salvar o último registro.
    for linha in texto.splitlines() + [""]:

        linha = linha.strip()

        if not linha:
            if registro:
                nome = registro.get("Arquivo")

                if nome is None:
                    raise ValueError(
                        "Registro do manifesto sem nome do arquivo."
                    )

                nome = Path(nome).name

                if nome in registros:
                    raise ValueError(
                        f"Arquivo duplicado no manifesto: {nome}"
                    )

                registros[nome] = registro
                registro = {}

            continue

        if ":" not in linha:
            continue

        chave, valor = linha.split(":", 1)

        registro[chave.strip()] = valor.strip()

    return registros


def conferir_integridade(arquivos):
    # Compara os hashes dos CSVs com o manifesto.
    registros = ler_manifesto()

    if len(registros) != 78:
        raise ValueError(
            f"Manifesto possui {len(registros)} registros. "
            "Esperados: 78."
        )

    nomes_arquivos = {arquivo.name for arquivo in arquivos}

    if nomes_arquivos != set(registros):
        faltando = nomes_arquivos - set(registros)
        extras = set(registros) - nomes_arquivos

        raise ValueError(
            "Diferença entre os arquivos e o manifesto. "
            f"Sem registro: {sorted(faltando)}. "
            f"Registros extras: {sorted(extras)}."
        )

    for arquivo in arquivos:

        registro = registros[arquivo.name]

        hash_esperado = registro.get("SHA-256")

        if not hash_esperado:
            raise ValueError(
                f"Hash ausente no manifesto: {arquivo.name}"
            )

        hash_atual = calcular_hash(arquivo)

        if hash_atual.lower() != hash_esperado.lower():
            raise ValueError(
                f"Falha de integridade: {arquivo.name}"
            )

        # Confere também a quantidade de linhas registrada.
        linhas_esperadas = registro.get("Linhas")

        if linhas_esperadas is None:
            raise ValueError(
                f"Quantidade de linhas ausente: {arquivo.name}"
            )

        try:
            quantidade_esperada = int(linhas_esperadas)
        except ValueError:
            raise ValueError(
                f"Quantidade de linhas inválida: {arquivo.name}"
            )

        quantidade_atual = len(pd.read_csv(arquivo))

        if quantidade_atual != quantidade_esperada:
            raise ValueError(
                f"Quantidade de linhas diferente: {arquivo.name}"
            )

        print(f"Integridade confirmada: {arquivo.name}")

    print("Integridade dos 78 arquivos confirmada.")


def inicio_ano_epidemiologico(ano):
    # A semana epidemiológica começa no domingo.
    # A primeira semana é a que contém pelo menos 4 dias do ano.
    quarto_dia = date(ano, 1, 4)

    dias_desde_domingo = (quarto_dia.weekday() + 1) % 7

    return quarto_dia - timedelta(days=dias_desde_domingo)


def gerar_semanas_esperadas():
    # Gera as semanas epidemiológicas de 2015 até 2025.
    inicio = inicio_ano_epidemiologico(ANO_INICIAL)

    fim = inicio_ano_epidemiologico(ANO_FINAL + 1)

    semanas = []
    data_atual = inicio

    while data_atual < fim:

        for ano in range(ANO_INICIAL, ANO_FINAL + 1):

            inicio_ano = inicio_ano_epidemiologico(ano)
            inicio_proximo = inicio_ano_epidemiologico(ano + 1)

            if inicio_ano <= data_atual < inicio_proximo:

                numero_semana = (
                    (data_atual - inicio_ano).days // 7
                ) + 1

                semanas.append({
                    "data": pd.Timestamp(data_atual),
                    "ano": ano,
                    "semana": numero_semana,
                    "SE": ano * 100 + numero_semana,
                })

                break

        data_atual += timedelta(days=7)

    return pd.DataFrame(semanas)


def ler_dados(caminho):
    # Lê os dados sem alterar o CSV original.
    dados = pd.read_csv(
        caminho,
        dtype={"SE": "string"}
    )

    # Confere as colunas obrigatórias.
    faltando = set(COLUNAS) - set(dados.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes em {caminho.name}: "
            f"{sorted(faltando)}"
        )

    dados = dados[COLUNAS].copy()

    # Converte as datas epidemiológicas.
    dados["data_iniSE"] = pd.to_datetime(
        dados["data_iniSE"],
        errors="coerce"
    )

    if dados["data_iniSE"].isna().any():
        raise ValueError(
            f"Datas inválidas em {caminho.name}"
        )

    # Converte os níveis para números.
    dados["nivel"] = pd.to_numeric(
        dados["nivel"],
        errors="coerce"
    )

    # Converte a semana para número.
    dados["SE"] = pd.to_numeric(
        dados["SE"],
        errors="coerce"
    )

    return dados


def conferir_calendario(dados, semanas_esperadas, nome):
    # Confere se as datas e os códigos SE são coerentes.
    calendario = semanas_esperadas.set_index("data")["SE"]

    datas_validas = dados["data_iniSE"].isin(calendario.index)

    if not datas_validas.all():
        raise ValueError(
            f"Datas fora do calendário epidemiológico: {nome}"
        )

    se_esperada = dados["data_iniSE"].map(calendario)

    se_valida = dados["SE"].eq(se_esperada).fillna(False)

    if not se_valida.all():
        raise ValueError(
            f"Inconsistência entre SE e data_iniSE: {nome}"
        )


def contar_transicoes_validas(dados):
    # Não conta transições envolvendo semanas duplicadas.
    # Não conta transições atravessando lacunas.
    dados = dados.sort_values("data_iniSE").copy()

    datas_duplicadas = dados["data_iniSE"].duplicated(
        keep=False
    )

    dados = dados.loc[~datas_duplicadas].copy()

    dados = dados.reset_index(drop=True)

    if len(dados) < 2:
        return 0

    niveis_validos = dados["nivel"].isin(NIVEIS)

    semanas_consecutivas = (
        dados["data_iniSE"].diff()
        == pd.Timedelta(days=7)
    )

    pares_validos = (
        semanas_consecutivas
        & niveis_validos
        & niveis_validos.shift(1, fill_value=False)
    )

    return int(pares_validos.sum())


def analisar_municipio(caminho, semanas_esperadas):
    # Realiza a auditoria de um município.
    dados = ler_dados(caminho)

    conferir_calendario(
        dados,
        semanas_esperadas,
        caminho.name
    )

    # Quantidade de semanas distintas observadas.
    semanas_observadas = dados["data_iniSE"].nunique()

    semanas_previstas = len(semanas_esperadas)

    semanas_ausentes = (
        semanas_previstas - semanas_observadas
    )

    # Conta registros excedentes em semanas duplicadas.
    duplicatas = int(
        dados["data_iniSE"].duplicated().sum()
    )

    # Identifica todas as semanas com duplicidade.
    duplicadas = dados["data_iniSE"].duplicated(
        keep=False
    )

    # Exclui semanas ambíguas apenas da análise de níveis.
    dados_validacao = dados.loc[~duplicadas].copy()

    niveis_validos = dados_validacao["nivel"].isin(NIVEIS)

    semanas_validas = int(niveis_validos.sum())

    semanas_invalidas = int(
        (~niveis_validos).sum()
    )

    # Conta semanas em cada nível.
    contagem_niveis = {}

    for nivel in NIVEIS:
        contagem_niveis[nivel] = int(
            (dados_validacao["nivel"] == nivel).sum()
        )

    # Confere se a soma dos níveis está correta.
    soma_niveis = sum(contagem_niveis.values())

    if soma_niveis != semanas_validas:
        raise ValueError(
            f"Soma dos níveis incorreta: {caminho.name}"
        )

    # Conta somente transições válidas.
    transicoes = contar_transicoes_validas(dados)

    # Identifica versões do modelo InfoDengue.
    versoes = (
        dados["versao_modelo"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    versoes = sorted(
        valor for valor in versoes.unique()
        if valor
    )

    return {
        "codigo_ibge": caminho.stem,
        "semanas_esperadas": semanas_previstas,
        "semanas_observadas": semanas_observadas,
        "semanas_ausentes": semanas_ausentes,
        "duplicatas": duplicatas,
        "semanas_nivel_valido": semanas_validas,
        "semanas_nivel_invalido_ou_ausente": semanas_invalidas,
        "nivel_1": contagem_niveis[1],
        "nivel_2": contagem_niveis[2],
        "nivel_3": contagem_niveis[3],
        "nivel_4": contagem_niveis[4],
        "transicoes_validas": transicoes,
        "versoes_modelo": "; ".join(versoes),
    }


def grafico_serra(semanas_esperadas):
    # Mostra a evolução dos níveis de alerta em Serra.
    arquivo = PASTA_DADOS / f"{SERRA}.csv"

    dados = ler_dados(arquivo)

    # Remove semanas duplicadas da visualização.
    duplicadas = dados["data_iniSE"].duplicated(
        keep=False
    )

    dados = dados.loc[~duplicadas].copy()

    # Níveis inválidos aparecem como lacunas no gráfico.
    dados.loc[
        ~dados["nivel"].isin(NIVEIS),
        "nivel"
    ] = np.nan

    serie = dados.set_index("data_iniSE")["nivel"]

    # Inclui todas as semanas esperadas no eixo temporal.
    datas = pd.DatetimeIndex(
        semanas_esperadas["data"]
    )

    serie = serie.reindex(datas)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        serie.index,
        serie.values,
        color="black",
        linewidth=1,
        marker=".",
        markersize=2
    )

    ax.set_title(
        "Evolução dos níveis de alerta de dengue - Serra (2015-2025)"
    )

    ax.set_xlabel("Ano")
    ax.set_ylabel("Nível de alerta")

    ax.set_yticks(NIVEIS)
    ax.set_ylim(0.5, 4.5)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_FIGURAS / "e1_serra_evolucao.png",
        dpi=300
    )

    plt.close(fig)


def mapa_calor_municipios(arquivos, semanas_esperadas):
    # Calcula a fração de semanas em níveis 3 ou 4.
    anos = list(range(ANO_INICIAL, ANO_FINAL + 1))

    matriz = []
    municipios = []

    calendario = semanas_esperadas.set_index("data")["ano"]

    for arquivo in arquivos:

        dados = ler_dados(arquivo)

        # Exclui semanas duplicadas da análise.
        duplicadas = dados["data_iniSE"].duplicated(
            keep=False
        )

        dados = dados.loc[~duplicadas].copy()

        dados["ano"] = dados["data_iniSE"].map(calendario)

        linha = []

        for ano in anos:

            dados_ano = dados[dados["ano"] == ano]

            # Considera somente semanas com níveis válidos.
            validos = dados_ano[
                dados_ano["nivel"].isin(NIVEIS)
            ]

            total = len(validos)

            if total == 0:
                fracao = np.nan
            else:
                alertas_altos = validos[
                    validos["nivel"].isin([3, 4])
                ]

                fracao = len(alertas_altos) / total

            linha.append(fracao)

        matriz.append(linha)
        municipios.append(arquivo.stem)

    matriz = np.array(matriz, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 18))

    # Células sem dados aparecem em branco.
    cores = plt.get_cmap("Greys").copy()
    cores.set_bad("white")

    imagem = ax.imshow(
        np.ma.masked_invalid(matriz),
        cmap=cores,
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(anos)))
    ax.set_xticklabels(anos)

    ax.set_yticks(range(len(municipios)))
    ax.set_yticklabels(municipios, fontsize=6)

    ax.set_xlabel("Ano")
    ax.set_ylabel("Município - código IBGE")

    ax.set_title(
        "Fração de semanas em alerta 3 ou 4 - Espírito Santo"
    )

    fig.colorbar(
        imagem,
        ax=ax,
        label="Fração de semanas"
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_FIGURAS / "e1_mapa_calor_municipios.png",
        dpi=300
    )

    plt.close(fig)


def executar():
    # Executa a auditoria completa da Etapa 1.
    arquivos = sorted(PASTA_DADOS.glob("*.csv"))

    if len(arquivos) != 78:
        raise ValueError(
            f"Esperados 78 arquivos CSV. "
            f"Encontrados: {len(arquivos)}."
        )

    # Confere se os nomes seguem o formato do código IBGE.
    codigos = [arquivo.stem for arquivo in arquivos]

    if any(
        len(codigo) != 7
        or not codigo.isdigit()
        or not codigo.startswith("32")
        for codigo in codigos
    ):
        raise ValueError(
            "Existem arquivos com códigos IBGE inválidos."
        )

    if SERRA not in codigos:
        raise ValueError(
            "Arquivo de Serra (3205002) não encontrado."
        )

    print("Verificando integridade dos arquivos...")

    # A auditoria só continua se os hashes estiverem corretos.
    conferir_integridade(arquivos)

    print("Gerando calendário epidemiológico...")

    semanas_esperadas = gerar_semanas_esperadas()

    resultados = []

    print("Analisando os 78 municípios...")

    for arquivo in arquivos:

        resultado = analisar_municipio(
            arquivo,
            semanas_esperadas
        )

        resultados.append(resultado)

        print(
            f"{arquivo.stem}: "
            f"{resultado['transicoes_validas']} "
            "transições válidas"
        )

    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_FIGURAS.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salva a tabela de auditoria.
    auditoria = pd.DataFrame(resultados)

    auditoria.to_csv(
        PASTA_RESULTADOS / "e1_auditoria.csv",
        index=False
    )

    print("Gerando gráfico de Serra...")

    grafico_serra(semanas_esperadas)

    print("Gerando mapa de calor...")

    mapa_calor_municipios(
        arquivos,
        semanas_esperadas
    )

    print("Etapa 1 concluída.")
    print("Resultados salvos na pasta results/.")


if __name__ == "__main__":
    executar()


# [/IA]