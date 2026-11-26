# [IA] ChatGPT – 26/11/2026 – testes automatizados da Etapa 5

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import linalg

import src.etapa5 as etapa5

from src.etapa1 import gerar_semanas_esperadas

from src.etapa3 import (
    carregar_matriz,
    estacionaria_sistema,
)

from src.etapa5 import (
    permanencia_teorica,
    preparar_segmentos,
    obter_arquivos,
    carregar_segmentos,
    duracoes_observadas,
    calcular_permanencias,
    primeira_passagem_sistemas,
    verificar_kac,
    primeira_passagem_fundamental,
    comparar_matrizes_passagem,
    tempos_empiricos_passagem,
    comparar_primeira_passagem_empirica,
    grafico_nivel4,
    salvar_matriz,
    conferir_referencia,
    analisar_localidade,
    executar,
)


# --------------------------------------------------
# CONFIGURAÇÕES DOS TESTES
# --------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data" / "raw"
PASTA_TABELAS = RAIZ / "results" / "tabelas"

TOLERANCIA = 1e-10
TOLERANCIA_REFERENCIA = 1e-6

LOCALIDADES = [
    "serra",
    "vila_velha",
    "es_agregado",
]

CODIGOS = {
    "serra": ["3205002"],
    "vila_velha": ["3205200"],
    "es_agregado": None,
}

# Matriz fictícia fornecida na Tabela 1 do professor.
P_REFERENCIA = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])

M14_REFERENCIA = 46.873065
M41_REFERENCIA = 17.035647

# Matriz artificial simples, com probabilidades iguais.
P_UNIFORME = np.full((4, 4), 0.25)

# Resultados reais informados após a execução da Etapa 5.
RESULTADOS_REAIS = {
    "serra": {
        "m14": 60.828326,
        "m41": 14.178112,
    },
    "vila_velha": {
        "m14": 45.332096,
        "m41": 15.263451,
    },
    "es_agregado": {
        "m14": 73.938840,
        "m41": 11.395159,
    },
}


# --------------------------------------------------
# FUNÇÃO AUXILIAR PARA OS TESTES
# --------------------------------------------------

def criar_csv_temporario(tmp_path, indices, niveis, nome="teste.csv"):
    # Cria dados artificiais com datas epidemiológicas válidas.

    calendario = gerar_semanas_esperadas()

    linhas = calendario.iloc[indices]

    dados = pd.DataFrame({
        "data_iniSE": linhas["data"].dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
        "SE": linhas["SE"].tolist(),
        "nivel": niveis,
        "versao_modelo": ["teste"] * len(indices),
    })

    caminho = tmp_path / nome

    dados.to_csv(
        caminho,
        index=False
    )

    return caminho


# --------------------------------------------------
# TESTES DA PERMANÊNCIA TEÓRICA
# --------------------------------------------------

def test_permanencia_teorica():
    # Verifica a fórmula 1 / (1 - pii).

    resultado = permanencia_teorica(P_REFERENCIA)

    esperado = np.array([
        1 / (1 - 0.90),
        1 / (1 - 0.65),
        1 / (1 - 0.65),
        1 / (1 - 0.75),
    ])

    np.testing.assert_allclose(
        resultado,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_permanencia_teorica_uniforme():
    # Verifica a permanência em uma matriz artificial.

    resultado = permanencia_teorica(P_UNIFORME)

    esperado = np.full(
        4,
        1 / 0.75
    )

    np.testing.assert_allclose(
        resultado,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_permanencia_estado_absorvente():
    # Um estado com pii igual a 1 possui permanência infinita.

    P = np.eye(4)

    resultado = permanencia_teorica(P)

    assert np.all(
        np.isinf(resultado)
    )


# --------------------------------------------------
# TESTES DAS SEQUÊNCIAS TEMPORAIS
# --------------------------------------------------

def test_preparar_segmentos_consecutivos(tmp_path):
    # Semanas consecutivas permanecem no mesmo segmento.

    arquivo = criar_csv_temporario(
        tmp_path,
        [0, 1, 2, 3, 4],
        [1, 2, 2, 3, 4]
    )

    calendario = gerar_semanas_esperadas()

    segmentos = preparar_segmentos(
        arquivo,
        calendario
    )

    assert segmentos == [
        [1, 2, 2, 3, 4]
    ]


def test_lacuna_interrompe_segmentos(tmp_path):
    # Uma semana ausente divide a série em dois segmentos.

    arquivo = criar_csv_temporario(
        tmp_path,
        [0, 1, 3, 4],
        [1, 2, 3, 4]
    )

    calendario = gerar_semanas_esperadas()

    segmentos = preparar_segmentos(
        arquivo,
        calendario
    )

    assert segmentos == [
        [1, 2],
        [3, 4],
    ]


@pytest.mark.parametrize(
    "nivel_invalido",
    [None, 0, 5, -1, np.nan]
)
def test_nivel_invalido_interrompe_segmentos(
    tmp_path,
    nivel_invalido
):
    # Níveis ausentes ou inválidos interrompem a sequência.

    arquivo = criar_csv_temporario(
        tmp_path,
        [0, 1, 2, 3, 4],
        [1, 2, nivel_invalido, 3, 4]
    )

    calendario = gerar_semanas_esperadas()

    segmentos = preparar_segmentos(
        arquivo,
        calendario
    )

    assert segmentos == [
        [1, 2],
        [3, 4],
    ]


def test_semanas_duplicadas_interrompem_segmentos(tmp_path):
    # Remove semanas duplicadas sem unir os registros vizinhos.

    arquivo = criar_csv_temporario(
        tmp_path,
        [0, 1, 1, 2, 3],
        [1, 2, 2, 3, 4]
    )

    calendario = gerar_semanas_esperadas()

    segmentos = preparar_segmentos(
        arquivo,
        calendario
    )

    assert segmentos == [
        [1],
        [3, 4],
    ]


def test_municipios_nao_sao_conectados(
    tmp_path,
    monkeypatch
):
    # Garante que as séries municipais permaneçam separadas.

    arquivo_serra = criar_csv_temporario(
        tmp_path,
        [0, 1],
        [1, 2],
        nome="3205002.csv"
    )

    arquivo_vila_velha = criar_csv_temporario(
        tmp_path,
        [2, 3],
        [2, 4],
        nome="3205200.csv"
    )

    monkeypatch.setattr(
        etapa5,
        "PASTA_DADOS",
        tmp_path
    )

    segmentos = carregar_segmentos([
        "3205002",
        "3205200",
    ])

    assert segmentos == [
        [1, 2],
        [2, 4],
    ]

    # Não pode existir uma passagem artificial de 1 para 4.
    tempos, censurados = tempos_empiricos_passagem(
        segmentos,
        1,
        4
    )

    assert tempos == []
    assert censurados == 1

    assert arquivo_serra.exists()
    assert arquivo_vila_velha.exists()


# --------------------------------------------------
# TESTES DAS DURAÇÕES OBSERVADAS
# --------------------------------------------------

def test_duracoes_observadas():
    # Conta somente permanências com entrada e saída observadas.

    segmentos = [
        [1, 2, 2, 3, 3, 4],
        [4, 4, 2, 1, 1, 3],
        [2, 2, 2],
    ]

    duracoes, censuradas = duracoes_observadas(
        segmentos
    )

    assert duracoes[1] == [2]
    assert duracoes[2] == [2, 1]
    assert duracoes[3] == [2]
    assert duracoes[4] == []

    # Os períodos nas extremidades são censurados.
    assert censuradas[1] == 1
    assert censuradas[2] == 1
    assert censuradas[3] == 1
    assert censuradas[4] == 2


def test_periodos_censurados_inicio_fim():
    # Uma permanência completa precisa ter as duas fronteiras.

    segmentos = [
        [1, 1, 2, 2, 3, 3],
    ]

    duracoes, censuradas = duracoes_observadas(
        segmentos
    )

    assert duracoes[1] == []
    assert duracoes[2] == [2]
    assert duracoes[3] == []

    assert censuradas[1] == 1
    assert censuradas[2] == 0
    assert censuradas[3] == 1


def test_segmento_unico_e_censurado():
    # Uma sequência inteira no mesmo nível é censurada.

    segmentos = [
        [4, 4, 4, 4],
    ]

    duracoes, censuradas = duracoes_observadas(
        segmentos
    )

    assert duracoes[4] == []
    assert censuradas[4] == 1


def test_calcular_permanencias():
    # Confere as médias observadas e teóricas.

    segmentos = [
        [1, 2, 2, 3, 4],
        [4, 2, 1, 1, 3],
    ]

    tabela, duracoes = calcular_permanencias(
        segmentos,
        P_REFERENCIA
    )

    assert len(tabela) == 4

    assert set(tabela["nivel"]) == {
        1, 2, 3, 4
    }

    # O nível 2 possui durações completas de 2 e 1.
    assert duracoes[2] == [2, 1]

    linha = tabela[
        tabela["nivel"] == 2
    ].iloc[0]

    assert linha["quantidade_periodos_completos"] == 2

    assert np.isclose(
        linha["permanencia_empirica"],
        1.5,
        atol=TOLERANCIA,
        rtol=0
    )

    assert np.isclose(
        linha["permanencia_teorica"],
        1 / (1 - P_REFERENCIA[1, 1]),
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DA DISTRIBUIÇÃO GEOMÉTRICA
# --------------------------------------------------

def test_distribuicao_geometrica_nivel4():
    # Confere a distribuição geométrica teórica.

    p44 = P_REFERENCIA[3, 3]

    prob_saida = 1 - p44

    semanas = np.arange(1, 101)

    probabilidades = (
        prob_saida
        * p44 ** (semanas - 1)
    )

    # A soma finita corresponde à probabilidade acumulada.
    esperado = 1 - p44 ** 100

    assert np.isclose(
        probabilidades.sum(),
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )

    assert np.isclose(
        probabilidades[0],
        prob_saida,
        atol=TOLERANCIA,
        rtol=0
    )

    # A média teórica da geométrica é 1 / (1 - p44).
    media = permanencia_teorica(P_REFERENCIA)[3]

    assert np.isclose(
        media,
        1 / prob_saida,
        atol=TOLERANCIA,
        rtol=0
    )


def test_histograma_nivel4(
    tmp_path,
    monkeypatch
):
    # Gera o histograma somente na pasta temporária.

    monkeypatch.setattr(
        etapa5,
        "PASTA_FIGURAS",
        tmp_path
    )

    duracoes = {
        1: [],
        2: [],
        3: [],
        4: [1, 2, 2, 3, 4, 5],
    }

    resultado = grafico_nivel4(
        "serra",
        duracoes,
        P_REFERENCIA
    )

    arquivo = (
        tmp_path / "e5_serra_nivel4_geometrica.png"
    )

    assert resultado is True
    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

    imagem = plt.imread(arquivo)

    assert imagem.ndim == 3


def test_histograma_sem_periodos_completos(
    tmp_path,
    monkeypatch
):
    # Não gera histograma quando não existem durações completas.

    monkeypatch.setattr(
        etapa5,
        "PASTA_FIGURAS",
        tmp_path
    )

    duracoes = {
        1: [],
        2: [],
        3: [],
        4: [],
    }

    resultado = grafico_nivel4(
        "serra",
        duracoes,
        P_REFERENCIA
    )

    assert resultado is False

    assert not (
        tmp_path / "e5_serra_nivel4_geometrica.png"
    ).exists()


# --------------------------------------------------
# TESTES DOS QUATRO SISTEMAS LINEARES 3x3
# --------------------------------------------------

def test_primeira_passagem_sistemas():
    # Confere os tempos de primeira passagem na matriz uniforme.

    pi = estacionaria_sistema(P_UNIFORME)

    M = primeira_passagem_sistemas(
        P_UNIFORME,
        pi
    )

    # Todos os tempos são iguais a 4 nesta matriz.
    esperado = np.full((4, 4), 4.0)

    np.testing.assert_allclose(
        M,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_sistemas_lineares_primeira_passagem():
    # Verifica independentemente as quatro equações 3x3.

    pi = estacionaria_sistema(P_REFERENCIA)

    M = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    for destino in range(4):

        transientes = [
            i for i in range(4)
            if i != destino
        ]

        Q = P_REFERENCIA[
            np.ix_(transientes, transientes)
        ]

        A = np.eye(3) - Q

        b = np.ones(3)

        tempos = M[transientes, destino]

        # Verifica (I - Q) * tempos = 1.
        np.testing.assert_allclose(
            A @ tempos,
            b,
            atol=TOLERANCIA,
            rtol=0
        )

        # Confere também com um sistema independente.
        esperado = linalg.solve(A, b)

        np.testing.assert_allclose(
            tempos,
            esperado,
            atol=TOLERANCIA,
            rtol=0
        )


def test_matriz_primeira_passagem_dimensoes():
    # A matriz M deve possuir quatro linhas e colunas.

    pi = estacionaria_sistema(P_REFERENCIA)

    M = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    assert M.shape == (4, 4)

    assert np.all(np.isfinite(M))

    assert np.all(M > 0)


# --------------------------------------------------
# TESTES DA MATRIZ FUNDAMENTAL Z
# --------------------------------------------------

def test_matriz_fundamental():
    # Confere Z = (I - P + 1*pi)^-1.

    pi = estacionaria_sistema(P_REFERENCIA)

    M, Z = primeira_passagem_fundamental(
        P_REFERENCIA,
        pi
    )

    matriz = (
        np.eye(4)
        - P_REFERENCIA
        + np.outer(np.ones(4), pi)
    )

    np.testing.assert_allclose(
        matriz @ Z,
        np.eye(4),
        atol=TOLERANCIA,
        rtol=0
    )

    assert M.shape == (4, 4)
    assert Z.shape == (4, 4)


def test_primeira_passagem_fundamental():
    # Compara os tempos com a fórmula usando Z.

    pi = estacionaria_sistema(P_REFERENCIA)

    M, Z = primeira_passagem_fundamental(
        P_REFERENCIA,
        pi
    )

    for i in range(4):

        for j in range(4):

            if i == j:
                esperado = 1 / pi[j]
            else:
                esperado = (
                    Z[j, j] - Z[i, j]
                ) / pi[j]

            assert np.isclose(
                M[i, j],
                esperado,
                atol=TOLERANCIA,
                rtol=0
            )


def test_concordancia_metodos_primeira_passagem():
    # Compara os dois métodos independentes.

    pi = estacionaria_sistema(P_REFERENCIA)

    M_sistemas = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    M_fundamental, Z = primeira_passagem_fundamental(
        P_REFERENCIA,
        pi
    )

    resultado = comparar_matrizes_passagem(
        M_sistemas,
        M_fundamental
    )

    # Usa comparação relativa para evitar testes frágeis.
    np.testing.assert_allclose(
        M_sistemas,
        M_fundamental,
        atol=TOLERANCIA,
        rtol=TOLERANCIA
    )

    diferencas = np.abs(
        M_sistemas - M_fundamental
    )

    assert np.isclose(
        resultado["maior_diferenca"],
        np.max(diferencas),
        atol=TOLERANCIA,
        rtol=0
    )

    assert np.isclose(
        resultado["norma_matricial_l1"],
        np.linalg.norm(
            M_sistemas - M_fundamental,
            ord=1
        ),
        atol=TOLERANCIA,
        rtol=0
    )


def test_comparacao_relativa_matrizes():
    # Diferenças pequenas em valores grandes são aceitas.

    M1 = np.full((4, 4), 1000.0)

    M2 = M1.copy()

    M2[0, 0] += 1e-8

    resultado = comparar_matrizes_passagem(
        M1,
        M2
    )

    assert resultado["maior_diferenca"] > 0

    np.testing.assert_allclose(
        M1,
        M2,
        atol=TOLERANCIA,
        rtol=TOLERANCIA
    )


def test_rejeitar_matrizes_divergentes():
    # Diferenças relevantes devem gerar erro.

    M1 = np.full((4, 4), 10.0)

    M2 = M1.copy()

    M2[0, 0] = 20.0

    with pytest.raises(ValueError):
        comparar_matrizes_passagem(
            M1,
            M2
        )


# --------------------------------------------------
# TESTES DA FÓRMULA DE KAC
# --------------------------------------------------

def test_formula_kac_referencia():
    # Verifica Kac com tolerância obrigatória de 1e-10.

    pi = estacionaria_sistema(P_REFERENCIA)

    M = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    resultado = verificar_kac(
        P_REFERENCIA,
        pi,
        M
    )

    assert len(resultado) == 4

    assert (
        resultado["diferenca_absoluta"] <= TOLERANCIA
    ).all()

    assert resultado["verificacao"].all()


def test_formula_kac_todos_niveis():
    # Verifica diretamente mjj = 1 / pi_j.

    pi = estacionaria_sistema(P_REFERENCIA)

    M = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    np.testing.assert_allclose(
        np.diag(M),
        1 / pi,
        atol=TOLERANCIA,
        rtol=0
    )

    # Confere também a equação de retorno.
    for j in range(4):

        retorno = 1.0

        for k in range(4):

            if k != j:
                retorno += P_REFERENCIA[j, k] * M[k, j]

        assert abs(
            retorno - 1 / pi[j]
        ) <= TOLERANCIA


# --------------------------------------------------
# TESTES DOS TEMPOS EMPÍRICOS
# --------------------------------------------------

def test_primeira_passagem_empirica_1_para_4():
    # Verifica as primeiras chegadas ao nível 4.

    segmentos = [
        [1, 2, 4, 1],
        [4, 3, 1, 4],
        [1, 1, 4],
    ]

    tempos, censurados = tempos_empiricos_passagem(
        segmentos,
        1,
        4
    )

    assert tempos == [2, 1, 2, 1]

    assert censurados == 1

    assert np.mean(tempos) == 1.5


def test_primeira_passagem_empirica_4_para_1():
    # Verifica as primeiras chegadas ao nível 1.

    segmentos = [
        [1, 2, 4, 1],
        [4, 3, 1, 4],
        [1, 1, 4],
    ]

    tempos, censurados = tempos_empiricos_passagem(
        segmentos,
        4,
        1
    )

    assert tempos == [1, 2]

    assert censurados == 2

    assert np.mean(tempos) == 1.5


def test_primeira_passagem_nao_atravessa_lacuna():
    # Segmentos diferentes não podem gerar uma chegada artificial.

    segmentos = [
        [1, 2, 3],
        [4, 4, 4],
    ]

    tempos, censurados = tempos_empiricos_passagem(
        segmentos,
        1,
        4
    )

    assert tempos == []

    assert censurados == 1


def test_comparacao_primeira_passagem_empirica():
    # Confere a comparação dos tempos teóricos e observados.

    segmentos = [
        [1, 2, 4, 1],
        [4, 3, 1, 4],
        [1, 1, 4],
    ]

    pi = estacionaria_sistema(P_REFERENCIA)

    M = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    tabela = comparar_primeira_passagem_empirica(
        segmentos,
        M
    )

    assert len(tabela) == 2

    linha_14 = tabela[
        (tabela["origem"] == 1)
        & (tabela["destino"] == 4)
    ].iloc[0]

    linha_41 = tabela[
        (tabela["origem"] == 4)
        & (tabela["destino"] == 1)
    ].iloc[0]

    assert linha_14["observacoes_completas"] == 4

    assert linha_14["observacoes_censuradas"] == 1

    assert linha_41["observacoes_completas"] == 2

    assert linha_41["observacoes_censuradas"] == 2

    assert np.isclose(
        linha_14["tempo_empirico"],
        1.5,
        atol=TOLERANCIA,
        rtol=0
    )

    assert np.isclose(
        linha_41["tempo_empirico"],
        1.5,
        atol=TOLERANCIA,
        rtol=0
    )

    assert np.isclose(
        linha_14["tempo_teorico"],
        M[0, 3],
        atol=TOLERANCIA,
        rtol=0
    )

    assert np.isclose(
        linha_41["tempo_teorico"],
        M[3, 0],
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DA TABELA 1 DO PROFESSOR
# --------------------------------------------------

def test_m14_tabela1():
    # Confere m14 da matriz de referência.

    pi = estacionaria_sistema(P_REFERENCIA)

    M = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    assert np.isclose(
        M[0, 3],
        M14_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


def test_m41_tabela1():
    # Confere m41 da matriz de referência.

    pi = estacionaria_sistema(P_REFERENCIA)

    M = primeira_passagem_sistemas(
        P_REFERENCIA,
        pi
    )

    assert np.isclose(
        M[3, 0],
        M41_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


def test_tabela1_metodo_fundamental():
    # Confere os valores também pelo método independente.

    pi = estacionaria_sistema(P_REFERENCIA)

    M, Z = primeira_passagem_fundamental(
        P_REFERENCIA,
        pi
    )

    assert np.isclose(
        M[0, 3],
        M14_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )

    assert np.isclose(
        M[3, 0],
        M41_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


def test_conferencia_tabela1_gera_csv(
    tmp_path,
    monkeypatch
):
    # Executa a conferência sem alterar os resultados existentes.

    monkeypatch.setattr(
        etapa5,
        "PASTA_TABELAS",
        tmp_path
    )

    conferir_referencia()

    arquivo = (
        tmp_path / "e5_conferencia_tabela1.csv"
    )

    assert arquivo.exists()

    tabela = pd.read_csv(arquivo)

    assert set(tabela["grandeza"]) == {
        "m14",
        "m41",
    }

    assert (
        tabela["erro_absoluto"] <= TOLERANCIA_REFERENCIA
    ).all()

    assert (
        tmp_path / "e5_referencia_M_sistemas.csv"
    ).exists()

    assert (
        tmp_path / "e5_referencia_M_fundamental.csv"
    ).exists()

    assert (
        tmp_path / "e5_referencia_Z.csv"
    ).exists()

    assert (
        tmp_path / "e5_referencia_kac.csv"
    ).exists()


# --------------------------------------------------
# TESTES DOS ARQUIVOS CSV
# --------------------------------------------------

def test_salvar_matriz_csv(
    tmp_path,
    monkeypatch
):
    # Salva uma matriz artificial em uma pasta temporária.

    monkeypatch.setattr(
        etapa5,
        "PASTA_TABELAS",
        tmp_path
    )

    salvar_matriz(
        "teste",
        "M_sistemas",
        P_REFERENCIA
    )

    arquivo = (
        tmp_path / "e5_teste_M_sistemas.csv"
    )

    assert arquivo.exists()

    tabela = pd.read_csv(
        arquivo,
        index_col=0
    )

    assert tabela.shape == (4, 4)

    assert list(tabela.index) == [
        "origem_1",
        "origem_2",
        "origem_3",
        "origem_4",
    ]

    assert list(tabela.columns) == [
        "destino_1",
        "destino_2",
        "destino_3",
        "destino_4",
    ]

    np.testing.assert_allclose(
        tabela.to_numpy(dtype=float),
        P_REFERENCIA,
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DE INTEGRAÇÃO COM OS DADOS REAIS
# --------------------------------------------------

@pytest.mark.parametrize("nome", LOCALIDADES)
def test_permanencias_reais(nome):
    # Calcula permanências utilizando os dados reais.

    P = carregar_matriz(nome)

    segmentos = carregar_segmentos(
        CODIGOS[nome]
    )

    tabela, duracoes = calcular_permanencias(
        segmentos,
        P
    )

    assert len(tabela) == 4

    teoricas = permanencia_teorica(P)

    np.testing.assert_allclose(
        tabela["permanencia_teorica"].to_numpy(),
        teoricas,
        atol=TOLERANCIA,
        rtol=0
    )

    for nivel in range(1, 5):

        linha = tabela[
            tabela["nivel"] == nivel
        ].iloc[0]

        assert (
            linha["quantidade_periodos_completos"]
            == len(duracoes[nivel])
        )

        assert linha["periodos_censurados"] >= 0


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_matriz_passagem_real(nome):
    # Confere os tempos reais pelos quatro sistemas.

    P = carregar_matriz(nome)

    pi = estacionaria_sistema(P)

    M = primeira_passagem_sistemas(
        P,
        pi
    )

    esperado = RESULTADOS_REAIS[nome]

    assert M.shape == (4, 4)

    assert np.all(np.isfinite(M))

    assert np.all(M > 0)

    assert np.isclose(
        M[0, 3],
        esperado["m14"],
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )

    assert np.isclose(
        M[3, 0],
        esperado["m41"],
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_matriz_fundamental_real(nome):
    # Confere os tempos pelo método independente Z.

    P = carregar_matriz(nome)

    pi = estacionaria_sistema(P)

    M_sistemas = primeira_passagem_sistemas(
        P,
        pi
    )

    M_fundamental, Z = primeira_passagem_fundamental(
        P,
        pi
    )

    np.testing.assert_allclose(
        M_sistemas,
        M_fundamental,
        atol=TOLERANCIA,
        rtol=TOLERANCIA
    )

    # Confere a equação da matriz fundamental.
    A = (
        np.eye(4)
        - P
        + np.outer(np.ones(4), pi)
    )

    np.testing.assert_allclose(
        A @ Z,
        np.eye(4),
        atol=TOLERANCIA,
        rtol=0
    )


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_kac_matrizes_reais(nome):
    # Verifica Kac com tolerância absoluta de 1e-10.

    P = carregar_matriz(nome)

    P = P / P.sum(axis=1, keepdims=True)

    pi = estacionaria_sistema(P)

    M = primeira_passagem_sistemas(
        P,
        pi
    )

    resultado = verificar_kac(
        P,
        pi,
        M
    )

    assert len(resultado) == 4

    assert resultado["verificacao"].all()

    assert (
        resultado["diferenca_absoluta"].max()
        <= TOLERANCIA
    )

    np.testing.assert_allclose(
        np.diag(M),
        1 / pi,
        atol=TOLERANCIA,
        rtol=0
    )


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_primeira_passagem_empirica_real(nome):
    # Confere os dados empíricos sem exigir igualdade com a teoria.

    P = carregar_matriz(nome)

    pi = estacionaria_sistema(P)

    M = primeira_passagem_sistemas(
        P,
        pi
    )

    segmentos = carregar_segmentos(
        CODIGOS[nome]
    )

    tabela = comparar_primeira_passagem_empirica(
        segmentos,
        M
    )

    assert len(tabela) == 2

    assert set(
        zip(tabela["origem"], tabela["destino"])
    ) == {
        (1, 4),
        (4, 1),
    }

    assert (
        tabela["observacoes_completas"] >= 0
    ).all()

    assert (
        tabela["observacoes_censuradas"] >= 0
    ).all()

    # As médias empíricas não precisam coincidir com a teoria.
    for _, linha in tabela.iterrows():

        assert np.isclose(
            linha["tempo_teorico"],
            M[
                int(linha["origem"]) - 1,
                int(linha["destino"]) - 1
            ],
            atol=TOLERANCIA,
            rtol=0
        )

        if linha["observacoes_completas"] > 0:
            assert np.isfinite(
                linha["tempo_empirico"]
            )
            assert linha["tempo_empirico"] > 0


# --------------------------------------------------
# TESTES DA ANÁLISE COMPLETA
# --------------------------------------------------

@pytest.mark.parametrize("nome", LOCALIDADES)
def test_analisar_localidade(
    nome,
    tmp_path,
    monkeypatch
):
    # Utiliza dados reais apenas para leitura.
    # Todos os resultados são salvos em pastas temporárias.

    pasta_tabelas = tmp_path / "tabelas"
    pasta_figuras = tmp_path / "figuras"

    pasta_tabelas.mkdir()
    pasta_figuras.mkdir()

    monkeypatch.setattr(
        etapa5,
        "PASTA_TABELAS",
        pasta_tabelas
    )

    monkeypatch.setattr(
        etapa5,
        "PASTA_FIGURAS",
        pasta_figuras
    )

    resumo = analisar_localidade(
        nome,
        CODIGOS[nome]
    )

    esperado = RESULTADOS_REAIS[nome]

    assert resumo["localidade"] == nome

    assert np.isclose(
        resumo["m14"],
        esperado["m14"],
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )

    assert np.isclose(
        resumo["m41"],
        esperado["m41"],
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )

    assert resumo["maior_erro_kac"] <= TOLERANCIA

    # Confere os CSVs da localidade.
    arquivos_csv = [
        f"e5_{nome}_permanencias.csv",
        f"e5_{nome}_duracoes.csv",
        f"e5_{nome}_M_sistemas.csv",
        f"e5_{nome}_M_fundamental.csv",
        f"e5_{nome}_Z.csv",
        f"e5_{nome}_diferencas_M.csv",
        f"e5_{nome}_kac.csv",
        f"e5_{nome}_primeira_passagem_empirica.csv",
    ]

    for arquivo in arquivos_csv:

        caminho = pasta_tabelas / arquivo

        assert caminho.exists(), (
            f"Arquivo não gerado: {arquivo}"
        )

        assert caminho.stat().st_size > 0

    # Confere a concordância dos CSVs produzidos.
    M_sistemas = pd.read_csv(
        pasta_tabelas / f"e5_{nome}_M_sistemas.csv",
        index_col=0
    ).to_numpy(dtype=float)

    M_fundamental = pd.read_csv(
        pasta_tabelas / f"e5_{nome}_M_fundamental.csv",
        index_col=0
    ).to_numpy(dtype=float)

    np.testing.assert_allclose(
        M_sistemas,
        M_fundamental,
        atol=TOLERANCIA,
        rtol=TOLERANCIA
    )

    # Confere o histograma quando houver dados completos.
    if resumo["grafico_nivel4_gerado"]:

        arquivo_png = (
            pasta_figuras
            / f"e5_{nome}_nivel4_geometrica.png"
        )

        assert arquivo_png.exists()
        assert arquivo_png.stat().st_size > 0

        imagem = plt.imread(arquivo_png)

        assert imagem.ndim == 3


def test_executar_etapa5(
    tmp_path,
    monkeypatch
):
    # Executa a Etapa 5 completa.
    # Não sobrescreve os resultados reais do projeto.

    pasta_tabelas = tmp_path / "tabelas"
    pasta_figuras = tmp_path / "figuras"

    monkeypatch.setattr(
        etapa5,
        "PASTA_TABELAS",
        pasta_tabelas
    )

    monkeypatch.setattr(
        etapa5,
        "PASTA_FIGURAS",
        pasta_figuras
    )

    executar()

    arquivo_resumo = (
        pasta_tabelas / "e5_resumo.csv"
    )

    assert arquivo_resumo.exists()

    resumo = pd.read_csv(
        arquivo_resumo
    )

    assert len(resumo) == 3

    assert set(resumo["localidade"]) == set(
        LOCALIDADES
    )

    # Verifica os resultados reais informados.
    for nome in LOCALIDADES:

        linha = resumo[
            resumo["localidade"] == nome
        ].iloc[0]

        esperado = RESULTADOS_REAIS[nome]

        assert np.isclose(
            linha["m14"],
            esperado["m14"],
            atol=TOLERANCIA_REFERENCIA,
            rtol=0
        )

        assert np.isclose(
            linha["m41"],
            esperado["m41"],
            atol=TOLERANCIA_REFERENCIA,
            rtol=0
        )

        assert linha["maior_erro_kac"] <= TOLERANCIA

    # Confere a tabela de referência do professor.
    assert (
        pasta_tabelas / "e5_conferencia_tabela1.csv"
    ).exists()

    # Confere os CSVs das três localidades.
    for nome in LOCALIDADES:

        assert (
            pasta_tabelas / f"e5_{nome}_M_sistemas.csv"
        ).exists()

        assert (
            pasta_tabelas / f"e5_{nome}_M_fundamental.csv"
        ).exists()

        assert (
            pasta_tabelas / f"e5_{nome}_kac.csv"
        ).exists()


# [/IA]