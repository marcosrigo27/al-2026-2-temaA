# [IA] ChatGPT – 24/11/2026 – testes automatizados da Etapa 4

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import linalg

import src.etapa4 as etapa4

from src.etapa4 import (
    calcular_espectro,
    calcular_meia_vida,
    diagonalizar_matriz,
    converter_real,
    potencia_quadrados,
    potencia_diagonalizacao,
    comparar_potencias,
    calcular_convergencia,
    gerar_grafico_convergencia,
    salvar_matriz,
    salvar_autovalores,
    analisar_localidade,
    executar,
)

from src.etapa3 import (
    carregar_matriz,
    estacionaria_sistema,
)


# --------------------------------------------------
# CONFIGURAÇÕES
# --------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent

PASTA_TABELAS = RAIZ / "results" / "tabelas"

TOLERANCIA = 1e-10
TOLERANCIA_REFERENCIA = 1e-6
TOLERANCIA_APROXIMADA = 1e-3

HORIZONTE = 52


# Matriz artificial simétrica e estocástica.
P_ARTIFICIAL = np.array([
    [0.7, 0.1, 0.1, 0.1],
    [0.1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
    [0.1, 0.1, 0.1, 0.7],
])


# Matriz identidade.
P_IDENTIDADE = np.eye(4)


# Matriz de permutação cíclica.
# Possui autovalores complexos.
P_COMPLEXA = np.array([
    [0.6, 0.4, 0.0, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.0, 0.6, 0.4],
    [0.4, 0.0, 0.0, 0.6],
])


# Matriz estocástica não diagonalizável.
P_NAO_DIAGONALIZAVEL = np.array([
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0],
])


# Matriz de referência da Tabela 1 do professor.
P_REFERENCIA = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])


AUTOVALORES_REFERENCIA = np.array([
    1.0,
    0.869971,
    0.618816,
    0.461213,
])


# Resultados reais informados após a execução da Etapa 4.
RESULTADOS_REAIS = {
    "serra": {
        "lambda2": 0.879408,
        "meia_vida": 5.393856,
        "condicao": 6.076859,
        "diferenca_p52": 4.6407e-14,
        "complexos": True,
    },
    "vila_velha": {
        "lambda2": 0.881074,
        "meia_vida": 5.474521,
        "condicao": 4.495626,
        "diferenca_p52": 9.4480e-14,
    },
    "es_agregado": {
        "lambda2": 0.885841,
        "meia_vida": 5.718216,
        "condicao": 6.395965,
        "diferenca_p52": 1.3067e-13,
    },
}


LOCALIDADES = [
    "serra",
    "vila_velha",
    "es_agregado",
]


# --------------------------------------------------
# TESTES DOS AUTOVALORES
# --------------------------------------------------

def test_calcular_quatro_autovalores():
    # Verifica se os quatro autovalores são calculados.

    resultado = calcular_espectro(P_ARTIFICIAL)

    autovalores = resultado["autovalores"]

    assert len(autovalores) == 4

    assert np.all(np.isfinite(autovalores))


def test_autovalores_matriz_artificial():
    # A matriz artificial possui autovalores 1 e 0,6.

    resultado = calcular_espectro(P_ARTIFICIAL)

    obtidos = np.sort(
        np.real(resultado["autovalores"])
    )

    esperados = np.array([
        0.6,
        0.6,
        0.6,
        1.0,
    ])

    np.testing.assert_allclose(
        obtidos,
        esperados,
        atol=TOLERANCIA,
        rtol=0
    )


def test_soma_autovalores_igual_traco():
    # A soma dos autovalores deve ser igual ao traço.

    resultado = calcular_espectro(P_REFERENCIA)

    soma = np.sum(resultado["autovalores"])

    traco = np.trace(P_REFERENCIA)

    assert abs(soma - traco) <= TOLERANCIA

    assert resultado["erro_traco"] <= TOLERANCIA


def test_produto_autovalores_igual_determinante():
    # O produto dos autovalores deve ser igual ao determinante.

    resultado = calcular_espectro(P_REFERENCIA)

    produto = np.prod(resultado["autovalores"])

    determinante = linalg.det(P_REFERENCIA)

    assert abs(produto - determinante) <= TOLERANCIA

    assert resultado["erro_determinante"] <= TOLERANCIA


def test_autovalores_tabela_referencia():
    # Verifica o espectro informado na Tabela 1.

    resultado = calcular_espectro(P_REFERENCIA)

    obtidos = np.sort(
        np.abs(resultado["autovalores"])
    )[::-1]

    np.testing.assert_allclose(
        obtidos,
        AUTOVALORES_REFERENCIA,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DE LAMBDA2
# --------------------------------------------------

def test_lambda2_artificial():
    # O segundo maior módulo da matriz artificial é 0,6.

    resultado = calcular_espectro(P_ARTIFICIAL)

    assert np.isclose(
        resultado["lambda2"],
        0.6,
        atol=TOLERANCIA,
        rtol=0
    )


def test_lambda2_referencia():
    # Verifica lambda2 da matriz de referência.

    resultado = calcular_espectro(P_REFERENCIA)

    assert np.isclose(
        resultado["lambda2"],
        0.869971,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


def test_lambda2_calculo_independente():
    # Confere lambda2 diretamente com SciPy.

    resultado = calcular_espectro(P_REFERENCIA)

    autovalores = linalg.eigvals(P_REFERENCIA)

    modulos = np.sort(
        np.abs(autovalores)
    )[::-1]

    esperado = modulos[1]

    assert np.isclose(
        resultado["lambda2"],
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DA MEIA-VIDA
# --------------------------------------------------

def test_formula_meia_vida():
    # Verifica a fórmula ln(0,5) / ln(lambda2).

    lambda2 = 0.6

    esperado = np.log(0.5) / np.log(0.6)

    resultado = calcular_meia_vida(lambda2)

    assert np.isclose(
        resultado,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_meia_vida_referencia():
    # Verifica a meia-vida da Tabela 1.

    resultado = calcular_meia_vida(0.869971)

    assert np.isclose(
        resultado,
        4.976086,
        atol=1e-3,
        rtol=0
    )


def test_meia_vida_lambda_zero():
    # Lambda2 igual a zero representa decaimento imediato.

    resultado = calcular_meia_vida(0)

    assert resultado == 0.0


def test_meia_vida_lambda_um():
    # Lambda2 igual a 1 não possui meia-vida finita.

    resultado = calcular_meia_vida(1)

    assert np.isinf(resultado)


@pytest.mark.parametrize(
    "valor",
    [-0.1, 1.1, np.nan, np.inf]
)
def test_meia_vida_rejeita_valores_invalidos(valor):
    # Valores fora do intervalo permitido devem ser rejeitados.

    with pytest.raises(ValueError):
        calcular_meia_vida(valor)


# --------------------------------------------------
# TESTES DA DIAGONALIZAÇÃO
# --------------------------------------------------

def test_diagonalizacao_artificial():
    # Verifica P = V * Lambda * V inversa.

    resultado = diagonalizar_matriz(P_ARTIFICIAL)

    V = resultado["V"]
    Lambda = resultado["Lambda"]
    V_inversa = resultado["V_inversa"]

    reconstruida = V @ Lambda @ V_inversa

    np.testing.assert_allclose(
        reconstruida,
        P_ARTIFICIAL,
        atol=TOLERANCIA,
        rtol=0
    )

    assert resultado["erro_reconstrucao"] <= TOLERANCIA


def test_diagonalizacao_referencia():
    # Verifica a reconstrução da matriz do professor.

    espectro = calcular_espectro(P_REFERENCIA)

    resultado = diagonalizar_matriz(
        P_REFERENCIA,
        espectro
    )

    np.testing.assert_allclose(
        resultado["reconstruida"],
        P_REFERENCIA,
        atol=TOLERANCIA,
        rtol=0
    )


def test_matriz_autovetores_invertivel():
    # Verifica se V é invertível.

    resultado = diagonalizar_matriz(P_REFERENCIA)

    V = resultado["V"]

    assert np.linalg.matrix_rank(V) == 4

    identidade = V @ resultado["V_inversa"]

    np.testing.assert_allclose(
        identidade,
        np.eye(4),
        atol=TOLERANCIA,
        rtol=0
    )


def test_numero_condicionamento():
    # Confere a fórmula da norma matricial 1.

    resultado = diagonalizar_matriz(P_REFERENCIA)

    V = resultado["V"]
    V_inversa = resultado["V_inversa"]

    esperado = (
        np.linalg.norm(V, ord=1)
        * np.linalg.norm(V_inversa, ord=1)
    )

    assert np.isclose(
        resultado["condicao"],
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_rejeitar_matriz_nao_diagonalizavel():
    # Matriz com autovetores dependentes deve ser rejeitada.

    with pytest.raises(ValueError):
        diagonalizar_matriz(P_NAO_DIAGONALIZAVEL)


# --------------------------------------------------
# TESTES DE AUTOVALORES COMPLEXOS
# --------------------------------------------------

def test_autovalores_complexos():
    # Verifica a existência de autovalores complexos.

    resultado = calcular_espectro(P_COMPLEXA)

    autovalores = resultado["autovalores"]

    assert np.any(
        np.abs(np.imag(autovalores)) > TOLERANCIA
    )


def test_autovalores_complexos_conjugados():
    # Os autovalores complexos aparecem em pares conjugados.

    resultado = calcular_espectro(P_COMPLEXA)

    autovalores = resultado["autovalores"]

    for valor in autovalores:

        conjugado = np.conjugate(valor)

        assert np.any(
            np.isclose(
                autovalores,
                conjugado,
                atol=TOLERANCIA,
                rtol=0
            )
        )


def test_diagonalizacao_com_autovalores_complexos():
    # A diagonalização complexa deve reconstruir P real.

    resultado = diagonalizar_matriz(P_COMPLEXA)

    np.testing.assert_allclose(
        resultado["reconstruida"],
        P_COMPLEXA,
        atol=TOLERANCIA,
        rtol=0
    )


def test_converter_real_erro_numerico():
    # Pequenas partes imaginárias podem ser descartadas.

    matriz = np.array([
        [1 + 1e-14j, 0],
        [0, 1 - 1e-14j],
    ])

    resultado = converter_real(
        matriz,
        "matriz de teste"
    )

    np.testing.assert_allclose(
        resultado,
        np.eye(2),
        atol=TOLERANCIA,
        rtol=0
    )

    assert not np.iscomplexobj(resultado)


def test_converter_real_rejeita_imaginario_significativo():
    # Partes imaginárias relevantes não podem ser descartadas.

    matriz = np.array([
        [1 + 0.1j, 0],
        [0, 1],
    ])

    with pytest.raises(ValueError):
        converter_real(
            matriz,
            "matriz inválida"
        )


# --------------------------------------------------
# TESTES DE QUADRADOS SUCESSIVOS
# --------------------------------------------------

def test_potencia_zero():
    # Toda matriz elevada a zero resulta na identidade.

    resultado = potencia_quadrados(
        P_ARTIFICIAL,
        0
    )

    np.testing.assert_allclose(
        resultado,
        np.eye(4),
        atol=TOLERANCIA,
        rtol=0
    )


def test_potencia_um():
    # P elevado a 1 deve ser igual a P.

    resultado = potencia_quadrados(
        P_ARTIFICIAL,
        1
    )

    np.testing.assert_allclose(
        resultado,
        P_ARTIFICIAL,
        atol=TOLERANCIA,
        rtol=0
    )


@pytest.mark.parametrize(
    "expoente",
    [2, 3, 4, 5, 8, 16, 52]
)
def test_quadrados_sucessivos(expoente):
    # Compara o algoritmo explícito com NumPy.

    obtido = potencia_quadrados(
        P_ARTIFICIAL,
        expoente
    )

    # NumPy é utilizado somente na verificação independente.
    esperado = np.linalg.matrix_power(
        P_ARTIFICIAL,
        expoente
    )

    np.testing.assert_allclose(
        obtido,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


@pytest.mark.parametrize(
    "expoente",
    [-1, 1.5, "52"]
)
def test_quadrados_rejeita_expoente_invalido(expoente):
    # Expoentes negativos ou não inteiros são rejeitados.

    with pytest.raises(ValueError):
        potencia_quadrados(
            P_ARTIFICIAL,
            expoente
        )


# --------------------------------------------------
# TESTES DE POTÊNCIAS POR DIAGONALIZAÇÃO
# --------------------------------------------------

def test_potencia_diagonalizacao():
    # Calcula P^52 por diagonalização.

    diagonalizacao = diagonalizar_matriz(
        P_ARTIFICIAL
    )

    obtido = potencia_diagonalizacao(
        diagonalizacao,
        HORIZONTE
    )

    esperado = np.linalg.matrix_power(
        P_ARTIFICIAL,
        HORIZONTE
    )

    np.testing.assert_allclose(
        obtido,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_potencia_diagonalizacao_complexa():
    # Verifica P^52 quando existem autovalores complexos.

    diagonalizacao = diagonalizar_matriz(
        P_COMPLEXA
    )

    obtido = potencia_diagonalizacao(
        diagonalizacao,
        HORIZONTE
    )

    esperado = np.linalg.matrix_power(
        P_COMPLEXA,
        HORIZONTE
    )

    np.testing.assert_allclose(
        obtido,
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


@pytest.mark.parametrize(
    "expoente",
    [-1, 2.5]
)
def test_diagonalizacao_rejeita_expoente_invalido(expoente):
    # Verifica a rejeição de expoentes inválidos.

    diagonalizacao = diagonalizar_matriz(
        P_ARTIFICIAL
    )

    with pytest.raises(ValueError):
        potencia_diagonalizacao(
            diagonalizacao,
            expoente
        )


# --------------------------------------------------
# TESTES DA CONCORDÂNCIA DOS MÉTODOS
# --------------------------------------------------

def test_comparar_potencias_artificial():
    # Verifica a concordância dos dois métodos.

    diagonalizacao = diagonalizar_matriz(
        P_ARTIFICIAL
    )

    resultado = comparar_potencias(
        P_ARTIFICIAL,
        diagonalizacao,
        HORIZONTE
    )

    assert resultado["diferenca_l1"] <= TOLERANCIA


def test_comparar_potencias_complexas():
    # Verifica a concordância com autovalores complexos.

    diagonalizacao = diagonalizar_matriz(
        P_COMPLEXA
    )

    resultado = comparar_potencias(
        P_COMPLEXA,
        diagonalizacao,
        HORIZONTE
    )

    assert resultado["diferenca_l1"] <= TOLERANCIA


def test_norma_matricial_um():
    # Verifica a norma matricial 1 da diferença.

    diagonalizacao = diagonalizar_matriz(
        P_REFERENCIA
    )

    resultado = comparar_potencias(
        P_REFERENCIA,
        diagonalizacao,
        HORIZONTE
    )

    diferenca = (
        resultado["quadrados"]
        - resultado["diagonalizacao"]
    )

    esperado = np.linalg.norm(
        diferenca,
        ord=1
    )

    assert np.isclose(
        resultado["diferenca_l1"],
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DA CONVERGÊNCIA
# --------------------------------------------------

def test_convergencia_quatro_estados():
    # Verifica a existência das quatro distribuições iniciais.

    pi = estacionaria_sistema(P_ARTIFICIAL)

    dados = calcular_convergencia(
        P_ARTIFICIAL,
        pi,
        max_semanas=10
    )

    assert len(dados) == 44

    assert set(dados["estado_inicial"]) == {
        1, 2, 3, 4
    }

    assert dados["semana"].min() == 0

    assert dados["semana"].max() == 10


def test_convergencia_semana_zero():
    # Confere a distância inicial de cada estado.

    pi = estacionaria_sistema(P_ARTIFICIAL)

    dados = calcular_convergencia(
        P_ARTIFICIAL,
        pi,
        max_semanas=0
    )

    for estado in range(1, 5):

        linha = dados[
            dados["estado_inicial"] == estado
        ].iloc[0]

        vetor_inicial = np.eye(4)[estado - 1]

        esperado = np.linalg.norm(
            vetor_inicial - pi,
            ord=1
        )

        assert np.isclose(
            linha["distancia_l1"],
            esperado,
            atol=TOLERANCIA,
            rtol=0
        )


def test_convergencia_calculo_independente():
    # Confere as distâncias em várias semanas.

    pi = estacionaria_sistema(P_ARTIFICIAL)

    dados = calcular_convergencia(
        P_ARTIFICIAL,
        pi,
        max_semanas=5
    )

    for semana in range(6):

        potencia = np.linalg.matrix_power(
            P_ARTIFICIAL,
            semana
        )

        for estado in range(1, 5):

            distribuicao = potencia[estado - 1]

            esperado = np.linalg.norm(
                distribuicao - pi,
                ord=1
            )

            linha = dados[
                (dados["semana"] == semana)
                & (dados["estado_inicial"] == estado)
            ].iloc[0]

            assert np.isclose(
                linha["distancia_l1"],
                esperado,
                atol=TOLERANCIA,
                rtol=0
            )


def test_convergencia_distancias_diminuem():
    # Na matriz artificial, as distâncias diminuem.

    pi = estacionaria_sistema(P_ARTIFICIAL)

    dados = calcular_convergencia(
        P_ARTIFICIAL,
        pi,
        max_semanas=20
    )

    for estado in range(1, 5):

        distancias = dados[
            dados["estado_inicial"] == estado
        ]["distancia_l1"].to_numpy()

        diferencas = np.diff(distancias)

        assert np.all(diferencas <= TOLERANCIA)


def test_convergencia_rejeita_semanas_negativas():
    # O horizonte não pode ser negativo.

    pi = estacionaria_sistema(P_ARTIFICIAL)

    with pytest.raises(ValueError):
        calcular_convergencia(
            P_ARTIFICIAL,
            pi,
            max_semanas=-1
        )


# --------------------------------------------------
# TESTES DA GERAÇÃO DOS GRÁFICOS
# --------------------------------------------------

def test_gerar_grafico_convergencia(
    tmp_path,
    monkeypatch
):
    # Gera o gráfico somente na pasta temporária.

    monkeypatch.setattr(
        etapa4,
        "PASTA_FIGURAS",
        tmp_path
    )

    pi = estacionaria_sistema(P_ARTIFICIAL)

    convergencia = calcular_convergencia(
        P_ARTIFICIAL,
        pi,
        max_semanas=20
    )

    espectro = calcular_espectro(P_ARTIFICIAL)

    gerar_grafico_convergencia(
        "serra",
        convergencia,
        espectro["lambda2"]
    )

    arquivo = tmp_path / "e4_serra_convergencia.png"

    assert arquivo.exists()

    assert arquivo.stat().st_size > 0

    # Confere se o arquivo gerado pode ser lido.
    imagem = plt.imread(arquivo)

    assert imagem.ndim == 3


def test_grafico_com_distancias_zero(
    tmp_path,
    monkeypatch
):
    # Verifica que distâncias zero não impedem o gráfico.

    monkeypatch.setattr(
        etapa4,
        "PASTA_FIGURAS",
        tmp_path
    )

    pi = estacionaria_sistema(P_ARTIFICIAL)

    convergencia = calcular_convergencia(
        P_ARTIFICIAL,
        pi,
        max_semanas=10
    )

    # Insere uma distância zero artificial.
    convergencia.loc[
        convergencia.index[0],
        "distancia_l1"
    ] = 0.0

    gerar_grafico_convergencia(
        "serra",
        convergencia,
        0.6
    )

    arquivo = tmp_path / "e4_serra_convergencia.png"

    assert arquivo.exists()

    assert arquivo.stat().st_size > 0


# --------------------------------------------------
# TESTES DA GERAÇÃO DOS CSVs
# --------------------------------------------------

def test_salvar_matriz_csv(
    tmp_path,
    monkeypatch
):
    # Salva uma matriz artificial na pasta temporária.

    monkeypatch.setattr(
        etapa4,
        "PASTA_TABELAS",
        tmp_path
    )

    salvar_matriz(
        "teste",
        "matriz",
        P_ARTIFICIAL
    )

    arquivo = tmp_path / "e4_teste_matriz.csv"

    assert arquivo.exists()

    dados = pd.read_csv(
        arquivo,
        index_col=0
    )

    assert dados.shape == (4, 4)

    np.testing.assert_allclose(
        dados.to_numpy(dtype=float),
        P_ARTIFICIAL,
        atol=TOLERANCIA,
        rtol=0
    )


def test_salvar_matriz_complexa_csv(
    tmp_path,
    monkeypatch
):
    # Verifica se os valores complexos são preservados.

    monkeypatch.setattr(
        etapa4,
        "PASTA_TABELAS",
        tmp_path
    )

    matriz = np.array([
        [1 + 0j, 0],
        [0, 0.6 + 0.4j],
    ])

    salvar_matriz(
        "teste",
        "complexa",
        matriz
    )

    arquivo = tmp_path / "e4_teste_complexa.csv"

    assert arquivo.exists()

    dados = pd.read_csv(
        arquivo,
        index_col=0
    )

    obtido = dados.to_numpy(
        dtype=np.complex128
    )

    np.testing.assert_allclose(
        obtido,
        matriz,
        atol=TOLERANCIA,
        rtol=0
    )


def test_salvar_autovalores_csv(
    tmp_path,
    monkeypatch
):
    # Verifica o registro das partes real e imaginária.

    monkeypatch.setattr(
        etapa4,
        "PASTA_TABELAS",
        tmp_path
    )

    autovalores = linalg.eigvals(P_COMPLEXA)

    salvar_autovalores(
        "teste",
        autovalores
    )

    arquivo = tmp_path / "e4_teste_autovalores.csv"

    assert arquivo.exists()

    dados = pd.read_csv(arquivo)

    assert len(dados) == 4

    colunas = {
        "localidade",
        "indice",
        "parte_real",
        "parte_imaginaria",
        "modulo",
    }

    assert colunas.issubset(dados.columns)

    obtidos = (
        dados["parte_real"].to_numpy()
        + 1j * dados["parte_imaginaria"].to_numpy()
    )

    np.testing.assert_allclose(
        obtidos,
        autovalores,
        atol=TOLERANCIA,
        rtol=0
    )


# --------------------------------------------------
# TESTES DE INTEGRAÇÃO COM MATRIZES REAIS
# --------------------------------------------------

@pytest.mark.parametrize("nome", LOCALIDADES)
def test_espectro_matrizes_reais(nome):
    # Lê a matriz real gerada na Etapa 2.

    P = carregar_matriz(nome)

    resultado = calcular_espectro(P)

    assert len(resultado["autovalores"]) == 4

    assert resultado["erro_traco"] <= TOLERANCIA

    assert resultado["erro_determinante"] <= TOLERANCIA

    esperado = RESULTADOS_REAIS[nome]["lambda2"]

    assert np.isclose(
        resultado["lambda2"],
        esperado,
        atol=TOLERANCIA_REFERENCIA,
        rtol=0
    )


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_meia_vida_matrizes_reais(nome):
    # Compara a meia-vida com os resultados informados.

    P = carregar_matriz(nome)

    espectro = calcular_espectro(P)

    meia_vida = calcular_meia_vida(
        espectro["lambda2"]
    )

    esperado = RESULTADOS_REAIS[nome]["meia_vida"]

    assert np.isclose(
        meia_vida,
        esperado,
        atol=TOLERANCIA_APROXIMADA,
        rtol=0
    )


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_diagonalizacao_matrizes_reais(nome):
    # Verifica a reconstrução das matrizes reais.

    P = carregar_matriz(nome)

    resultado = diagonalizar_matriz(P)

    np.testing.assert_allclose(
        resultado["reconstruida"],
        P,
        atol=TOLERANCIA,
        rtol=0
    )

    assert resultado["erro_reconstrucao"] <= TOLERANCIA

    assert np.isfinite(resultado["condicao"])


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_condicionamento_matrizes_reais(nome):
    # Compara o condicionamento com os valores aproximados.

    P = carregar_matriz(nome)

    resultado = diagonalizar_matriz(P)

    esperado = RESULTADOS_REAIS[nome]["condicao"]

    # O condicionamento pode variar ligeiramente entre ambientes.
    assert np.isclose(
        resultado["condicao"],
        esperado,
        atol=1e-2,
        rtol=0
    )


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_p52_matrizes_reais(nome):
    # Calcula P^52 pelos dois métodos.

    P = carregar_matriz(nome)

    diagonalizacao = diagonalizar_matriz(P)

    resultado = comparar_potencias(
        P,
        diagonalizacao,
        HORIZONTE
    )

    assert resultado["diferenca_l1"] <= TOLERANCIA

    # Confere também com a implementação do NumPy.
    esperado = np.linalg.matrix_power(
        P,
        HORIZONTE
    )

    np.testing.assert_allclose(
        resultado["quadrados"],
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )

    np.testing.assert_allclose(
        resultado["diagonalizacao"],
        esperado,
        atol=TOLERANCIA,
        rtol=0
    )


def test_autovalores_complexos_serra():
    # Serra apresentou autovalores complexos conjugados.

    P = carregar_matriz("serra")

    resultado = calcular_espectro(P)

    autovalores = resultado["autovalores"]

    assert np.any(
        np.abs(np.imag(autovalores)) > TOLERANCIA
    )

    for valor in autovalores:

        assert np.any(
            np.isclose(
                autovalores,
                np.conjugate(valor),
                atol=TOLERANCIA,
                rtol=0
            )
        )


@pytest.mark.parametrize("nome", LOCALIDADES)
def test_convergencia_matrizes_reais(nome):
    # Verifica a convergência dos quatro estados.

    P = carregar_matriz(nome)

    pi = estacionaria_sistema(P)

    dados = calcular_convergencia(
        P,
        pi,
        max_semanas=52
    )

    assert len(dados) == 53 * 4

    assert set(dados["estado_inicial"]) == {
        1, 2, 3, 4
    }

    assert np.all(
        dados["distancia_l1"] >= 0
    )

    # Compara as distâncias da semana 0 e da semana 52.
    for estado in range(1, 5):

        dados_estado = dados[
            dados["estado_inicial"] == estado
        ]

        distancia_inicial = dados_estado[
            dados_estado["semana"] == 0
        ]["distancia_l1"].iloc[0]

        distancia_final = dados_estado[
            dados_estado["semana"] == 52
        ]["distancia_l1"].iloc[0]

        assert distancia_final < distancia_inicial


# --------------------------------------------------
# TESTES DA ANÁLISE COMPLETA
# --------------------------------------------------

@pytest.mark.parametrize("nome", LOCALIDADES)
def test_analisar_localidade(
    nome,
    tmp_path,
    monkeypatch
):
    # Executa a análise usando os CSVs reais somente para leitura.
    # Os resultados são gravados em pastas temporárias.

    pasta_tabelas = tmp_path / "tabelas"
    pasta_figuras = tmp_path / "figuras"

    pasta_tabelas.mkdir()
    pasta_figuras.mkdir()

    monkeypatch.setattr(
        etapa4,
        "PASTA_TABELAS",
        pasta_tabelas
    )

    monkeypatch.setattr(
        etapa4,
        "PASTA_FIGURAS",
        pasta_figuras
    )

    resumo = analisar_localidade(nome)

    assert resumo["localidade"] == nome

    assert resumo["verificacao_traco"]

    assert resumo["verificacao_determinante"]

    assert resumo["erro_traco"] <= TOLERANCIA

    assert resumo["erro_determinante"] <= TOLERANCIA

    assert resumo["diferenca_P52_l1"] <= TOLERANCIA

    assert resumo["erro_reconstrucao_l1"] <= TOLERANCIA

    # Confere os arquivos gerados.
    assert (
        pasta_tabelas / f"e4_{nome}_autovalores.csv"
    ).exists()

    assert (
        pasta_tabelas / f"e4_{nome}_V.csv"
    ).exists()

    assert (
        pasta_tabelas / f"e4_{nome}_Lambda.csv"
    ).exists()

    assert (
        pasta_tabelas / f"e4_{nome}_V_inversa.csv"
    ).exists()

    assert (
        pasta_tabelas / f"e4_{nome}_reconstrucao.csv"
    ).exists()

    assert (
        pasta_tabelas / f"e4_{nome}_P52_quadrados.csv"
    ).exists()

    assert (
        pasta_tabelas / f"e4_{nome}_P52_diagonalizacao.csv"
    ).exists()

    assert (
        pasta_tabelas / f"e4_{nome}_convergencia.csv"
    ).exists()

    assert (
        pasta_figuras / f"e4_{nome}_convergencia.png"
    ).exists()


def test_executar_etapa4(
    tmp_path,
    monkeypatch
):
    # Executa as três análises completas.
    # Mantém os arquivos gerados em pastas temporárias.

    pasta_tabelas = tmp_path / "tabelas"
    pasta_figuras = tmp_path / "figuras"

    monkeypatch.setattr(
        etapa4,
        "PASTA_TABELAS",
        pasta_tabelas
    )

    monkeypatch.setattr(
        etapa4,
        "PASTA_FIGURAS",
        pasta_figuras
    )

    executar()

    arquivo_resumo = (
        pasta_tabelas / "e4_resumo.csv"
    )

    arquivo_comparacao = (
        pasta_tabelas / "e4_comparacao_localidades.csv"
    )

    assert arquivo_resumo.exists()

    assert arquivo_comparacao.exists()

    resumo = pd.read_csv(arquivo_resumo)

    comparacao = pd.read_csv(arquivo_comparacao)

    assert len(resumo) == 3

    assert len(comparacao) == 3

    assert set(resumo["localidade"]) == set(LOCALIDADES)

    assert resumo["verificacao_traco"].all()

    assert resumo["verificacao_determinante"].all()

    assert (
        resumo["diferenca_P52_l1"] <= TOLERANCIA
    ).all()

    # Confere a ordem crescente das meias-vidas.
    meias_vidas = comparacao[
        "meia_vida_semanas"
    ].to_numpy()

    assert np.all(
        np.diff(meias_vidas) >= 0
    )


# [/IA]