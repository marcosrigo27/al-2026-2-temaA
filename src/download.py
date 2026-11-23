# Baixa e registra os dados brutos de dengue dos municípios do ES.

import hashlib
import io
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


# [IA] ChatGPT – 23/11/2026 – download e registro dos dados

URL_IBGE = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/32/municipios"
URL_INFODENGUE = "https://info.dengue.mat.br/api/alertcity"

# Usa caminhos relativos à pasta principal do projeto.
RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw"
MANIFESTO = RAIZ / "data" / "MANIFEST.txt"


def requisitar(session, url, parametros=None, tentativas=3):
    # Faz uma requisição com novas tentativas em caso de falha.
    for tentativa in range(1, tentativas + 1):
        try:
            resposta = session.get(url, params=parametros, timeout=60)
            resposta.raise_for_status()
            return resposta
        except requests.RequestException as erro:
            print(f"Erro na tentativa {tentativa}/{tentativas}: {erro}")
            if tentativa == tentativas:
                raise
        finally:
            # Pausa também quando ocorrer um erro.
            time.sleep(1)


def obter_municipios(session):
    # Consulta os códigos dos 78 municípios do Espírito Santo.
    resposta = requisitar(session, URL_IBGE)
    municipios = resposta.json()

    if not isinstance(municipios, list) or len(municipios) != 78:
        raise ValueError("A API do IBGE não retornou os 78 municípios esperados.")

    codigos = sorted(str(municipio["id"]) for municipio in municipios)

    if len(set(codigos)) != 78 or any(len(codigo) != 7 for codigo in codigos):
        raise ValueError("A lista do IBGE contém códigos inválidos ou repetidos.")

    return codigos


def registrar_manifesto(arquivo, url, linhas, sha256):
    # Acrescenta o registro de um arquivo baixado ao manifesto.
    momento = datetime.now().astimezone().isoformat(timespec="seconds")

    registro = (
        f"Data e hora: {momento}\n"
        f"Arquivo: {arquivo.relative_to(RAIZ).as_posix()}\n"
        f"URL: {url}\n"
        f"Linhas: {linhas}\n"
        f"SHA-256: {sha256}\n\n"
    )

    with MANIFESTO.open("a", encoding="utf-8") as manifesto:
        manifesto.write(registro)


def baixar_municipio(session, codigo):
    # Baixa o CSV original de um município sem modificar seu conteúdo
    destino = PASTA_RAW / f"{codigo}.csv"

    # Nunca substitui um arquivo bruto já existente.
    if destino.exists():
        print(f"Arquivo já existe, sem alteração: {destino.name}")
        return False

    parametros = {
        "geocode": codigo,
        "disease": "dengue",
        "format": "csv",
        "ew_start": 1,
        "ew_end": 53,
        "ey_start": 2015,
        "ey_end": 2025,
    }

    resposta = requisitar(session, URL_INFODENGUE, parametros)
    conteudo = resposta.content

    # Pandas é usado apenas para conferir o CSV, sem regravá-lo.
    try:
        dados = pd.read_csv(io.BytesIO(conteudo))
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeError
    ) as erro:
        raise ValueError(f"CSV inválido do município {codigo}: {erro}") from erro

    colunas = {str(coluna).strip().lower() for coluna in dados.columns}

    if dados.empty or not {"se", "nivel"}.issubset(colunas):
        raise ValueError(f"CSV vazio ou sem as colunas SE/nivel: {codigo}")

    sha256 = hashlib.sha256(conteudo).hexdigest()

    # 'xb' impede sobrescrever arquivos existentes.
    try:
        with destino.open("xb") as arquivo:
            arquivo.write(conteudo)
    except FileExistsError:
        print(f"Arquivo já existe, sem alteração: {destino.name}")
        return False

    registrar_manifesto(destino, resposta.url, len(dados), sha256)

    print(f"Salvo: {destino.name} | {len(dados)} linhas")

    return True


def main():
    # Executa o download dos dados dos municípios
    PASTA_RAW.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        codigos = obter_municipios(session)

        print(f"Municípios identificados: {len(codigos)}")

        sucessos = 0
        falhas = []

        for codigo in codigos:
            try:
                if baixar_municipio(session, codigo):
                    sucessos += 1

            except (
                requests.RequestException,
                ValueError,
                OSError
            ) as erro:
                falhas.append(codigo)
                print(f"Falha no município {codigo}: {erro}")

    print(f"Novos arquivos salvos: {sucessos}")

    if falhas:
        print(f"Municípios com falha: {', '.join(falhas)}")
        raise SystemExit(1)

    print("Download concluído. Arquivos originais preservados.")


if __name__ == "__main__":
    main()

# [/IA]