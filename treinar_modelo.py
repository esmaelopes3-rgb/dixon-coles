# -*- coding: utf-8 -*-
"""
treinar_modelo.py — Treina o Dixon-Coles e salva em disco.

Rode 1x por semana via cron. Não chama LLM nem gasta análise.
Só precisa de resultados históricos (FT/AET/PEN).

Exemplo de cron (toda segunda às 05:00 Brasília = 08:00 UTC):
    0 8 * * 1 cd /caminho/footbet && /caminho/venv/bin/python treinar_modelo.py >> logs/treino.log 2>&1

Substitua `coletar_historico()` pela sua fonte real (API-Football
ou CSV). O resto é genérico.
"""

import logging
from datetime import datetime

from agente_dixon_coles import ModeloDixonColes, jogos_de_api_football

# >>> ajuste para o seu config.py <<<
# from config import API_FOOTBALL_KEY, LIGAS, CAMINHO_MODELO
CAMINHO_MODELO = "modelo_dc.pkl"
TEMPORADAS = [2023, 2024, 2025]  # janela de treino

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("treino")


def coletar_historico() -> list[dict]:
    """
    PONTO DE INTEGRAÇÃO 1.
    Devolva a lista de jogos finalizados no formato do modelo:
        {"data","casa","fora","gols_casa","gols_fora"}

    Opção A — API-Football (você já usa):
        fixtures = []
        for liga in LIGAS:
            for temp in TEMPORADAS:
                resp = requests.get(
                    "https://v3.football.api-sports.io/fixtures",
                    headers={"x-apisports-key": API_FOOTBALL_KEY},
                    params={"league": liga, "season": temp, "status": "FT"},
                ).json()["response"]
                fixtures += resp
        return jogos_de_api_football(fixtures)

    Opção B — CSV grátis (sem gastar API): ver demo no fim do
    agente_dixon_coles.py.
    """
    raise NotImplementedError(
        "Implemente coletar_historico() com sua fonte de dados."
    )


def main():
    log.info("Iniciando treino do modelo Dixon-Coles")
    jogos = coletar_historico()
    log.info("Jogos coletados: %d", len(jogos))

    if len(jogos) < 200:
        log.warning("Poucos jogos (%d). Previsões serão instáveis.", len(jogos))

    modelo = ModeloDixonColes()
    modelo.treinar(jogos)
    modelo.salvar(CAMINHO_MODELO)

    log.info("Modelo salvo em %s | %d times | treino %s",
             CAMINHO_MODELO, len(modelo.times), datetime.now().isoformat())


if __name__ == "__main__":
    main()
