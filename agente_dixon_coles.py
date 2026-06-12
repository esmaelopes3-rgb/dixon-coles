# -*- coding: utf-8 -*-
"""
agente_dixon_coles.py — Modelo estatístico Dixon-Coles para o FootBet.

Substitui/complementa o Agente Estatístico com probabilidades calibradas
em vez de depender só do LLM. Usa a biblioteca penaltyblog (v1.11+).

Instalação na VM (dentro do venv, sem sudo):
    pip install penaltyblog

Fluxo:
    1. Coletar resultados históricos via API-Football (últimas 2-3 temporadas)
    2. jogos = jogos_de_api_football(fixtures_json)
    3. modelo = ModeloDixonColes(); modelo.treinar(jogos)
    4. previsao = modelo.prever("Flamengo", "Palmeiras")
    5. valor = detectar_valor(previsao, odds_1x2=[2.10, 3.30, 3.60])
    6. Passar previsao + valor como contexto pro Agente Juiz

Retreinar 1x por semana via cron é suficiente (o modelo salva em pickle).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import penaltyblog as pb

# Meia-vida do decaimento temporal. xi=0.0019 ~ jogo de 1 ano atrás
# pesa ~50% de um jogo de hoje. Valor clássico do paper Dixon-Coles.
XI_PADRAO = 0.0019


class ModeloDixonColes:
    def __init__(self, xi: float = XI_PADRAO):
        self.xi = xi
        self.model = None
        self.times = set()

    def treinar(self, jogos: list[dict]) -> None:
        """
        jogos: lista de dicts com chaves:
            data (str ISO ou datetime), casa, fora, gols_casa, gols_fora
        """
        df = pd.DataFrame(jogos)
        df["data"] = pd.to_datetime(df["data"])
        df = df.sort_values("data").reset_index(drop=True)

        # IMPORTANTE: penaltyblog usa Cython e exige arrays float64
        # contíguos e graváveis — sem isso dá "buffer source array is read-only"
        gh = np.ascontiguousarray(df["gols_casa"].astype(int), dtype=np.float64)
        ga = np.ascontiguousarray(df["gols_fora"].astype(int), dtype=np.float64)
        pesos = np.ascontiguousarray(
            pb.models.dixon_coles_weights(df["data"], xi=self.xi),
            dtype=np.float64,
        )

        self.model = pb.models.DixonColesGoalModel(
            gh, ga, df["casa"].to_numpy(), df["fora"].to_numpy(), pesos
        )
        self.model.fit()
        self.times = set(df["casa"]) | set(df["fora"])

    def prever(self, casa: str, fora: str) -> dict:
        """Retorna dict JSON-friendly com os principais mercados."""
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes.")
        for t in (casa, fora):
            if t not in self.times:
                raise ValueError(
                    f"Time '{t}' não está no histórico. "
                    f"Confira o nome exato usado no treino."
                )

        p = self.model.predict(casa, fora)
        ph, px, pa = p.home_draw_away

        # Top 5 placares mais prováveis
        placares = []
        for h in range(6):
            for a in range(6):
                placares.append((f"{h}x{a}", round(p.exact_score(h, a), 4)))
        placares.sort(key=lambda x: -x[1])

        return {
            "jogo": f"{casa} x {fora}",
            "1x2": {
                "casa": round(ph, 4),
                "empate": round(px, 4),
                "fora": round(pa, 4),
            },
            "odds_justas": {
                "casa": round(1 / ph, 2),
                "empate": round(1 / px, 2),
                "fora": round(1 / pa, 2),
            },
            "gols": {
                "over_1.5": round(p.total_goals("over", 1.5), 4),
                "over_2.5": round(p.total_goals("over", 2.5), 4),
                "under_2.5": round(p.total_goals("under", 2.5), 4),
                "over_3.5": round(p.total_goals("over", 3.5), 4),
            },
            "btts": {
                "sim": round(p.btts_yes, 4),
                "nao": round(p.btts_no, 4),
            },
            "dupla_chance": {
                "1x": round(p.double_chance_1x, 4),
                "12": round(p.double_chance_12, 4),
                "x2": round(p.double_chance_x2, 4),
            },
            "dnb": {
                "casa": round(p.draw_no_bet_home, 4),
                "fora": round(p.draw_no_bet_away, 4),
            },
            "placares_provaveis": placares[:5],
        }

    def salvar(self, caminho: str = "modelo_dc.pkl") -> None:
        with open(caminho, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def carregar(caminho: str = "modelo_dc.pkl") -> "ModeloDixonColes":
        with open(caminho, "rb") as f:
            return pickle.load(f)


def jogos_de_api_football(fixtures: list[dict]) -> list[dict]:
    """
    Converte a resposta de fixtures da API-Football
    (endpoint /fixtures, lista em response[]) para o formato do modelo.
    Ignora jogos não finalizados.
    """
    jogos = []
    for fx in fixtures:
        status = fx["fixture"]["status"]["short"]
        if status not in ("FT", "AET", "PEN"):
            continue
        jogos.append({
            "data": fx["fixture"]["date"][:10],
            "casa": fx["teams"]["home"]["name"],
            "fora": fx["teams"]["away"]["name"],
            "gols_casa": fx["goals"]["home"],
            "gols_fora": fx["goals"]["away"],
        })
    return jogos


def detectar_valor(
    previsao: dict,
    odds_1x2: list[float],
    ev_minimo: float = 0.03,
) -> dict:
    """
    Compara o modelo com as odds do mercado.
    EV = prob_modelo * odd - 1. Positivo = aposta de valor.
    Também remove a margem da casa pra mostrar a prob. implícita real.
    """
    imp = pb.implied.calculate_implied(odds_1x2, method="power")
    nomes = ["casa", "empate", "fora"]
    resultado = {"margem_casa_pct": round(imp.margin * 100, 2), "mercados": []}

    for nome, odd, p_mercado in zip(nomes, odds_1x2, imp.probabilities):
        p_modelo = previsao["1x2"][nome]
        ev = p_modelo * odd - 1
        resultado["mercados"].append({
            "mercado": nome,
            "odd": odd,
            "prob_mercado": round(p_mercado, 4),
            "prob_modelo": p_modelo,
            "ev": round(ev, 4),
            "valor": ev >= ev_minimo,
        })
    return resultado


if __name__ == "__main__":
    # Demo com dados reais do Brasileirão (2021-2024)
    import urllib.request

    url = ("https://raw.githubusercontent.com/adaoduque/"
           "Brasileirao_Dataset/master/campeonato-brasileiro-full.csv")
    urllib.request.urlretrieve(url, "/tmp/brasileirao.csv")

    df = pd.read_csv("/tmp/brasileirao.csv")
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df[df["data"] >= "2021-01-01"]

    jogos = [
        {"data": r["data"], "casa": r["mandante"], "fora": r["visitante"],
         "gols_casa": int(r["mandante_Placar"]),
         "gols_fora": int(r["visitante_Placar"])}
        for _, r in df.iterrows()
    ]

    modelo = ModeloDixonColes()
    modelo.treinar(jogos)
    modelo.salvar("/tmp/modelo_dc.pkl")

    prev = modelo.prever("Flamengo", "Palmeiras")
    import json
    print(json.dumps(prev, indent=2, ensure_ascii=False))

    valor = detectar_valor(prev, odds_1x2=[2.10, 3.30, 3.60])
    print(json.dumps(valor, indent=2, ensure_ascii=False))
