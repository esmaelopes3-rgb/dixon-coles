# -*- coding: utf-8 -*-
"""
modelo_selecoes.py — Dixon-Coles para SELEÇÕES (Copa do Mundo 2026).

Reusa o mesmo ModeloDixonColes do agente_dixon_coles.py — o que muda
é só a fonte de dados (jogos internacionais) e o tratamento de campo
neutro, já que a maioria dos jogos da Copa não tem mandante de verdade.

Dados: github.com/martj42/international_results (grátis, atualizado).
Nomes dos times em INGLÊS: "Mexico", "Brazil", "South Korea", etc.

Regra de uso na Copa:
  - Jogo dos anfitriões em casa (México, EUA, Canadá) -> prever()
  - Qualquer outro jogo (campo neutro)               -> prever_neutro()
"""

import urllib.request
from pathlib import Path

import pandas as pd

from agente_dixon_coles import ModeloDixonColes

URL_DADOS = ("https://raw.githubusercontent.com/martj42/"
             "international_results/master/results.csv")
INICIO_CICLO = "2022-08-01"  # pós-Copa do Catar


def carregar_jogos_selecoes(caminho_cache: str = "selecoes.csv") -> list[dict]:
    """Baixa (ou usa cache) e converte pro formato do modelo."""
    if not Path(caminho_cache).exists():
        urllib.request.urlretrieve(URL_DADOS, caminho_cache)

    df = pd.read_csv(caminho_cache)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= INICIO_CICLO]
    df = df.dropna(subset=["home_score", "away_score"])  # remove agendados

    return [
        {"data": r["date"], "casa": r["home_team"], "fora": r["away_team"],
         "gols_casa": int(r["home_score"]), "gols_fora": int(r["away_score"])}
        for _, r in df.iterrows()
    ]


def prever_neutro(modelo: ModeloDixonColes, time_a: str, time_b: str) -> dict:
    """
    Campo neutro: o Dixon-Coles embute vantagem de mando no predict().
    Solução padrão: prever nas duas direções e tirar a média.
    Retorna no mesmo formato de prever(), com 'casa' = time_a.
    """
    p_ab = modelo.prever(time_a, time_b)
    p_ba = modelo.prever(time_b, time_a)

    def m(x, y):
        return round((x + y) / 2, 4)

    px = m(p_ab["1x2"]["empate"], p_ba["1x2"]["empate"])
    pa = m(p_ab["1x2"]["casa"], p_ba["1x2"]["fora"])
    pb = m(p_ab["1x2"]["fora"], p_ba["1x2"]["casa"])
    # renormaliza pra somar 1
    s = pa + px + pb
    pa, px, pb = round(pa / s, 4), round(px / s, 4), round(pb / s, 4)

    return {
        "jogo": f"{time_a} x {time_b} (campo neutro)",
        "1x2": {"casa": pa, "empate": px, "fora": pb},
        "odds_justas": {"casa": round(1 / pa, 2),
                        "empate": round(1 / px, 2),
                        "fora": round(1 / pb, 2)},
        "gols": {
            k: m(p_ab["gols"][k], p_ba["gols"][k]) for k in p_ab["gols"]
        },
        "btts": {"sim": m(p_ab["btts"]["sim"], p_ba["btts"]["sim"]),
                 "nao": m(p_ab["btts"]["nao"], p_ba["btts"]["nao"])},
    }


if __name__ == "__main__":
    import json

    jogos = carregar_jogos_selecoes("/tmp/selecoes.csv")
    print(f"Treinando com {len(jogos)} jogos de seleções (ciclo 2022-2026)...")

    # xi maior = decaimento mais rápido (elencos de seleção mudam muito)
    modelo = ModeloDixonColes(xi=0.003)
    modelo.treinar(jogos)
    modelo.salvar("modelo_copa.pkl")

    # JOGO 1 de hoje: México x África do Sul — México joga EM CASA (Azteca)
    p1 = modelo.prever("Mexico", "South Africa")
    print("\n=== ABERTURA: México x África do Sul (Azteca, mando real) ===")
    print(json.dumps({k: p1[k] for k in ("1x2", "odds_justas", "gols")},
                     indent=2, ensure_ascii=False))

    # JOGO 2 de hoje: Coreia do Sul x Rep. Tcheca — campo NEUTRO (Guadalajara)
    p2 = prever_neutro(modelo, "South Korea", "Czech Republic")
    print("\n=== Coreia do Sul x Rep. Tcheca (campo neutro) ===")
    print(json.dumps(p2, indent=2, ensure_ascii=False))
