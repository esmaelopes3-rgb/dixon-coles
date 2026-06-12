# -*- coding: utf-8 -*-
"""
testar_implantacao.py — Valida a instalação completa na VM.

Roda SEM nenhuma chave de API (usa dados grátis do GitHub).
Se terminar com "TUDO OK", pode plugar no bot com segurança.

Uso:
    source venv/bin/activate
    python testar_implantacao.py
"""

import json
import sys
import traceback

PASSOS_OK = []


def passo(nome):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                PASSOS_OK.append(nome)
                print(f"  [OK] {nome}")
            except Exception as e:
                print(f"\n  [FALHOU] {nome}")
                print(f"  Erro: {e}\n")
                traceback.print_exc()
                print("\nCorrija o erro acima antes de continuar.")
                sys.exit(1)
        return wrapper
    return decorator


@passo("1/6 Importar penaltyblog")
def t1():
    import penaltyblog as pb
    versao = tuple(int(x) for x in pb.__version__.split(".")[:2])
    assert versao >= (1, 11), (
        f"penaltyblog {pb.__version__} é antigo demais. "
        "Rode: pip install -U 'penaltyblog>=1.11'"
    )


@passo("2/6 Importar módulos do projeto")
def t2():
    import agente_dixon_coles  # noqa
    import pipeline_hibrido    # noqa
    import modelo_selecoes     # noqa


@passo("3/6 Treinar com Brasileirão (CSV grátis)")
def t3():
    import urllib.request
    import pandas as pd
    from agente_dixon_coles import ModeloDixonColes

    url = ("https://raw.githubusercontent.com/adaoduque/"
           "Brasileirao_Dataset/master/campeonato-brasileiro-full.csv")
    urllib.request.urlretrieve(url, "/tmp/brasileirao_teste.csv")
    df = pd.read_csv("/tmp/brasileirao_teste.csv")
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df[df["data"] >= "2021-01-01"]

    jogos = [
        {"data": r["data"], "casa": r["mandante"], "fora": r["visitante"],
         "gols_casa": int(r["mandante_Placar"]),
         "gols_fora": int(r["visitante_Placar"])}
        for _, r in df.iterrows()
    ]
    m = ModeloDixonColes()
    m.treinar(jogos)
    m.salvar("modelo_dc.pkl")

    p = m.prever("Flamengo", "Palmeiras")
    soma = sum(p["1x2"].values())
    assert 0.99 < soma < 1.01, f"Probabilidades 1x2 não somam 1: {soma}"


@passo("4/6 Treinar com seleções (Copa 2026)")
def t4():
    from agente_dixon_coles import ModeloDixonColes
    from modelo_selecoes import carregar_jogos_selecoes, prever_neutro

    jogos = carregar_jogos_selecoes("/tmp/selecoes_teste.csv")
    assert len(jogos) > 3000, f"Poucos jogos de seleções: {len(jogos)}"

    m = ModeloDixonColes(xi=0.003)
    m.treinar(jogos)
    m.salvar("modelo_copa.pkl")

    p = prever_neutro(m, "Brazil", "Argentina")
    soma = sum(p["1x2"].values())
    assert 0.99 < soma < 1.01, f"Probabilidades neutras não somam 1: {soma}"


@passo("5/6 Detecção de valor (Mercado)")
def t5():
    from agente_dixon_coles import ModeloDixonColes, detectar_valor
    m = ModeloDixonColes.carregar("modelo_dc.pkl")
    prev = m.prever("Flamengo", "Palmeiras")
    v = detectar_valor(prev, odds_1x2=[2.10, 3.30, 3.60])
    assert "margem_casa_pct" in v and len(v["mercados"]) == 3


@passo("6/6 Pipeline completo com juiz simulado")
def t6():
    from agente_dixon_coles import ModeloDixonColes
    from pipeline_hibrido import pipeline_completo

    m = ModeloDixonColes.carregar("modelo_dc.pkl")

    def juiz_fake(system, user):
        return json.dumps({"recomendacao": "passar", "mercado": None,
                           "confianca_0a10": 3, "justificativa": "teste"})

    r = pipeline_completo(m, "Flamengo", "Palmeiras",
                          [2.10, 3.30, 3.60], juiz_fake)
    assert r["veredito"]["recomendacao"] == "passar"
    assert "PROBABILIDADES DO MODELO" in r["contexto_enviado_ao_juiz"]


if __name__ == "__main__":
    print("Validando implantação do módulo Dixon-Coles...\n")
    t1(); t2(); t3(); t4(); t5(); t6()
    print(f"\nTUDO OK — {len(PASSOS_OK)}/6 passos validados.")
    print("Modelos salvos: modelo_dc.pkl (Brasileirão) e "
          "modelo_copa.pkl (Copa 2026).")
    print("Próximo passo: plugar coletar_historico() e chamar_llm "
          "(ver GUIA_IMPLANTACAO.md).")
