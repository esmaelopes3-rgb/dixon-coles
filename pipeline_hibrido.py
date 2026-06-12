# -*- coding: utf-8 -*-
"""
pipeline_hibrido.py — Une o modelo matemático com o juiz LLM.

Fluxo por jogo:
    1. Estatístico:  Dixon-Coles -> probabilidades calibradas
    2. Mercado:      compara com odds -> EV e flags de valor
    3. Juiz (LLM):   recebe TUDO já mastigado + contexto qualitativo
                     e dá o veredito final

A ideia central: o LLM NUNCA inventa probabilidade. Ele recebe os
números prontos do Dixon-Coles e só decide se confia, ajusta por
contexto (lesão, motivação, desfalque) e escolhe a melhor entrada.
Isso elimina a alucinação numérica que era o ponto fraco do bot só-LLM.
"""

import json

from agente_dixon_coles import ModeloDixonColes, detectar_valor


# ---------------------------------------------------------------------------
# 1 + 2. ESTATÍSTICO + MERCADO  (puro, sem LLM, instantâneo)
# ---------------------------------------------------------------------------
def analisar_jogo(
    modelo: ModeloDixonColes,
    casa: str,
    fora: str,
    odds_1x2: list[float] | None = None,
) -> dict:
    """Roda Dixon-Coles e, se houver odds, calcula valor. Sem LLM."""
    previsao = modelo.prever(casa, fora)
    bloco = {"previsao": previsao}
    if odds_1x2:
        bloco["valor"] = detectar_valor(previsao, odds_1x2)
    return bloco


# ---------------------------------------------------------------------------
# 3. JUIZ — monta o prompt e chama o LLM
# ---------------------------------------------------------------------------
def montar_contexto_juiz(
    analise: dict,
    contexto_qualitativo: str = "",
) -> str:
    """
    Transforma os números do modelo num bloco de texto limpo pro LLM.
    É isto que entra no prompt do Qwen — não a previsão crua em JSON.
    """
    p = analise["previsao"]
    g = p["gols"]
    linhas = [
        f"JOGO: {p['jogo']}",
        "",
        "PROBABILIDADES DO MODELO ESTATÍSTICO (Dixon-Coles, calibrado):",
        f"  Vitória casa : {p['1x2']['casa']:.1%}  (odd justa {p['odds_justas']['casa']})",
        f"  Empate       : {p['1x2']['empate']:.1%}  (odd justa {p['odds_justas']['empate']})",
        f"  Vitória fora : {p['1x2']['fora']:.1%}  (odd justa {p['odds_justas']['fora']})",
        f"  Over 2.5     : {g['over_2.5']:.1%}   |  Under 2.5: {g['under_2.5']:.1%}",
        f"  BTTS sim     : {p['btts']['sim']:.1%}",
        "  Placares mais prováveis: "
        + ", ".join(f"{s} ({pr:.1%})" for s, pr in p["placares_provaveis"][:3]),
    ]

    if "valor" in analise:
        v = analise["valor"]
        linhas += ["", f"MERCADO (margem da casa: {v['margem_casa_pct']}%):"]
        for m in v["mercados"]:
            tag = "  <<< VALOR DETECTADO" if m["valor"] else ""
            linhas.append(
                f"  {m['mercado']:7s} odd {m['odd']:.2f} | "
                f"modelo {m['prob_modelo']:.1%} vs mercado {m['prob_mercado']:.1%} | "
                f"EV {m['ev']:+.1%}{tag}"
            )

    if contexto_qualitativo:
        linhas += ["", "CONTEXTO QUALITATIVO (notícias, lesões, etc.):",
                   contexto_qualitativo]

    return "\n".join(linhas)


PROMPT_SISTEMA_JUIZ = """Você é o Juiz, o agente final de um bot de análise de apostas de futebol.

Você recebe probabilidades JÁ CALCULADAS por um modelo estatístico Dixon-Coles \
treinado em dados históricos reais. NUNCA invente ou recalcule probabilidades — \
elas são confiáveis e calibradas. Seu papel é:

1. Avaliar SE o contexto qualitativo (lesões, desfalques, motivação, sequência) \
justifica desviar do que o modelo diz.
2. Decidir se há uma entrada de valor real — só recomende apostas com EV positivo \
sinalizado pelo Mercado.
3. Ser honesto: se não há valor claro, diga "sem entrada" em vez de forçar um palpite.

Responda em JSON: {"recomendacao","mercado","confianca_0a10","justificativa"}.
Se não houver valor, recomendacao = "passar"."""


def consultar_juiz(contexto: str, chamar_llm) -> dict:
    """
    PONTO DE INTEGRAÇÃO 2.
    `chamar_llm` é a SUA função que fala com o Qwen. Assinatura esperada:
        chamar_llm(system: str, user: str) -> str   (devolve texto/JSON)

    Exemplo com seu cliente atual:
        def chamar_llm(system, user):
            return cliente_qwen.chat(system=system, user=user)
    """
    resposta = chamar_llm(PROMPT_SISTEMA_JUIZ, contexto)
    try:
        return json.loads(resposta)
    except (json.JSONDecodeError, TypeError):
        # fallback: devolve texto cru se o LLM não respeitar o JSON
        return {"recomendacao": "erro_parse", "raw": resposta}


# ---------------------------------------------------------------------------
# Orquestrador completo
# ---------------------------------------------------------------------------
def pipeline_completo(
    modelo: ModeloDixonColes,
    casa: str,
    fora: str,
    odds_1x2: list[float],
    chamar_llm,
    contexto_qualitativo: str = "",
) -> dict:
    analise = analisar_jogo(modelo, casa, fora, odds_1x2)
    contexto = montar_contexto_juiz(analise, contexto_qualitativo)
    veredito = consultar_juiz(contexto, chamar_llm)
    return {
        "analise_numerica": analise,
        "contexto_enviado_ao_juiz": contexto,
        "veredito": veredito,
    }


# ---------------------------------------------------------------------------
# Demo — testa tudo MENOS a chamada real do LLM (usa um juiz fake)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    modelo = ModeloDixonColes.carregar("modelo_dc.pkl")

    def juiz_fake(system, user):
        # Simula o Qwen: lê o contexto e devolve um JSON plausível.
        tem_valor = "VALOR DETECTADO" in user
        return json.dumps({
            "recomendacao": "apostar" if tem_valor else "passar",
            "mercado": "vitória fora" if tem_valor else None,
            "confianca_0a10": 7 if tem_valor else 3,
            "justificativa": ("Modelo aponta EV positivo na vitória fora e o "
                              "contexto não contraindica." if tem_valor
                              else "Sem valor claro nas linhas disponíveis."),
        }, ensure_ascii=False)

    resultado = pipeline_completo(
        modelo,
        casa="Flamengo", fora="Palmeiras",
        odds_1x2=[2.10, 3.30, 3.60],
        chamar_llm=juiz_fake,
        contexto_qualitativo="Palmeiras com força máxima; Flamengo sem o "
                             "zagueiro titular (suspenso).",
    )

    print("===== CONTEXTO ENVIADO AO JUIZ =====")
    print(resultado["contexto_enviado_ao_juiz"])
    print("\n===== VEREDITO =====")
    print(json.dumps(resultado["veredito"], indent=2, ensure_ascii=False))
