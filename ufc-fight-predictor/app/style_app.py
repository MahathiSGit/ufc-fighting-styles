"""Style-aware Fight Predictor page.   Run:  streamlit run app/style_app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predictor as pr  # noqa: E402
import style_predictor as sp  # noqa: E402

ROOT = HERE.parent
st.set_page_config(page_title="Style-Aware UFC Predictor", page_icon="🥋", layout="centered")


@st.cache_resource
def assets():
    state, _ = pr.load_assets()
    profiles = pd.read_csv(ROOT / "data" / "processed" / "style_profiles.csv").set_index("style")
    return state.set_index("fighter_id", drop=False), sp.load_style_assets(), pr.fighter_options(state), profiles


state, bundle, options, profiles = assets()
names = bundle["assets"]["names"]
sk, st_m = bundle["skill"]["test_metrics"], bundle["style"]["test_metrics"]

st.title("🥋 Style-Aware UFC Predictor")
st.caption("Does fighting style add anything beyond skill? This page predicts with a skill-only model and a "
           "skill + style model, so you can compare them for any matchup.")

labels = list(options)
c1, c2 = st.columns(2)
a_lab = c1.selectbox("Fighter A", labels, index=None, placeholder="Type a name...")
b_lab = c2.selectbox("Fighter B", labels, index=None, placeholder="Type a name...")
date = st.date_input("Fight date", value=pd.Timestamp.today())

if a_lab and b_lab and options[a_lab] != options[b_lab]:
    a, b = state.loc[options[a_lab]], state.loc[options[b_lab]]
    r = sp.predict_with_style(a, b, bundle, date)
    nm = lambda s: names[s] if s >= 0 else "Unknown"

    st.subheader(f"{a['name']}  vs  {b['name']}")
    s1, s2 = st.columns(2)
    s1.markdown(f"**{a['name']}**  \nStyle: `{nm(r['style_a'])}`")
    s2.markdown(f"**{b['name']}**  \nStyle: `{nm(r['style_b'])}`")
    if r["style_a"] < 0 or r["style_b"] < 0:
        st.info("A style is only assigned after 3+ fights with detailed stats, so at least one fighter is 'Unknown'. "
                "The skill + style model then relies on skill alone for that fighter.")

    p1, p2 = st.columns(2)
    p1.metric("Skill-only model", f"{r['p_skill']:.0%}", help=f"Probability that {a['name']} wins")
    p2.metric("Skill + style model", f"{r['p_style']:.0%}", delta=f"{100 * (r['p_style'] - r['p_skill']):+.1f} pts vs skill-only",
              help=f"Probability that {a['name']} wins")
    st.progress(float(r["p_style"]))

    st.markdown("**Style matchup**")
    mu = r["matchup"]
    if mu:
        st.write(f"In 2014-2022 UFC fights, **{nm(r['style_a'])}** fighters faced **{nm(r['style_b'])}** fighters "
                 f"{mu['n']} times and won **{mu['edge_pp']:+.1f} points** more than their Elo ratings predicted "
                 f"(a positive number favours {a['name']}'s style).")
        if mu["n"] < 100:
            st.caption(f"Only {mu['n']} fights, so this estimate is noisy.")
    else:
        st.write("No style-vs-style history applies (same style, or a style is unknown).")
    with st.expander("See the full style-vs-style chart"):
        st.image(str(ROOT / "reports" / "figures" / "style_matrix.png"))
        st.dataframe(profiles.round(2), width="stretch")

    st.markdown("**What drove the skill + style prediction**")
    for c, v in r["drivers"][:6]:
        st.write(f"- {c}: favours **{a['name'] if v > 0 else b['name']}** ({abs(v):.2f})")

    st.markdown("**Side-by-side stats (past UFC fights only)**")
    t = pr.predict_fight(a, b, bundle["skill"], date)["table"]
    for n_ in (a["name"], b["name"]):
        t[n_] = t[n_].astype(float).round(2)
    st.dataframe(t, hide_index=True, width="stretch")

    st.warning(f"**Honest result:** on 1,429 held-out 2024-2026 fights, the skill-only model scored {sk['acc']:.1%} accuracy "
               f"and the skill + style model {st_m['acc']:.1%}. That gap is far smaller than the uncertainty "
               "(about ±2.5 points), so there is no reliable evidence that style improves predictions beyond skill. "
               "Differences between the two models for a single fight are mostly noise. Learning project, not betting advice.")
elif a_lab and b_lab:
    st.warning("Pick two different fighters.")
else:
    st.write("Choose two fighters to compare.")
