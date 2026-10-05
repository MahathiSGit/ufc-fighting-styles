"""Fight Predictor web page.   Run:  streamlit run app/app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predictor as pr  # noqa: E402

st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="centered")


@st.cache_resource
def assets():
    state, bundle = pr.load_assets()
    return state.set_index("fighter_id", drop=False), bundle, pr.fighter_options(state)


state, bundle, options = assets()
st.title("🥊 UFC Fight Predictor")
st.caption(f"Model: {bundle['name']} · trained on {bundle['n_train']:,} UFC fights through {bundle['trained_through']}. "
           "Uses only what was known before the fight: past results, rates and ratings.")

labels = list(options)
c1, c2 = st.columns(2)
a_lab = c1.selectbox("Fighter A", labels, index=None, placeholder="Type a name...")
b_lab = c2.selectbox("Fighter B", labels, index=None, placeholder="Type a name...")
date = st.date_input("Fight date (age and layoff are measured on this date)", value=pd.Timestamp.today())

if a_lab and b_lab:
    if options[a_lab] == options[b_lab]:
        st.warning("Pick two different fighters.")
    else:
        a, b = state.loc[options[a_lab]], state.loc[options[b_lab]]
        r = pr.predict_fight(a, b, bundle, date)
        st.subheader(f"{a['name']}  vs  {b['name']}")
        m1, m2 = st.columns(2)
        m1.metric(a["name"], f"{r['p_a']:.0%}")
        m2.metric(b["name"], f"{r['p_b']:.0%}")
        st.progress(float(r["p_a"]))

        st.markdown("**What drove this prediction**")
        for c, v in r["drivers"][:5]:
            who = a["name"] if v > 0 else b["name"]
            st.write(f"- {pr.LABELS.get(c, c)} favours **{who}** ({abs(v):.2f})")

        st.markdown("**Side-by-side (stats are from each fighter's past UFC fights)**")
        t = r["table"].copy()
        t[a["name"]] = t[a["name"]].astype(float).round(2)
        t[b["name"]] = t[b["name"]].astype(float).round(2)
        st.dataframe(t, hide_index=True, width="stretch")

        thin = min(a["n_fights"], b["n_fights"]) <= 2
        if thin:
            st.info("One fighter has 2 or fewer UFC fights, so the stats rest on very little history. Treat this prediction with extra caution.")
        st.caption("Held-out accuracy on 2024-2026 fights was about 63% - better than the 55% from always picking "
                   "the red corner, but far from certain. Injuries, camp changes, weight cuts and styles are not modelled. "
                   "For learning purposes only, not betting advice.")
else:
    st.write("Choose two fighters to compare.")
