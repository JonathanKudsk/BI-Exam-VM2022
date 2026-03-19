"""
Stage 4 Business Application: Streamlit dashboard til VM 2022 analysen.
Kør fra projektmappen: streamlit run app_streamlit.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vm2022_pipeline import (
 FEATURE_COLS_V4,
 NOTEBOOK_MIN_MINUTES,
 load_players_off_v4,
 player_rank,
)

st.set_page_config(
 page_title="VM 2022 Messi vs. Mbappé",
 layout="wide",
 initial_sidebar_state="expanded",
)

FEATURE_LABELS_DA: dict[str, str] = {
 "nonpen_goals_per90": "Mål uden straffe pr. 90 min",
 "pen_goals_per90": "Straffemål pr. 90 min",
 "pens_won_per90": "Vundne straffe pr. 90 min",
 "assists_per90": "Assists pr. 90 min",
 "xg_assist_per90": "Forventede assists (xA) pr. 90 min",
 "npxg_net_per90": "npxG netto pr. 90 min",
 "games_played": "Kampe spillet (≈ 90 min enheder)",
 "sca_shots_per90": "SCA fra skud pr. 90 min",
 "sca_dribbles_per90": "SCA fra driblinger pr. 90 min",
 "sca_fouled_per90": "SCA efter frispark mod pr. 90 min",
 "pass_completion_factor_per90": "Pasningskvalitet (fuldført × pct) pr. 90 min",
 "passes_received_per90": "Modtagne afleveringer pr. 90 min",
 "progressive_passes_received_per90": "Progressive modtagne afleveringer pr. 90 min",
 "weighted_dribbles_completed_per90": "Vægtede fuldførte driblinger pr. 90 min",
 "miscontrols_per90": "Fejlbeherskelse pr. 90 min",
 "dispossessed_per90": "Bold tabt til modstander pr. 90 min",
 "touches_mid_3rd_per90": "Berøringer i midte tredjedel pr. 90 min",
 "touches_att_3rd_per90": "Berøringer i offensiv tredjedel pr. 90 min",
 "touches_att_pen_area_per90": "Berøringer i feltet pr. 90 min",
 "progressive_passes_per90": "Progressive afleveringer pr. 90 min",
 "passes_into_final_third_per90": "Afleveringer til sidste tredjedel pr. 90 min",
 "passes_into_penalty_area_per90": "Afleveringer ind i feltet pr. 90 min",
 "assisted_shots_per90": "Assisterede skud pr. 90 min",
 "shots_on_target_per90": "Skud på mål pr. 90 min",
 "shots_not_on_target_per90": "Skud uden for mål pr. 90 min",
 "aerials_won_per90": "Vundne dueller i luften pr. 90 min",
 "ball_recoveries_per90": "Boldgenvindinger pr. 90 min",
 "tackles_won_per90": "Vundne tacklinger pr. 90 min",
 "interceptions_per90": "Afleveringsafskæringer pr. 90 min",
 "blocked_passes_per90": "Blokerede afleveringer pr. 90 min",
 "fouled_per90": "Frispark vundet (foulet) pr. 90 min",
 "fouls_per90": "Begåede frispark pr. 90 min",
 "cards_yellow_per90": "Gule kort pr. 90 min",
 "cards_red_per90": "Røde kort pr. 90 min",
 "dribbled_past_per90": "Driblet forbi pr. 90 min",
 "offsides_per90": "Offsides pr. 90 min",
 "errors_per90": "Fejl pr. 90 min",
 "aerials_lost_per90": "Tabte hovedstødsdueller pr. 90 min",
 "pens_missed_per90": "Brændte straffe pr. 90 min",
}


def label_feature(col: str) -> str:
 return FEATURE_LABELS_DA.get(col, col.replace("_", " "))


@st.cache_data(show_spinner=False)
def run_analysis(min_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
 root = Path(__file__).resolve().parent
 return load_players_off_v4(data_dir=root, min_minutes=min_minutes)


def apply_position_filter(df: pd.DataFrame, positions: list[str]) -> pd.DataFrame:
 if not positions:
 return df
 return df[df["position"].isin(positions)].copy()


def decomposition_figure(
 contributions_row: pd.Series,
 top_n: int = 12,
) -> go.Figure:
 pos = contributions_row[contributions_row >= 0].sort_values(ascending=False).head(top_n)
 neg = contributions_row[contributions_row < 0].sort_values(ascending=True).head(top_n)

 fig = go.Figure()
 if len(pos):
 fig.add_bar(
 orientation="h",
 y=[label_feature(c) for c in pos.index],
 x=pos.values,
 name="Positive bidrag",
 marker_color="#2ecc71",
 )
 if len(neg):
 fig.add_bar(
 orientation="h",
 y=[label_feature(c) for c in neg.index],
 x=neg.values,
 name="Negative bidrag",
 marker_color="#e74c3c",
 )
 fig.update_layout(
 barmode="relative",
 margin=dict(l=220, r=24, t=40, b=40),
 xaxis_title="Pointbidrag (feature × vægt)",
 height=max(420, 28 * (len(pos) + len(neg) + 4)),
 legend_yanchor="top",
 legend_y=0.99,
 legend_x=0.99,
 paper_bgcolor="rgba(0,0,0,0)",
 plot_bgcolor="rgba(248,249,252,1)",
 )
 return fig


def main() -> None:
 st.markdown(
 """
 <style>
 .main header { font size: 1.85rem; font weight: 700; margin bottom: 0.25rem; }
 .subtle { color: #5c6370; font size: 1rem; }
 </style>
 """,
 unsafe_allow_html=True,
 )

 st.markdown('<p class="main header">VM 2022 Messi vs. Mbappé: Data vs. narrative</p>', unsafe_allow_html=True)
 st.markdown(
 '<p class="subtle">Stage 4: En interaktiv oversigt over den samme analyse som i projektets notebook '
 "(V4 score og clustering). Målet er at gøre resultaterne tilgængelige for læsere uden kodeerfaring.</p>",
 unsafe_allow_html=True,
 )

 st.sidebar.header("Indstillinger")
 min_minutes = st.sidebar.slider(
 "Minimum minutter på banen",
 min_value=270,
 max_value=700,
 value=NOTEBOOK_MIN_MINUTES,
 step=10,
 help="I notebooken bruges 270 minutter som standard for at undgå meget små stikprøver.",
 )
 position_filter = st.sidebar.multiselect(
 "Positioner",
 options=["FW", "MF"],
 default=["FW", "MF"],
 )
 top_n = st.sidebar.slider("Antal i toplisten", 5, 30, 15)

 players_off, contributions = run_analysis(min_minutes)
 filtered = apply_position_filter(players_off, position_filter)
 if filtered.empty:
 st.warning("Ingen spillere matcher filtrene. Vælg evt. begge positioner.")
 st.stop()

 score_col = "offensive_score_v4"
 cluster_col = "cluster_v4"
 ranked = filtered.sort_values(score_col, ascending=False).reset_index(drop=True)
 ranked["_label"] = ranked["player"] + " (" + ranked["team"] + ")"

 cluster_means = filtered.groupby(cluster_col)[FEATURE_COLS_V4].mean()
 cluster_score_mean = filtered.groupby(cluster_col)[score_col].mean()
 high_cluster = int(cluster_score_mean.idxmax())
 low_cluster = int(cluster_score_mean.idxmin())

 tab_over, tab_top, tab_player, tab_cluster, tab_duel = st.tabs(
 ["Overblik", "Top spillere", "Score for én spiller", "Clusters", "Messi vs. Mbappé"]
 )

 with tab_over:
 st.markdown(
 """
 **Kontekst:** Ved VM 2022 blev Lionel Messi kåret til *Player of the Tournament*, mens mange pegede på
 Kylian Mbappés mål og offensive output. Her samler jeg **objektive nøgletal** (normaliseret pr. 90 minutter)
 i én **offensiv V4 score** og grupperer spillere med **KMeans** præcis som i Stage 3 i notebooken.

 **Hvad viser appen?** Topliste, hvordan scoren er bygget op for en valgfri spiller, hvilken **cluster**
 spilleren ligger i, og en direkte sammenligning mellem Messi og Mbappé.
 """
 )
 m_row = filtered[filtered["player"].str.contains("Messi", case=False, na=False)]
 mb_row = filtered[filtered["player"].str.contains("Mbappé", case=False, na=False)]
 if not m_row.empty and not mb_row.empty:
 m = m_row.iloc[0]
 mb = mb_row.iloc[0]
 c1, c2, c3 = st.columns(3)
 c1.metric("Messi offensiv score (V4)", f"{m[score_col]:.2f}")
 c2.metric("Mbappé offensiv score (V4)", f"{mb[score_col]:.2f}")
 c3.metric(
 "Forskel (Mbappé − Messi)",
 f"{mb[score_col]-m[score_col]:+.2f}",
 )
 st.info(
 f"Data genberegnes fra CSV filerne i projektmappen med samme formler som i notebooken "
 f"(minimum {min_minutes} min., positioner: {', '.join(position_filter) or ''})."
 )

 with tab_top:
 st.subheader("Bedst efter offensiv V4 score")
 st.caption(
 "V4 balancerer mål, assists, forventede mål/assists, involvering fremad på banen og "
 "trækker fra for boldtab, kort og andre hændelser se notebooken for fuld argumentation."
 )
 head = ranked.head(top_n)
 fig_top = go.Figure(
 go.Bar(
 x=head[score_col].iloc[::-1],
 y=head["_label"].iloc[::-1],
 orientation="h",
 marker_color="#3498db",
 text=[f"{v:.2f}" for v in head[score_col].iloc[::-1]],
 textposition="outside",
 )
 )
 fig_top.update_layout(
 height=28 * top_n + 80,
 margin=dict(l=8, r=80, t=24, b=40),
 xaxis_title="Offensiv score (V4)",
 paper_bgcolor="rgba(0,0,0,0)",
 plot_bgcolor="rgba(248,249,252,1)",
 )
 st.plotly_chart(fig_top, use_container_width=True)
 show_cols = [
 "player",
 "team",
 "position",
 "minutes",
 "goals",
 "assists",
 score_col,
 cluster_col,
 ]
 st.dataframe(
 head[show_cols],
 use_container_width=True,
 hide_index=True,
 )

 with tab_player:
 st.subheader("Hvad driver spillerscoren?")
 choice = st.selectbox(
 "Vælg spiller",
 options=ranked["_label"].tolist(),
 index=0,
 )
 row = ranked[ranked["_label"] == choice].iloc[0]
 pmask = (filtered["player"] == row["player"]) & (filtered["team"] == row["team"])
 true_idx = filtered[pmask].index[0]
 contrib_row = contributions.loc[true_idx, FEATURE_COLS_V4]
 rk = player_rank(filtered[score_col], true_idx)

 c1, c2, c3, c4 = st.columns(4)
 c1.metric("Rang (i aktuelt udvalg)", f"#{rk}")
 c2.metric("Offensiv score (V4)", f"{row[score_col]:.2f}")
 c3.metric("Cluster", f"{int(row[cluster_col])}")
 c4.metric("Minutter", f"{int(row['minutes'])}")
 st.plotly_chart(decomposition_figure(contrib_row), use_container_width=True)

 pos_sum = contrib_row[contrib_row >= 0].sum()
 neg_sum = contrib_row[contrib_row < 0].sum()
 st.markdown(
 f"**Opsummering:** Positive bidrag summerer til **+{pos_sum:.2f}** point, "
 f"negative til **{neg_sum:.2f}**. Totalen matcher spillerscoren (**{row[score_col]:.2f}**)."
 )

 with tab_cluster:
 st.subheader("KMeans profiler (V4)")
 st.markdown(
 f"Jeg bruger **to clusters** på standardiserede V4-features (som i notebooken). "
 f"Baseret på gennemsnitlig offensiv score kan **cluster {high_cluster}** beskrives som den mest "
 f"offensivt tunge gruppe, og **cluster {low_cluster}** som den anden profil."
 )
 cc1, cc2 = st.columns(2)
 vc = filtered[cluster_col].value_counts().sort_index()
 cc1.metric("Spillere i cluster 0", int(vc.get(0, 0)))
 cc2.metric("Spillere i cluster 1", int(vc.get(1, 0)))

 colors = filtered[cluster_col].map({0: "#9b59b6", 1: "#f39c12"})
 fig_sc = go.Figure(
 go.Scatter(
 x=filtered["minutes"],
 y=filtered[score_col],
 mode="markers",
 marker=dict(size=9, color=colors, opacity=0.75),
 text=filtered["player"] + " (" + filtered["team"] + ")",
 hovertemplate="%{text}<br>Minutter: %{x}<br>Score: %{y:.2f}<extra></extra>",
 )
 )
 fig_sc.update_layout(
 xaxis_title="Minutter",
 yaxis_title="Offensiv score (V4)",
 height=480,
 paper_bgcolor="rgba(0,0,0,0)",
 plot_bgcolor="rgba(248,249,252,1)",
 )
 st.plotly_chart(fig_sc, use_container_width=True)
 st.caption("Lilla og orange svarer til cluster 0 og 1 (farver som i notebook visualiseringer).")

 st.markdown("**Gennemsnit pr. feature (kun udvalgte se notebook for fuld tabel)**")
 highlight = [
 "nonpen_goals_per90",
 "assists_per90",
 "xg_assist_per90",
 "npxg_net_per90",
 "progressive_passes_received_per90",
 "weighted_dribbles_completed_per90",
 "miscontrols_per90",
 "dispossessed_per90",
 ]
 disp = cluster_means[highlight].T.round(3)
 disp.index = [label_feature(c) for c in disp.index]
 st.dataframe(disp, use_container_width=True)

 st.markdown(
 "**Fortolkning (som i problemformuleringen):** Ligger Messi og Mbappé i samme cluster, deler de "
 "i praksis samme *type* offensive profil ifølge modellen selv om den ene kan have højere totalscore. "
 "Det er et datapunkt i diskussionen om **performance vs. narrative** ved kåringen."
 )

 with tab_duel:
 st.subheader("Direkte sammenligning")
 m_rows = filtered[filtered["player"].str.contains("Messi", case=False, na=False)]
 mb_rows = filtered[filtered["player"].str.contains("Mbappé", case=False, na=False)]
 if m_rows.empty or mb_rows.empty:
 st.warning("Messi eller Mbappé findes ikke med de valgte filtre (tjek minimum minutter).")
 else:
 m = m_rows.iloc[0]
 mb = mb_rows.iloc[0]
 duel_metrics = [
 ("Offensiv score (V4)", score_col),
 ("Mål (total)", "goals"),
 ("Assists", "assists"),
 ("Minutter", "minutes"),
 ("Mål uden straffe pr. 90", "nonpen_goals_per90"),
 ("Assists pr. 90", "assists_per90"),
 ("xA pr. 90", "xg_assist_per90"),
 ("Cluster", cluster_col),
 ]
 dcols = st.columns(2)

 def _fmt_metric(title: str, col: str, row: pd.Series) -> None:
 v = row[col]
 if col == cluster_col:
 st.metric(title, str(int(v)))
 elif col in ("minutes", "goals", "assists"):
 st.metric(title, str(int(round(float(v)))))
 elif col == score_col:
 st.metric(title, f"{float(v):.2f}")
 else:
 st.metric(title, f"{float(v):.2f}")

 with dcols[0]:
 st.markdown(f"### {m['player']}")
 for title, col in duel_metrics:
 _fmt_metric(title, col, m)
 with dcols[1]:
 st.markdown(f"### {mb['player']}")
 for title, col in duel_metrics:
 _fmt_metric(title, col, mb)

 diff_rows = []
 for title, col in duel_metrics[:-1]:
 try:
 dv = float(mb[col])-float(m[col])
 diff_rows.append({"Nøgletal": title, "Mbappé − Messi": dv})
 except (TypeError, ValueError):
 continue
 if diff_rows:
 st.markdown("**Forskel (Mbappé − Messi)**")
 st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

 # Få akser radardiagrammer med alle V4-features (~40) bliver ulæselige; fuld dekomposition er under
 # «Score for én spiller». Her: et læsbart udvalg af centrale pr. 90 min nøgletal.
 radar_feats = [
 "nonpen_goals_per90",
 "assists_per90",
 "xg_assist_per90",
 "npxg_net_per90",
 "progressive_passes_received_per90",
 "shots_on_target_per90",
 ]
 st.caption(
 "Radaren viser **6 udvalgte** parametre for overskuelighed. Den samlede V4 score bruger **alle** features "
 "med deres vægte se dem under fanen *Score for én spiller* for hver af de to."
 )
 labels = [label_feature(f) for f in radar_feats]
 vals_m = [float(m[f]) for f in radar_feats]
 vals_mb = [float(mb[f]) for f in radar_feats]
 # Per akse: skaler til max(Messi, Mbappé) på den dimension ellers dominerer én stor værdi hele plottet.
 vals_m_n = []
 vals_mb_n = []
 for vm, vmb in zip(vals_m, vals_mb):
 mx = max(vm, vmb, 1e 9)
 vals_m_n.append(vm / mx)
 vals_mb_n.append(vmb / mx)

 fig_r = go.Figure()
 fig_r.add_trace(
 go.Scatterpolar(
 r=vals_m_n + [vals_m_n[0]],
 theta=labels + [labels[0]],
 fill="toself",
 name=m["player"],
 line_color="#3498db",
 )
 )
 fig_r.add_trace(
 go.Scatterpolar(
 r=vals_mb_n + [vals_mb_n[0]],
 theta=labels + [labels[0]],
 fill="toself",
 name=mb["player"],
 line_color="#e67e22",
 )
 )
 fig_r.update_layout(
 polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
 showlegend=True,
 height=520,
 title="Normaliseret radar (hver akse: max af Messi og Mbappé på den dimension)",
 )
 st.plotly_chart(fig_r, use_container_width=True)

 st.divider()
 st.markdown(
 """
 **Om appen:** Stage 4 følger eksamenskravet om en brugervenlig præsentation af analysen.
 Tallene beregnes med `vm2022_pipeline.py`, så de matcher logikken i `VM2022_messi_mbappe.ipynb` (V4).
 """
 )


if __name__ == "__main__":
 main()
