"""Beyond the Ratings | interactive ACDT Project 1 data story.

Run: streamlit run app.py
Analysis derives from the supplied ACDT USD-converted dataset and notebook.
"""
from pathlib import Path
import io
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

from analysis import (
    DATA, GENRES, ROOT, fit_regressions, load_raw, per_ten_points,
    prediction, prediction_curve, prepare_data,
)

ASSETS = ROOT / "assets"
FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
NAVY = "#142C43"
BLUE = "#2E617E"
TEAL = "#147A80"
PALE = "#E5F2F0"
SLATE = "#6D7C8E"
BORDER = "#DBE5EE"
GENRE_COLORS = {"Action": "#116D75", "Comedy": "#7391A7", "Drama": "#283F60", "Horror": "#A0B1C0"}

st.set_page_config(
    page_title="Beyond the Ratings | ACDT Data Story",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --navy:#142C43; --slate:#65778A; --teal:#147A80; --border:#DBE5EE; }
html { scroll-behavior:smooth; }
[data-testid="stAppViewContainer"] { background:linear-gradient(180deg,#FAFCFF 0%,#F6F9FC 52%,#FCFDFE 100%); }
[data-testid="stHeader"] { background:rgba(250,252,255,.76); }
[data-testid="stSidebar"] { background:#EEF3F8; border-right:1px solid #DCE6F0; }
.block-container {max-width:1230px; padding-top:1.25rem; padding-bottom:5.5rem;}
h1,h2,h3,h4,h5 { color:#142C43; letter-spacing:-.035em; }
p,li {line-height:1.65;}
[data-testid="stMarkdownContainer"] p {color:#40566B;}
.hero {position:relative;padding:3.0rem 3.35rem 2.8rem;border-radius:22px;background:linear-gradient(123deg,#102941 0%,#1B425A 72%,#176D74 142%);color:#fff;overflow:hidden;box-shadow:0 13px 35px rgba(18,42,60,.09);}
.hero::after {content:"";position:absolute;right:-7%;top:-70px;width:370px;height:370px;border:1px solid rgba(255,255,255,.13);border-radius:50%;box-shadow:0 0 0 80px rgba(255,255,255,.025),0 0 0 155px rgba(255,255,255,.015);}
.hero .eyebrow {font-size:12px;letter-spacing:2.7px;text-transform:uppercase;font-weight:700;color:#93D6D4;}
.hero h1 {font-size:clamp(2.2rem,5vw,4.0rem);line-height:1.06;color:white;max-width:850px;margin:1.15rem 0 1rem;}
.hero .deck {font-size:1.1rem;color:#D7E5EB;line-height:1.8;max-width:740px;}
.hero .tags {margin-top:1.9rem;display:flex;gap:.55rem;flex-wrap:wrap;}
.hero .tag {padding:.42rem .78rem;border-radius:32px;color:#EAF7F6;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.17);font-size:12px;font-weight:600;}
.sectionHead {padding-top:2.2rem;padding-bottom:.45rem;}
.sectionHead .eyebrow{font-size:.8rem;font-weight:800;letter-spacing:2px;color:#147A80;text-transform:uppercase;}
.sectionHead h2 {font-size:clamp(1.65rem,3vw,2.45rem);line-height:1.2;margin:.45rem 0 .55rem;}
.sectionHead p {max-width:840px;color:#65788B;line-height:1.7;font-size:1rem;}
.chapterrule {height:1px;background:linear-gradient(90deg,#d2dfe8,transparent);margin:1.8rem 0 .15rem;}
.numberstrip {background:#EAF3F5;border:1px solid #D5E8EB;border-radius:18px;padding:1.2rem 1.45rem;}
.factlabel {text-transform:uppercase;letter-spacing:1.2px;color:#688092;font-size:11px;font-weight:750;}
.factvalue {color:#152F47;font-size:2.05rem;line-height:1.3;font-weight:780;letter-spacing:-.04em;}
.factnote {color:#738699;font-size:12px;line-height:1.5;}
.editorial {padding:1.35rem 1.6rem;border:1px solid #DCE7EF;border-radius:16px;background:white;box-shadow:0 5px 18px rgba(29,53,74,.025);margin:.7rem 0;}
.editorial h3 {margin:0 0 .45rem;font-size:1.15rem;}
.editorial p {margin:0;}
.callout {border-left:4px solid #147A80;padding:1.15rem 1.5rem;border-radius:0 12px 12px 0;background:#EEF6F6;margin:1.2rem 0;}
.callout strong {color:#12374A;}
.mini-card {background:#fff;border:1px solid #D9E5ED;border-radius:14px;padding:1.25rem;margin:.35rem 0;min-height:137px;}
.mini-card .metric {font-size:2.3rem;line-height:1.17;letter-spacing:-.045em;font-weight:800;color:#147A80;margin:.25rem 0;}
.mini-card .metric.muted {color:#73879A;}
.mini-card .title {font-weight:750;color:#1F3952;font-size:13px;}
.mini-card .sub {font-size:12px;color:#748596;line-height:1.45;}
.mapheading {font-size:1.25rem;font-weight:800;color:#14334B;margin:0 0 .45rem;}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"] {background:#147A80;border-color:#147A80;}
[data-testid="stPlotlyChart"] {border-radius:15px;overflow:hidden;}
footer {visibility:hidden;}
@media(max-width:750px){.hero{padding:2.2rem 1.5rem}.hero h1{font-size:2.4rem}.block-container{padding:1rem 1rem 4rem}}
</style>
""", unsafe_allow_html=True)


def chapter(num: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="chapterrule"></div><div class="sectionHead" id="chapter-{num}"><div class="eyebrow">Chapter {num}</div><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def chart_style(fig, height: int = 410):
    fig.update_layout(
        height=height, template="plotly_white", font=dict(family=FONT, color=NAVY, size=13),
        title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=15, t=25, b=24), legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#112A42", font_color="#FFFFFF"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ECF0F5", zeroline=False, linecolor="#D9E5ED")
    fig.update_yaxes(showgrid=True, gridcolor="#ECF0F5", zeroline=False, linecolor="#D9E5ED")
    return fig


def card(label, value, note="", muted=False):
    m = " muted" if muted else ""
    st.markdown(f'<div class="mini-card"><div class="title">{label}</div><div class="metric{m}">{value}</div><div class="sub">{note}</div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    raw = load_raw()
    df, validated = prepare_data(raw)
    return raw, df, validated


@st.cache_resource(show_spinner=False)
def get_models():
    _, data, _ = get_data()
    return fit_regressions(data)


@st.cache_resource(show_spinner=False)
def get_budget_only_model():
    """Intermediate HC3-robust regression; same 385-film sample as the main models."""
    _, data, _ = get_data()
    return smf.ols("log_revenue ~ metascore + log_budget", data=data).fit(cov_type="HC3")


@st.cache_data(show_spinner=False)
def load_table(name):
    return pd.read_csv(DATA / name)


raw, df, validated = get_data()
simple, multiple, interaction = get_models()
budget_only = get_budget_only_model()
ols_table = load_table("ACDT_OLS_comparison.csv")
interaction_table = load_table("ACDT_genre_interaction.csv")
robustness = load_table("ACDT_genre_robustness_checks.csv")
prediction_cv = load_table("ACDT_prediction_5fold_cv.csv")
prediction_test = load_table("ACDT_prediction_comparison.csv")
dml = load_table("ACDT_DML_results.csv").iloc[0]
rf_importance = load_table("RF_feature_importance.csv")

with st.sidebar:
    st.markdown("### 🎞️ BEYOND THE RATINGS")
    st.caption("ACDT Project 1 · Group research")
    st.markdown("**Reading guide**")
    st.markdown("""[01 · The myth](#chapter-01)  
[02 · The data](#chapter-02)  
[03 · The reversal](#chapter-03)  
[04 · The genre question](#chapter-04)  
[05 · Beyond linear models](#chapter-05)  
[06 · System maps](#chapter-06)  
[07 · Try it yourself](#chapter-07)  
[08 · Takeaways & sources](#chapter-08)""")
    st.divider()
    st.metric("Study sample", f"{len(df)} films")
    st.caption("2022–2025 · Action / Comedy / Drama / Horror")
    st.caption("Metascore: critic ratings (0–100). Outcome: worldwide gross (USD).")
    st.info("**Interpretation:** This is an observational study. Coefficients are adjusted associations, not proven causal effects.", icon="ℹ️")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Interactive data story · ACDT project 1</div>
  <h1>Beyond the Ratings.</h1>
  <div class="deck">Do better-reviewed films earn more?—From a familiar assumption to the hidden systems shaping global box-office success.</div>
  <div class="tags"><span class="tag">2022–2025 releases</span><span class="tag">385 films analyzed</span><span class="tag">4 genres</span><span class="tag">Regression + Random Forest + DML</span></div>
</div>
""", unsafe_allow_html=True)
st.caption("A research narrative based on the supplied IMDb-derived Kaggle dataset, subsequent cleaning and USD conversion, and the group’s Python analysis notebook.")

chapter("01", "The myth: a better review means a bigger hit", "It sounds intuitive: acclaimed movies should attract audiences, generating higher box-office revenue. But is this visible in the data—and is it consistent across genres?")
c1,c2 = st.columns([3,2],gap="large")
with c1:
    st.markdown("""<div class="editorial"><h3>Our research question</h3><p>How is <strong>Metacritic Metascore</strong> associated with <strong>worldwide box-office revenue</strong> after accounting for production budget, release year, and genre? Does the association change by genre?</p></div>""",unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="editorial"><h3>What would change our minds?</h3><p>First compare the raw Metascore relationship with revenue, then adjust for other measured factors, then test whether a more flexible model changes the conclusion.</p></div>""",unsafe_allow_html=True)
st.markdown('<div class="callout"><strong>Hypothesis.</strong> Higher critic scores tend to be associated with higher worldwide revenue; genre may moderate that relationship. This is a test of association, not proof that reviews cause ticket sales.</div>',unsafe_allow_html=True)

chapter("02", "Meet the evidence", "Our data was cleaned before modeling. Hover over individual films, filter genres, and reveal how the available release years change the picture.")
col1,col2,col3,col4 = st.columns(4,gap="small")
with col1: card("USD-CONVERTED INPUT",f"{len(raw):,}","2022–2025 candidate films")
with col2: card("VALIDATED RECORDS",f"{len(validated):,}","After log-compatible checks")
with col3: card("FINAL SAMPLE",f"{len(df):,}","Movies assigned to 4 study genres")
with col4: card("GENRE GROUPS","4","First matching study genre in listed genres")

left,right = st.columns([2,1],gap="large")
with left:
    movie_genres = st.multiselect("Select genres to explore", options=list(GENRES), default=list(GENRES),key="explorer_genres")
with right:
    through_year = st.select_slider("Reveal releases through",options=[2022,2023,2024,2025],value=2025,key="explorer_year")
subset = df.loc[df.primary_genre.isin(movie_genres) & (df.release_year <= through_year)].copy()
if len(subset):
    subset["gross_m"] = subset.gross_worldwide / 1e6
    fig = px.scatter(subset,x="metascore",y="gross_m",color="primary_genre",color_discrete_map=GENRE_COLORS,
        custom_data=["title","release_year","budget","rating"],opacity=.78,
        labels={"metascore":"Critic Metascore (0–100)","gross_m":"Worldwide box office ($M, log scale)","primary_genre":"Genre"})
    fig.update_traces(marker=dict(size=8, line=dict(color="white",width=.55)),
        hovertemplate="<b>%{customdata[0]}</b><br>Metascore: %{x:.0f}<br>Worldwide gross: $%{y:,.2f}M<br>Year: %{customdata[1]}<br>Budget (USD): $%{customdata[2]:,.0f}<br>IMDb rating: %{customdata[3]:.1f}<extra></extra>")
    fig.update_yaxes(type="log")
    chart_style(fig,450)
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.caption(f"Showing {len(subset):,} individual films released through {through_year}; worldwide gross is on a logarithmic display scale. Points represent movies, not model predictions.")
else:
    st.warning("Select at least one genre to display the data.")
st.markdown(
    """<div class="editorial"><h3>Reading the logarithmic axis</h3>
    <p>Movie grosses differ dramatically. On this graph, equal vertical distances mean equal
    <strong>proportional changes</strong> (such as $1M to $10M and $10M to $100M),
    not equal dollar increases. This makes smaller and larger movies easier to compare.</p></div>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="editorial"><h3>Data Limitation — Potential Selection Bias</h3>
    <p>Movies with missing budget or revenue information were excluded from our analysis.
    Because financial data may be less available for smaller or independent films,
    our final sample may overrepresent larger, better-documented productions.
    Therefore, the findings should not be generalized to all movies.</p></div>""",
    unsafe_allow_html=True,
)
with st.expander("How the 385-film sample was built"):

    st.markdown("""The original USD-converted file contains **407** movies. The notebook requires valid Metascore, votes ≥ 100, worldwide gross > 0, and budget ≥ $1,000. One film is excluded for an implausibly low reported budget, and 21 of the remaining 406 do not match one of our four target genres. Each film is assigned the **first matching** target genre from its original genre list. This selection and missing-budget exclusions may bias the sample. The source dataset is not a random sample of every film worldwide. The 2025 period is incomplete.""")
    st.dataframe(df["primary_genre"].value_counts().rename_axis("Genre").reset_index(name="Movies"),hide_index=True,use_container_width=True)

chapter("03", "The reversal: adjustment changes the story", "A raw one-variable trend is not the same as a comparison of films with similar production budgets, years, and genres.")
c1,c2 = st.columns(2,gap="medium")
with c1:
    card("SIMPLE REGRESSION", f"{per_ten_points(simple.params['metascore']):+.1f}%",f"Per +10 Metascore points · p = {simple.pvalues['metascore']:.3f}",muted=True)
with c2:
    card("MULTIPLE REGRESSION", f"{per_ten_points(multiple.params['metascore']):+.1f}%",f"Per +10 points · p = {multiple.pvalues['metascore']:.4f}")
st.markdown('<div class="callout"><strong>The twist:</strong> Metascore was not significant by itself (p ≈ 0.205). After controlling for budget, release year and genre, it showed a positive adjusted association (p ≈ 0.0008). This difference is not itself evidence of causation.</div>',unsafe_allow_html=True)
st.markdown("**What changed as we added controls?**")
explain_models=pd.DataFrame([
    {"Comparison":"Metascore only", "Association per +10 points":f"{per_ten_points(simple.params['metascore']):+.1f}%", "p-value":f"{simple.pvalues['metascore']:.4f}","Controls":"None"},
    {"Comparison":"Metascore + budget", "Association per +10 points":f"{per_ten_points(budget_only.params['metascore']):+.1f}%", "p-value":f"{budget_only.pvalues['metascore']:.4f}","Controls":"ln(budget)"},
    {"Comparison":"Metascore + budget + year + genre", "Association per +10 points":f"{per_ten_points(multiple.params['metascore']):+.1f}%", "p-value":f"{multiple.pvalues['metascore']:.4f}","Controls":"ln(budget), release year, primary genre"},
])
st.dataframe(explain_models,hide_index=True,use_container_width=True)
st.markdown(
    """<div class="editorial"><h3>Why can the sign reverse?</h3><p>
    Production budget is strongly related to box-office revenue. In this sample, critic
    scores and budgets are modestly negatively correlated, so the unadjusted comparison
    mixes differences in ratings with differences in film budgets.
    <strong>After controlling for budget alone, the association becomes positive (+11.8%),
    although still not statistically significant.</strong> Adding year and genre makes the
    positive adjusted association clearer (+22.9%). This is consistent with confounding
    or masking by observed film characteristics, <strong>not proof that changing reviews
    would change ticket sales.</strong></p></div>""",
    unsafe_allow_html=True,
)


results=[]
for label,fit in [("Simple",simple),("Adjusted",multiple)]:
    beta=fit.params["metascore"]
    lo,hi=fit.conf_int().loc["metascore"]
    results.append({"Model":label,"Effect":per_ten_points(beta),"Low":per_ten_points(lo),"High":per_ten_points(hi)})
ci_df=pd.DataFrame(results)
fig=go.Figure()
fig.add_vline(x=0,line_dash="dash",line_color="#95A8B7",line_width=1.3)
for _,row in ci_df.iterrows():
    color=TEAL if row.Model=="Adjusted" else "#94A8B8"
    fig.add_trace(go.Scatter(x=[row.Low,row.High],y=[row.Model,row.Model],mode="lines",line=dict(color=color,width=7),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[row.Effect],y=[row.Model],mode="markers+text",text=[f"{row.Effect:+.1f}%"],textposition="top center",marker=dict(size=15,color=color),showlegend=False,
        hovertemplate=f"{row.Model}: {row.Effect:+.1f}%<br>95% CI: [{row.Low:+.1f}%, {row.High:+.1f}%]<extra></extra>"))
fig.update_xaxes(title="Modeled association per +10 Metascore points (% gross difference)")
fig.update_yaxes(title="",categoryorder="array",categoryarray=["Adjusted","Simple"])
chart_style(fig,315)
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
st.caption(f"95% HC3 robust confidence intervals. Simple R² = {simple.rsquared:.3f}; multiple regression R² = {multiple.rsquared:.3f}. Original model: ln(revenue) ~ Metascore + ln(budget) + release-year and primary-genre indicators.")
st.markdown(
    "**In everyday terms:** A +22.9% association means movies with scores 10 points "
    "higher are estimated to have about 22.9% higher worldwide gross *when the model holds "
    "budget, genre, and year constant*. It does **not** mean $22.9 million more, nor does "
    "it show that raising a score would cause revenue to increase."
)

with st.expander("Show the regression formula and statistical interpretation"):
    st.latex(r"\ln(Revenue_i)=\beta_0+\beta_1 Metascore_i+\beta_2\ln(Budget_i)+\gamma_{Year(i)}+\delta_{Genre(i)}+\varepsilon_i")
    st.markdown("Each +10 score-point association is converted using `100 × (exp(10 × β₁) − 1)`. A coefficient does not prove that changing critics’ reviews would cause the same revenue change. HC3 robust standard errors allow more reliable inference under heteroskedasticity, but do not fix omitted-variable bias.")

chapter("04", "Does genre change the answer?", "The interaction model tests whether Metascore’s association with revenue differs across Action, Comedy, Drama, and Horror.")
interaction_p=0.049296 # matches notebook's joint HC3 Wald test
st.markdown(f'<div class="editorial"><h3>Genre interaction test</h3><p>Joint Wald test: <strong>p = {interaction_p:.4f}</strong>. The main model suggests some differences by genre, but robustness checks show this result is sensitive to how genre is coded and which films are included.</p></div>',unsafe_allow_html=True)

rows=interaction_table.copy()
rows["Effect (per +10 points)"] = rows["10-point %"].map(lambda v:f"{v:+.1f}%")
rows["p-value"] = rows["p"].map(lambda v:f"{v:.4f}")
st.dataframe(rows[["Genre","N","Effect (per +10 points)","p-value","10-point % 95% CI"]],hide_index=True,use_container_width=True)
choose_genre=st.selectbox("Visualize the adjusted association for a genre",list(GENRES),index=0,key="genre_plot")
median_budget_m=float(df.budget.median()/1e6)
curve=prediction_curve(interaction,choose_genre,median_budget_m,2024)
fig=go.Figure()
fig.add_trace(go.Scatter(x=curve.metascore,y=curve.hi/1e6,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
fig.add_trace(go.Scatter(x=curve.metascore,y=curve.lo/1e6,fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(20,122,128,.15)",name="95% CI for mean log revenue",hoverinfo="skip"))
fig.add_trace(go.Scatter(x=curve.metascore,y=curve.usd/1e6,mode="lines",line=dict(color=TEAL,width=3.5),name="Adjusted model trend",hovertemplate="Metascore %{x:.0f}<br>Conditional typical revenue $%{y:,.1f}M<extra></extra>"))
fig.update_xaxes(title="Metascore (0–100)")
fig.update_yaxes(title="Exponentiated fitted log revenue ($M)")
chart_style(fig,365)
st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
st.caption(f"For illustration: {choose_genre}, budget fixed at sample median (${median_budget_m:,.1f}M) and release year 2024. Shaded ribbon is the 95% confidence interval for the conditional mean on the log scale (exponentiated)—not an individual film prediction interval.")
st.caption("Plain-English guide: the line is the model's estimated typical revenue for a specified rating, budget, genre, and year. The ribbon shows uncertainty about the fitted typical value, not the likely range of receipts for an individual film.")
with st.expander("Check whether the genre finding is robust"):
    display_robust=robustness.rename(columns={"Metascore 10-point assoc. (%)":"Overall +10 score (%)","Genre interaction joint p":"Genre interaction p"}).copy()
    st.dataframe(display_robust,hide_index=True,use_container_width=True)
    st.caption("The interaction is near the 0.05 threshold in the main model and changes substantially in the single-genre-only subsample. Avoid a definitive claim that the genre differences have been established.")

chapter("05", "Beyond the straight line", "We compared linear prediction with Random Forest, then estimated Metascore’s adjusted association using cross-fitted Double Machine Learning (DML).")
col1,col2=st.columns([1,1],gap="medium")
with col1: card("5-FOLD CV · LINEAR R²", "0.547","Linear Regression on log gross")
with col2: card("5-FOLD CV · RF R²", "0.585","Random Forest on log gross")

importance = rf_importance.sort_values("Importance",ascending=True)
fig=go.Figure(go.Bar(x=importance["Importance"],y=importance["Feature"],orientation="h",marker_color=[TEAL if v>0.01 else "#A8B7C5" for v in importance["Importance"]],
    customdata=importance["SD"],hovertemplate="%{y}<br>Change in log-MAE: %{x:+.3f}<br>SD: %{customdata:.3f}<extra></extra>"))
fig.add_vline(x=0,line_width=1,line_color="#74879A")
fig.update_xaxes(title="Permutation importance · increase in log-scale MAE")
fig.update_yaxes(title="")
chart_style(fig,300)
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
st.caption("Held-out-set Random Forest permutation importance: budget dominates; genre contributes modestly. Small or negative estimates for Metascore / release year mean no reliable improvement in this holdout under this importance method—not proof that their relationships are zero.")

st.markdown(f'<div class="callout"><strong>RF-based DML result:</strong> +10 Metascore points correspond to approximately <strong>{float(dml["10-point %"]):+.1f}%</strong> higher worldwide gross after flexible adjustment (approximate p = {float(dml["p (approx)"]):.5f}). DML uses Random Forest for nuisance adjustment; it is distinct from the Random Forest revenue prediction model.</div>',unsafe_allow_html=True)
with st.expander("DML methodology and limits"):
    st.markdown("5-fold cross-fitted partialling-out DML uses budget (log USD), release year, and primary genre as measured adjustment variables, and estimates a linear Metascore coefficient on log revenue. The approximate 95% coefficient interval is "
        f"[{float(dml['CI low']):.4f}, {float(dml['CI high']):.4f}] log points per score point. "
        "The observational design cannot confirm the causal identification assumptions; unobserved marketing, franchise appeal, selection effects, or rating/revenue timing may confound this relationship.")
    st.dataframe(prediction_cv,hide_index=True,use_container_width=True)
    st.caption("CV mean absolute error and R² are reported on log revenue; note that holdout R² and CV R² are different evaluations.")

chapter("06", "Three ways to see one system", "The maps intentionally answer different questions: what our regression supports, what the broader movie industry might contain, and what ML learns for prediction and adjustment.")
map_tabs=st.tabs(["Map A · Statistical","Map B · Theoretical","Map C · ML-based"])
map_specs=[
    ("map_a.png","A. Regression-supported relationships","This map summarizes sample associations, including a positive adjusted Metascore link and a borderline genre interaction. Arrows are statistical associations, not demonstrated mechanisms."),
    ("map_b.png","B. Plausible hidden mechanisms","This theoretical scenario map includes unobserved marketing, distribution, franchise and other potential mechanisms. Orange/dashed arrows are hypotheses—not measured effects or tested significance."),
    ("map_c.png","C. Predictive ML and adjusted DML","The Random Forest predictive view and DML adjusted association come from complementary but distinct analyses. ML importance is predictive, not causal. DML alone does not establish causality."),
]
for tab,(image_file,title,desc) in zip(map_tabs,map_specs):
    with tab:
        st.markdown(f"**{title}**")
        st.image(str(ASSETS/image_file),use_container_width=True)
        st.caption(desc)
        st.download_button(f"Download {title[0]} diagram (PNG)",data=(ASSETS/image_file).read_bytes(),file_name=f"ACDT_system_map_{title[0]}.png",mime="image/png",key=f"dl-map-{image_file}")
st.markdown('<div class="editorial"><h3>Why compare the maps?</h3><p><b>Map A</b> emphasizes regression-supported associations. <b>Map B</b> broadens the narrative to plausible but unmeasured causes. <b>Map C</b> contrasts prediction with adjustment. Ignoring hidden variables could lead studios to over-attribute success to reviews and under-appreciate marketing, distribution, audience reach, and timing.</p></div>',unsafe_allow_html=True)

chapter("07", "Try it yourself", "Compare hypothetical movie profiles while holding budget, genre, and year fixed. These are modeled associations—not interventions or guaranteed forecasts.")
l,r=st.columns([1,1],gap="large")
with l:
    calc_genre=st.selectbox("Primary genre",list(GENRES),index=0,key="calc_genre")
    calc_score=st.slider("Critic Metascore",min_value=0,max_value=100,value=70,step=1,key="calc_score")
with r:
    calc_budget=st.slider("Production budget ($ million)",min_value=1,max_value=250,value=50,step=1,key="calc_budget")
    calc_year=st.selectbox("Release year",[2022,2023,2024,2025],index=2,key="calc_year")
compare_score=min(100,calc_score+10)
score_diff=compare_score-calc_score
est=prediction(interaction,calc_score,calc_budget,calc_genre,calc_year)
est_plus=prediction(interaction,compare_score,calc_budget,calc_genre,calc_year)
main,compare=st.columns([1.1,.9],gap="medium")
with main:
    st.metric("Model-estimated typical worldwide gross",f"${est['usd']/1e6:,.1f}M")
    st.caption(f"95% confidence interval for fitted typical gross: ${est['lo']/1e6:,.1f}M – ${est['hi']/1e6:,.1f}M")
with compare:
    st.metric(
        f"Modeled score comparison: {calc_score} vs {compare_score}",
        f"${est_plus['usd']/1e6:,.1f}M",
        delta=f"{(est_plus['usd']/est['usd']-1)*100:+.1f}%" if score_diff>0 else None,
    )
    st.caption(f"95% confidence interval at score {compare_score}: ${est_plus['lo']/1e6:,.1f}M – ${est_plus['hi']/1e6:,.1f}M")
    if score_diff<10:
        st.caption(f"The rating scale ends at 100, so this comparison changes the score by only {score_diff} points—not 10.")
st.info(
    "**What this calculator means:** It compares hypothetical movie profiles with different "
    "Metascores, keeping production budget, release year, and genre fixed. The results are "
    "statistical associations, **not** the causal impact of changing a critic rating or "
    "guaranteed box-office outcomes. The displayed intervals describe uncertainty about "
    "the model-estimated typical value; they are **not** prediction intervals for individual films.",
    icon="ℹ️",
)
st.caption(
    "Technical note: the estimates equal exp(fitted mean log revenue). They should not be "
    "read as arithmetic mean dollar revenue or as reliable forecasts for unusual combinations "
    "of inputs. The percentage comparison is a model point estimate; a confidence interval "
    "for the difference is not displayed."
)

chapter("08", "The takeaway is more than a rating", "Data-informed decision-making means learning from what the data can show while keeping visible the factors it cannot.")
a,b=st.columns([1,1],gap="medium")
with a:
    st.markdown("""<div class="editorial"><h3>What our sample supports</h3><p>Metascore is positively associated with worldwide gross <strong>after adjustment</strong>, while the unadjusted trend is not significant. Budget is strongly associated with revenue. Genre variation appears plausible, but the moderation finding is fragile across reasonable coding choices.</p></div>""",unsafe_allow_html=True)
with b:
    st.markdown("""<div class="editorial"><h3>What remains uncertain</h3><p>Unmeasured marketing, franchise strength, screen reach and release strategy may shape these patterns. Films lacking disclosed budgets are excluded; 2025 is partial; critics’ scores may also reflect timing and success. These models do not identify a causal effect of scores.</p></div>""",unsafe_allow_html=True)

st.markdown("### Practical recommendations · who can use this?")
recommendations = [
    ("Film producers", "Benchmark films within comparable budget and genre groups before relying on critic scores. Check whether the intended project resembles the movies with complete financial data in our sample."),
    ("Distributors", "Before planning theatrical reach, collect information on promotion, number of screens, franchise recognition, and competition around the release date. These were not measured in our models."),
    ("Analysts & decision-makers", "Check missing-budget patterns, compare model errors across film groups, and report uncertainty. Do not treat a higher fitted revenue at a different Metascore as the financial return to improving reviews."),
]
rec_cols=st.columns(3,gap="medium")
for col,(audience,advice) in zip(rec_cols,recommendations):
    with col:
        st.markdown(f'<div class="mini-card"><div class="title">{audience}</div><p>{advice}</p></div>',unsafe_allow_html=True)
st.markdown("### Supporting Materials · download & verify")
st.caption("Resources are bundled in this project repository. Files are downloadable individually here.")
resources=[
    ("Executive summary · PDF",ROOT/"Executive_Summary.pdf","application/pdf"),
    ("Modeling dataset · CSV",DATA/"movies_2022_2025_usd.csv","text/csv"),
    ("Analysis notebook · ipynb",ROOT/"analysis_notebook.ipynb","application/json"),
    ("Regression comparison · CSV",DATA/"ACDT_OLS_comparison.csv","text/csv"),
    ("Genre interaction · CSV",DATA/"ACDT_genre_interaction.csv","text/csv"),
    ("DML results · CSV",DATA/"ACDT_DML_results.csv","text/csv"),
    ("Research justification · Markdown",ROOT/"System_Map_Justification.md","text/markdown"),
]
for idx,(label,path,mime) in enumerate(resources):
    if path.exists():
        st.download_button(label,data=path.read_bytes(),file_name=path.name,mime=mime,key=f"dl-res-{idx}")
with st.expander("Source, transparency, and project methodology"):
    st.markdown("""**Dataset provenance:** [Raed Addala — IMDb Movies from 1960 to 2023 (Kaggle)](https://www.kaggle.com/datasets/raedaddala/imdb-movies-from-1960-to-2023) as supplied and subsequently cleaned/converted to USD for this class project. The provided file includes entries dated 2022–2025; the dataset label is not a guarantee of full 2025 coverage. Budget conversion uses approximate release-year rates and need not equal historical production spending exchange rates.

**Analysis:** `analysis_notebook.ipynb` prepared by Group 4, matching the 385-film sample. Multiple regression uses `ln(revenue) ~ metascore + ln(budget) + C(release_year) + C(primary_genre)` and HC3 standard errors. Interaction adds `Metascore × Genre`. A Random Forest evaluates predictive value; 5-fold cross-fitting is used for DML (see notebook). Maps B and C include theoretical or modeling interpretations that should not be read as experimental evidence.

**Critical distinctions:** R² quantifies explained variation, not causality. Statistical significance is sensitive to sample selection, specification, and omitted variables. Film score/revenue timing can create reverse-causality or post-release selection issues. More complete budget, marketing and release information could alter these findings.""")
st.markdown('<div style="color:#8293A3;text-align:center;padding:3rem 0 0;font-size:12px;letter-spacing:.1em">BEYOND THE RATINGS · ACDT PROJECT 1 · DATA, INTERPRETATION & UNCERTAINTY</div>',unsafe_allow_html=True)