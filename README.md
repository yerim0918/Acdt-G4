# Beyond the Ratings — ACDT Project 1

A polished, English-language **Option 4: Data Story Website** built with Streamlit. It combines interactive movie exploration, robust OLS, genre-interaction charts, Random Forest validation, RF-based DML results, a live regression calculator, and System Maps A–C.

## What is inside?

- `app.py` — the Streamlit web app and visual design.
- `analysis.py` — auditable dataset cleaning, HC3 OLS fits, and regression predictions.
- `data/movies_2022_2025_usd.csv` — cleaned USD-converted 407-film input; the app uses 385 after quality checks and target-genre assignment.
- `data/*results.csv` — regression, genre, robustness, ML and DML summaries; `RF_feature_importance.csv` is recomputed using the notebook’s training/holdout settings.
- `assets/map_a.png`, `assets/map_b.png`, `assets/map_c.png` — latest final system-map images supplied in this conversation.
- `analysis_notebook.ipynb` — the original executed analysis notebook.
- `Executive_Summary.pdf` and `System_Map_Justification.md` — downloadable supporting materials.
- `requirements.txt` / `.streamlit/config.toml` — deployment environment.

## Launch locally (Windows / macOS / Linux)

1. Extract the ZIP to a directory. Keep all folders together.
2. Open Terminal / PowerShell in the `ACDT_Beyond_The_Ratings` folder.
3. (Recommended) Create a Python virtual environment: `python -m venv .venv` and activate it.
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Run: `python -m streamlit run app.py`
6. Open the address Streamlit displays (typically http://localhost:8501).

### Deploy to Streamlit Community Cloud (GitHub)

1. Sign in to [GitHub](https://github.com) and create a new repository (e.g., `beyond-the-ratings`).
2. Upload **the contents of this folder** (including `app.py`, `analysis.py`, `requirements.txt`, `data/`, `assets/`, `.streamlit/`, and the notebook / summary). Do not upload just the ZIP without extracting it. Keep the same folder structure.
3. Open [Streamlit Community Cloud](https://share.streamlit.io) and sign in using GitHub.
4. Choose **Create app**, select your repository and branch (typically `main`), and set main file path to **`app.py`**.
5. Deploy, wait for dependencies to install, and open your new `*.streamlit.app` URL.
6. Check all page sections and calculator controls, open the 3 maps, and confirm download buttons. Record the 5–7 minute pitch required by the assignment instructions; share the URL to Slack/LXP and the video link as required.

**Important:** A public GitHub repository and public Streamlit app make included files downloadable or publicly accessible. This package contains academic datasets, not passwords. Review your class's publication and citation requirements before publishing.

## Assignment Option 4 requirements coverage

| Requirement | Implementation |
|---|---|
| Scrollytelling | Eight editorial chapters on one scrollable page, with sticky narrative navigation in the sidebar |
| Progressive disclosure | Evidence unfolds by chapter; year reveal slider, genre selectors, and expandable methods/robustness notes |
| Embedded statistical calculators | Live Metascore / budget / genre / year controls backed by fitted genre-interaction OLS, with confidence interval |
| Downloadable resources | CSV, executed notebook, maps, executive-summary PDF, and justification document |

## Data and methodological cautions

- Based on the user-supplied Kaggle-derived movie dataset and the group’s processed USD dataset (not an independently audited record of every worldwide release).
- Primary genre is assigned as the **first matching** target genre in the source genre list.
- Excludes missing reported budgets/gross/metascore and movies outside the four target genres. Selection bias is possible.
- 2025 is incomplete. Budget currency conversions approximate release-year values, not exact spending-date exchange rates.
- Ratings and revenue are observational, potentially influenced by common causes and timing.
- RF model interpretation is predictive; DML identifies a causal effect only under unverified assumptions. Accordingly the website describes DML as an **adjusted association**.
- Statistical report values match the supplied notebook and user-provided results.

## Test and regenerate

- `python -m pytest tests -q` — checks 385-film sample, regressions, genre assignments, calculator outputs and bundled assets.
- `python build_summary_pdf.py` — regenerate one-page PDF using ReportLab (not needed for hosting).

## Pitch suggestion (6 minutes)

0:00–0:35 hook → 0:35–1:25 sample/explorer → 1:25–2:30 simple vs adjusted regression → 2:30–3:15 genre interaction → 3:15–4:05 Random Forest / DML → 4:05–4:50 System Maps A–C → 4:50–5:35 interactive calculator → 5:35–6:00 limits and takeaway. Recording and subtitles are still required by the brief.
