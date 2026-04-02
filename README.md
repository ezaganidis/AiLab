# AI λab

A production-style Streamlit multipage app.

## Pages structure

- `app.py` (entry/home, shared style bootstrap)
- `pages/0_🏠_Main_Page.py`
- `pages/1_🤖_Auto_ML.py`
- `pages/2_🧪_Custom_ML.py`
- `pages/3_🔮_Predict.py`

## Helpers

Reusable functions are under `helpers/`:
- `helpers/config.py`
- `helpers/state.py`
- `helpers/style.py`
- `helpers/data_utils.py`
- `helpers/ml_utils.py`

## Session memory

The app initializes and keeps workflow state in `st.session_state`, so users can switch pages without losing uploaded data, split context, selected features, trained summaries, or best-model references.

## Config

- Streamlit UI/runtime config: `.streamlit/config.toml`
- App-level metadata/paths: `app_config.toml`

## Artifacts

- `artifacts/saved_models` for model bundles
- `artifacts/saved_pipelines` for reusable feature pipelines
- `images/logo_app.svg` for app branding
- `images/logo_research_team.svg` for research team branding
- `images/logo_university.svg` for university branding

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```


## Logging and error handling

- Application logs are written to `artifacts/app.log`.
- Custom ML training shows progress stage-by-stage with status messages.
- If any model fails, the app shows the error, proposes a solution, and provides a **Rerun failed models only** button.
- Verbose mode is enabled via `app_config.toml` (`verbose = true`).


## Windows one-click run

- `environment.yml` creates/updates the conda environment `streamlit_app_duth`.
- `run_ml_lab.bat` bootstraps conda env, activates it, starts Streamlit, and opens the browser.

## Classification label-specific metrics

In **Custom ML → Results**, when task is classification, you can select a label (from available target labels) and inspect label-specific precision/recall/F1 for in-sample and out-of-sample predictions.


### If you see WinError 32 / invalid distribution `~treamlit`

The updated `run_ml_lab.bat` now:
- closes running `streamlit.exe` / `python.exe` processes,
- removes stale `streamlit.exe.deleteme` and `~treamlit*` leftovers,
- retries environment creation/update once automatically.

If it still fails, close all terminals/IDEs using the env and run:

```bat
conda remove -n streamlit_app_duth --all
```

Then run `run_ml_lab.bat` again.


## How to start correctly

Run from the project root (same folder that contains `app.py`, `helpers/`, and `pages/`):

```bash
cd /path/to/Manos
streamlit run app.py
```

`app.py` is the entrypoint and now includes direct page links to all modules when sidebar navigation is disabled.


## New modeling controls

- Classification probability outputs are shown when `predict_proba` is available.
- Decision-threshold tuning is supported with threshold curves (precision/recall/F1).
- Confusion matrix normalization is user-selectable (`none`, `true`, `pred`, `all`).
- Optional stacking ensemble can be enabled from the modeling tab.


- Training configuration summary is shown in both Auto ML and Custom ML before running, so users can inspect model characteristics and settings.


- Top navigation bar (Main / Auto ML / Custom ML / Predict) is rendered on every page and remains visible while navigating.
- Logs are stored in `logs/app.log`.

- All major widget selections persist in `st.session_state` (with stable keys) across page/tab navigation.

- Random state is fixed to **42** in training split for reproducibility (no user input).
- Model saving is now end-to-end via a selected trained model bundle (pipeline+selector+model together).

- Bootstrap mean/std now use the same selected optimization scorer as CV and out-of-sample, preserving scale consistency.

- Auto ML now includes Results and XAI tabs aligned with Custom ML output structure.

- Page routing now uses `st.navigation(..., position="top")` for a persistent top navigation bar.
