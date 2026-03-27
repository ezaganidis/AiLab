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
- `images/logo.svg` for app branding

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
