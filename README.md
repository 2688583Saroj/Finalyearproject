# Crop Yield Prediction using Random Forest Regression

This is a Streamlit application for crop-yield prediction using a trained Random Forest Regression model.

## Features

- India crop yield dataset support
- Random Forest Regression prediction model
- User registration and login
- Farm location selection from dataset locations
- Prediction history for registered users
- Admin dashboard for users, predictions, datasets and model evaluation
- R2, MAE and RMSE model metrics
- Dataset preview, yield distribution and correlation charts
- Model feature importance chart
- Safer password hashing with PBKDF2
- Admin credentials loaded from secrets or environment variables

## Folder Structure

```text
Random forest/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- secrets.example.toml
|-- data/
|   |-- crop_yield.csv
|   `-- app_records.json
|-- model/
|   |-- india_crop_yield_metrics.json
|   `-- india_crop_yield_model.pkl
`-- outputs/
```

## How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Admin Login Setup

Admin credentials are not stored directly in `app.py`.

For local development, create `.streamlit/secrets.toml` and use the format from `secrets.example.toml`:

```toml
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "change-this-password"
```

For Streamlit Community Cloud, add the same values in the app's Secrets settings.

## Dataset

### India Crop Yield Dataset

Features used:

- Crop
- Crop_Year
- Season
- State
- Area
- Production
- Annual_Rainfall
- Fertilizer
- Pesticide

Target:

- Yield

## Deployment Checklist

Make sure these files are included when deploying:

- `app.py`
- `train_model.py`
- `requirements.txt`
- `data/crop_yield.csv`
- `model/india_crop_yield_metrics.json`
- `model/india_crop_yield_model.pkl`

Do not publish real user records. `data/app_records.json` should be empty or contain only safe sample data before submission.

## Notes for Dissertation

You can describe the system as a machine-learning decision-support tool using Random Forest Regression. The evaluation dashboard provides R2, MAE, RMSE, data preview charts and feature importance values that can be included in the results chapter.
