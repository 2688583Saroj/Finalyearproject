# Crop Yield Prediction using Random Forest Regression

This is an improved VS Code-ready project for your dissertation.

## Features

- Supports both uploaded CSV datasets
- Random Forest Regression model
- Automatic handling of numeric and categorical columns
- Streamlit UI
- Prediction page
- Evaluation dashboard
- R², MAE and RMSE metrics
- Dataset preview and yield charts

## Folder Structure

```text
crop_yield_project_improved/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── crop_yield.csv
│   └── crop-yield.csv
├── model/
└── outputs/
```

## How to Run in VS Code

Open the folder in VS Code, then run:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset Modes

### 1. India Crop Yield Dataset
Uses:
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

### 2. Soil + Weather Dataset
Uses:
- N, P, K
- Soil pH and moisture
- Temperature
- Humidity
- Rainfall
- Region
- Season
- Crop type
- Irrigation type
- Fertilizer used
- Pesticide used

Target:
- Crop_Yield_ton_per_hectare

## Notes for Dissertation

You can describe the system as a machine-learning decision-support tool using Random Forest Regression. 
The evaluation dashboard provides R², MAE and RMSE values, which can be included in your results chapter.
