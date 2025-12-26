# Bulldozer Sale Price Predictor (Streamlit)

A lightweight Streamlit application that predicts bulldozer auction sale prices using a pre-trained RandomForest model. This repository contains the app (`app.py`), a trained model (`ideal_model.joblib`), and the training feature list (`X_train_columns.joblib`). The original data preprocessing and model training steps are documented in the included Jupyter notebook `Bulldozer_Auction_Price_Prediction.ipynb`.

---

## Contents
- `app.py` — Streamlit application (UI + preprocessing + prediction)
- `ideal_model.joblib` — serialized RandomForest model used for inference
- `X_train_columns.joblib` — list of training feature column names used to align inputs
- `Bulldozer_Auction_Price_Prediction.ipynb` — notebook with data exploration, preprocessing and model training steps
- `requirements.txt` — minimal Python dependencies

---

## Project overview
This app allows a user to supply a small set of machine and sale details (year, meter hours, categorical attributes, sale date) through a friendly UI. The app then:
1. Maps the user's inputs to the model's expected feature schema.
2. Applies the same preprocessing used during training (date-derived features, missing-value indicators, categorical encoding).
3. Aligns and orders columns to match the training set (`X_train_columns.joblib`).
4. Uses the loaded `ideal_model.joblib` RandomForest to predict the expected sale price.

The UI intentionally displays friendly labels (for human clarity) while keeping the original model feature names behind the scenes so predictions remain consistent with training.

---

## Quick setup (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app (from project root):

```powershell
streamlit run "d:/Projects/Bulldozer Price Prediction/app.py"
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`) in your browser.

---

## UI guide and input mappings
The app shows friendly labels to help users know what to enter. Below are the current friendly labels (what you see in the UI) and the model feature names they map to.

- Manufacture Year (model feature: `YearMade`)
  - What it is: 4-digit year from the machine plate or documentation.
  - Example: `2005`

- Meter Hours (model feature: `MachineHoursCurrentMeter`)
  - What it is: Hour meter reading; if unknown, estimate carefully or leave `0`.
  - Example: `1234`

- Model ID (model feature: `ModelID`)
  - What it is: The numeric model identifier (if available).
  - Example: `12345`

- Product Size (model feature: `ProductSize`)
  - What it is: General size class (e.g., `Mini`, `Compact`, `Medium`, `Large`). Choose `Unknown/Other` if unsure.
  - Example: `Compact`

- Operator Enclosure (model feature: `Enclosure`)
  - What it is: Enclosure type like open, ROPS, ROPS w/ AC, etc.
  - Example: `EROPS`

- Sale date (derived features: `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, `saleDayOfYear`)
  - What it is: Date of the auction. The app will derive several date-based features automatically.

- Additional optional text inputs (examples of other model fields):
  - `Hydraulics` (`Hydraulics`)
  - `Usage Band` (`UsageBand`)
  - `Base Model / Model Description` (various `fi*` fields)

Note: The app exposes only a small number of inputs by default (top model drivers). Advanced inputs (other columns) are processed if provided and will be aligned to the model's expected columns.

---

## Example walkthrough
1. Launch the app.
2. In the form, enter:
   - Manufacture Year: `2008`
   - Meter Hours: `4500`
   - Product Size: `Compact`
   - Operator Enclosure: `EROPS`
   - Sale date: today's date
3. Click `Predict sale price`.
4. The app shows an estimated sale price, a small uncertainty note (if per-tree estimates are available), a histogram of per-tree predictions (for RandomForest) and a table showing the actual inputs used (friendly label, model feature name, and value).

---

## How preprocessing works (important details)
The inference preprocessing in `app.py` mirrors the steps performed in the notebook used during training. Key steps:

1. Date feature engineering
   - From `saledate`, the app derives `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, and `saleDayOfYear`, then drops `saledate`.

2. Numeric missing values
   - For numeric columns with missing values, a companion boolean column `<col>_is_missing` is created and the missing values are filled with the column median.

3. Categorical columns
   - For non-numeric columns, a companion boolean column `<col>_is_missing` is created, then the column is converted to categorical codes using `pd.Categorical(...).codes + 1`.

4. Column alignment
   - The preprocessed DataFrame is aligned to the original `X_train_columns` list. Missing columns are added with sensible defaults (`False` for `_is_missing` flags, `0` for numeric/coded fields). Extra columns are dropped.

These steps are implemented by the `preprocess_data(df, X_train_columns)` function in `app.py`.

---

