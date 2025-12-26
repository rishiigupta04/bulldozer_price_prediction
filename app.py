import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Load the trained machine learning model and feature columns
ideal_model = joblib.load('ideal_model.joblib')
X_train_columns = joblib.load('X_train_columns.joblib')

# Helpful mapping to pick widget types
numeric_like = {"YearMade", "saleYear", "saleMonth", "MachineHoursCurrentMeter", "ModelID"}
categorical_like = {"ProductSize", "Enclosure", "Hydraulics", "UsageBand", "fiSecondaryDesc", "fiProductClassDesc", "fiBaseModel", "fiModelDesc"}

# Attempt to load training categorical mappings from preprocessed_data.csv so we use the same category ordering
def load_training_categories(csv_path, columns, chunk_size=100000):
    """Read CSV in chunks and return dict: column -> list of unique values (strings).
    If file not found or columns missing, returns an empty dict.
    """
    if not os.path.exists(csv_path):
        return {}

    uniques = {col: set() for col in columns}
    try:
        for chunk in pd.read_csv(csv_path, usecols=[c for c in columns if c in pd.read_csv(csv_path, nrows=0).columns], dtype=str, chunksize=chunk_size, low_memory=False):
            for col in chunk.columns:
                uniques[col].update(chunk[col].dropna().unique())
    except Exception:
        # if something goes wrong while reading in chunks, return what we have
        pass

    # Convert sets to sorted lists for deterministic category ordering
    return {col: sorted(list(vals)) for col, vals in uniques.items() if vals}

# define which categorical columns we'd like to load mappings for (expandable)
PREFERRED_CATEGORICALS = sorted(list(categorical_like))
TRAINING_CATEGORIES = load_training_categories('preprocessed_data.csv', PREFERRED_CATEGORICALS)

# Hardcode training categories for key features to ensure consistent encoding
# These are the string values used during training (from the notebook/UI options)
HARDCODED_CATEGORIES = {
    'ProductSize': ["Mini", "Small", "Compact", "Medium", "Large", "Large / Medium"],
    'Enclosure': ["EROPS", "EROPS w AC", "OROPS", "EROPS AC"],
    # Add more if needed, e.g., 'Hydraulics': [...], etc.
}

# Merge hardcoded with loaded (loaded may be empty if CSV has codes)
for col, cats in HARDCODED_CATEGORIES.items():
    if col not in TRAINING_CATEGORIES or not TRAINING_CATEGORIES[col]:
        TRAINING_CATEGORIES[col] = cats


# Define the preprocess_data function
def preprocess_data(df, X_train_columns):
    """
    Performs transformations on df to match the training data format.
    Assumes df has a 'saledate' column and needs feature engineering and missing value imputation.
    """

    # Feature engineering for saledate
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    df.drop("saledate", axis=1, inplace=True)

    # Fill numeric missing values and create _is_missing columns
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label + "_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
        else:
            # For categorical columns, add a binary column for missing values
            df[label + "_is_missing"] = pd.isnull(content)

            # Use training category ordering if available so codes match the training mapping
            if label in TRAINING_CATEGORIES:
                df[label] = pd.Categorical(content, categories=TRAINING_CATEGORIES[label]).codes + 1
            else:
                # Fallback: infer categories from the provided values (may not match training mapping)
                df[label] = pd.Categorical(content).codes + 1

    # Align columns with training data
    # First, handle columns that might be missing in the input but present in training
    missing_in_input_cols = set(X_train_columns) - set(df.columns)
    for col in missing_in_input_cols:
        if col.endswith("_is_missing"):
            df[col] = False
        else:
            df[col] = 0

    # Then, remove extra columns in the input that were not in training
    extra_in_input_cols = set(df.columns) - set(X_train_columns)
    df.drop(columns=list(extra_in_input_cols), inplace=True)

    # Ensure the order of columns matches X_train_columns
    df = df[X_train_columns]

    return df


# --- Friendly feature metadata and helpers (new) ---
FEATURE_META = {
    # common numeric features
    'YearMade': {
        'label': 'Manufacture Year',
        'help': 'Year the machine was manufactured. Use the 4-digit year from the plate or documentation.',
        'example': 'e.g. 2005'
    },
    'MachineHoursCurrentMeter': {
        'label': 'Meter Hours',
        'help': 'Total engine hours shown on the hour meter. If unknown, leave 0 or estimate carefully.',
        'example': 'e.g. 1234'
    },
    'ModelID': {
        'label': 'Model ID',
        'help': 'The numerical model identifier used internally (if you have the exact model code).',
        'example': 'e.g. 12345'
    },
    # sale date derived features are created automatically, but provide context
    'saleYear': {
        'label': 'Sale Year (derived)',
        'help': 'Year of the auction sale (derived from the sale date you enter).',
        'example': ''
    },
    # categorical features
    'ProductSize': {
        'label': 'Product Size',
        'help': 'General size class of the machine (Mini, Small, Compact, Medium, Large). Choose Unknown/Other if unsure.',
        'example': 'e.g. Compact'
    },
    'Enclosure': {
        'label': 'Operator Enclosure',
        'help': 'Type of operator enclosure: open, ROPS (rollover protective structure), or ROPS with AC, etc.',
        'example': 'e.g. EROPS'
    },
    'Hydraulics': {
        'label': 'Hydraulics',
        'help': 'Brief description of the hydraulics package (if known). You can leave blank if unknown.',
        'example': 'e.g. Standard'
    },
    'UsageBand': {
        'label': 'Usage Band',
        'help': 'A rough categorical band for usage (Light, Medium, Heavy). If unsure, leave blank.',
        'example': 'e.g. Light'
    },
    'fiSecondaryDesc': {
        'label': 'Secondary Description',
        'help': 'Vendor/secondary model description field from the dataset (optional).',
        'example': ''
    },
    'fiProductClassDesc': {
        'label': 'Product Class',
        'help': 'Product class description (optional).',
        'example': ''
    },
    'fiBaseModel': {
        'label': 'Base Model',
        'help': 'Base model code/name (optional).',
        'example': ''
    },
    'fiModelDesc': {
        'label': 'Model Description',
        'help': 'More detailed model description (optional).',
        'example': ''
    }
}


def friendly_label(feature_name):
    meta = FEATURE_META.get(feature_name, {})
    return meta.get('label', feature_name.replace('_', ' ').title())


def help_text(feature_name):
    meta = FEATURE_META.get(feature_name, {})
    h = meta.get('help', '')
    ex = meta.get('example', '')
    if ex:
        return f"{h} ({ex})"
    return h


# Get feature importances to find top 5 features
feature_importances = ideal_model.feature_importances_
importance_df = pd.Series(feature_importances, index=X_train_columns)
top_5_features = importance_df.nlargest(5).index.tolist()

# Known categorical options (small curated lists)
product_size_options = ["Unknown/Other", "Mini", "Small", "Compact", "Medium", "Large", "Large / Medium"]
enclosure_options = ["Unknown/Other", "EROPS", "EROPS w AC", "OROPS", "EROPS AC"]


# Set up the page early for a more polished look
st.set_page_config(page_title="Bulldozer Price Predictor", layout="centered")

# Sidebar context to guide the user
with st.sidebar:
    st.header("About this tool")
    st.write(
        "Predicts expected auction sale price of a bulldozer using a trained model."
    )
    st.markdown(
        "- Model trained on Blue Book auction data (preprocessed similarly to the included notebook)\n"
        "- Provide accurate year/hours and categorical labels when available\n"
        "- If unsure, choose 'Unknown/Other' or leave numeric defaults"
    )
    st.subheader("Top drivers in the model")
    top_features_table = (
        importance_df.loc[top_5_features]
        .rename("Importance")
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"index": "Feature", "Importance": "Importance (%)"})
    )
    st.table(top_features_table)
    st.caption("Higher % indicates greater influence on the predicted price.")

    # New small section: where to find values
    st.subheader("Where to find values / examples")
    st.markdown("Enter the most accurate information you have. Examples:")
    for feat in ['YearMade', 'MachineHoursCurrentMeter', 'ProductSize', 'Enclosure']:
        lbl = friendly_label(feat)
        ex = FEATURE_META.get(feat, {}).get('example', '')
        if ex:
            st.markdown(f"- **{lbl}**: {ex}")

# Main app content
st.title("Bulldozer Sale Price Predictor")
st.write("Enter the details below to predict the sale price of a bulldozer.")

with st.expander("What information should I provide?", expanded=True):
    st.markdown(
        "Provide the most accurate values you have (year, hours, model details). "
        "Sale date is used to derive seasonal features. If you don't know a categorical field, select 'Unknown/Other'."
    )

# Form keeps widgets together and avoids reruns until submitted
with st.form("prediction_form"):
    st.subheader("Key inputs")
    cols = st.columns(2)
    user_inputs = {}

    # Render a friendly input for each top feature, choosing widget by heuristic
    for i, feature in enumerate(top_5_features):
        col = cols[i % 2]
        # Use friendly label for the UI, but keep the key as the model feature name
        label = friendly_label(feature)
        underlying = feature

        # Show the label and the underlying feature name (for transparency)
        display_label = f"{label} ({underlying})"

        if feature in numeric_like:
            # sensible defaults for year vs hours
            if 'Year' in feature:
                val = col.slider(display_label, min_value=1950, max_value=datetime.today().year, value=2005)
                user_inputs[feature] = int(val)
                col.caption(help_text(feature))
            else:
                user_inputs[feature] = col.number_input(display_label, value=0.0)
                col.caption(help_text(feature))

        elif feature in categorical_like:
            if feature == 'ProductSize':
                choice = col.selectbox(display_label, options=product_size_options, index=0)
                user_inputs[feature] = None if choice == 'Unknown/Other' else choice
                col.caption(help_text(feature))
            elif feature == 'Enclosure':
                choice = col.selectbox(display_label, options=enclosure_options, index=0)
                user_inputs[feature] = None if choice == 'Unknown/Other' else choice
                col.caption(help_text(feature))
            else:
                user_inputs[feature] = col.text_input(display_label, value="")
                col.caption(help_text(feature))

        else:
            # Fallback to text input for unknown types
            user_inputs[feature] = col.text_input(display_label, value="")
            col.caption(help_text(feature))

    sale_date = st.date_input("Sale date", value=datetime.today().date())

    submitted = st.form_submit_button("Predict sale price")
    reset = st.form_submit_button("Reset form")

    if reset:
        rerun = getattr(st, "experimental_rerun", None)
        if callable(rerun):
            rerun()

if submitted:
    try:
        # Basic validation
        if 'YearMade' in user_inputs and user_inputs.get('YearMade'):
            if user_inputs['YearMade'] > sale_date.year:
                st.warning("Year made is after sale date — please double-check.")

        # Combine form inputs with sale date
        user_inputs['saledate'] = datetime.combine(sale_date, datetime.min.time())

        # Convert to DataFrame
        input_df = pd.DataFrame([user_inputs])

        # Preprocess and align columns
        processed = preprocess_data(input_df.copy(), X_train_columns)

        # Predict
        pred = ideal_model.predict(processed)

        # Compute per-tree predictions for uncertainty if available
        tree_preds = None
        try:
            estimators = getattr(ideal_model, 'estimators_', [])
            if estimators:
                tree_preds = np.vstack([est.predict(processed) for est in estimators])
                mean_pred = float(tree_preds.mean())
                std_pred = float(tree_preds.std())
            else:
                mean_pred = float(pred[0])
                std_pred = 0.0
        except Exception:
            mean_pred = float(pred[0])
            std_pred = 0.0

        st.success("Prediction ready")
        st.metric("Estimated sale price", f"${mean_pred:,.2f}")

        if std_pred > 0:
            ci = 1.96 * std_pred
            st.info(f"Model uncertainty (approx. 95% CI): ±${ci:,.2f} (std: ${std_pred:,.2f})")

            if tree_preds is not None and getattr(tree_preds, 'size', 0) > 0:
                with st.expander("Per-tree predictions distribution"):
                    fig, ax = plt.subplots()
                    ax.hist(tree_preds.ravel(), bins=30)
                    ax.set_xlabel('Predicted sale price')
                    ax.set_ylabel('Number of trees')
                    st.pyplot(fig)

        with st.expander("Inputs used for prediction"):
            # Show both the friendly label and the underlying feature name/value for clarity
            friendly_rows = []
            for k, v in input_df.iloc[0].items():
                friendly_rows.append({
                    'Feature': friendly_label(str(k)),
                    'Model Feature': k,
                    'Value': v
                })
            st.dataframe(pd.DataFrame(friendly_rows))

    except Exception as e:
        st.error("Error during prediction — see details below")
        st.exception(e)

st.caption("Estimate based on historical auction data; factors like condition and location are not fully captured.")