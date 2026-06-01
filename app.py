import streamlit as st
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors


# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Antibacterial Activity Predictor",
    page_icon="🧪",
    layout="wide"
)


# =========================
# Header
# =========================

st.title("🧪 Antibacterial Activity Predictor")

st.markdown(
    """
### Machine Learning-Based QSAR Screening Platform
"""
)

st.markdown("---")


# =========================
# Custom style
# =========================

st.markdown(
    """
<style>

.description-box {
    background-color: #E8F5E9;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #2E7D32;
    margin-bottom: 15px;
}

.output-box {
    background-color: #F5F5F5;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #616161;
    margin-bottom: 15px;
}

.limit-box {
    background-color: #FFF3E0;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #EF6C00;
    margin-bottom: 25px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================
# Information boxes
# =========================

st.markdown(
    """
<div class="description-box">

<h3>📖 Project Description</h3>

This web application provides a machine learning–based tool for predicting whether a compound has antibacterial activity using its SMILES representation.

The model was developed using experimentally validated antibacterial compounds and is based on a Decision Tree algorithm, achieving an accuracy of 0.92. It is designed to support early-stage antibacterial screening.

</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="limit-box">

<h3>⚠️ Limitations</h3>

<ul>
<li>For research use only.</li>
<li>Not intended for clinical decision making.</li>
<li>The model learns structural patterns from known antibacterial compounds.</li>
<li>Performance may decrease for underrepresented chemical classes.</li>
<li>Experimental validation remains essential.</li>
</ul>

</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="output-box">

<h3>🔍 Understanding Prediction Output</h3>

<b>Active</b><br>
Predicted to possess antibacterial activity.

<br><br>

<b>Inactive</b><br>
Predicted not to possess antibacterial activity.

<br><br>

Predictions should be considered preliminary screening results and validated experimentally.

</div>
""",
    unsafe_allow_html=True
)


# =========================
# Load saved files
# =========================

@st.cache_resource
def load_files():
    model = joblib.load("final_top20_dt_model.pkl")
    scaler = joblib.load("final_top20_scaler.pkl")
    selected_features = joblib.load("selected_features.pkl")

    return model, scaler, selected_features


final_dt_model, scaler_top20, selected_features = load_files()


# =========================
# Descriptor calculation
# =========================

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return mol


def calculate_descriptors(smiles):
    mol = smiles_to_mol(smiles)

    descriptor_values = {
        descriptor_name: descriptor_function(mol)
        for descriptor_name, descriptor_function in Descriptors.descList
    }

    descriptor_df = pd.DataFrame([descriptor_values])

    missing_features = [
        feature
        for feature in selected_features
        if feature not in descriptor_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing descriptors: {missing_features}"
        )

    descriptor_df = descriptor_df[selected_features]

    return descriptor_df


# =========================
# Prediction function
# =========================

def predict_antibacterial_activity(smiles):
    descriptor_df = calculate_descriptors(smiles)

    scaled_data = scaler_top20.transform(descriptor_df)

    prediction = final_dt_model.predict(scaled_data)[0]

    probability = final_dt_model.predict_proba(scaled_data)[0]

    activity = "Active" if prediction == 1 else "Inactive"

    result = {
        "SMILES": smiles,
        "Prediction": activity,
        "Probability_Inactive": float(probability[0]),
        "Probability_Active": float(probability[1]),
    }

    return result, descriptor_df


# =========================
# Single prediction
# =========================

st.markdown("---")
st.header("🧬 Single SMILES Prediction")

example_smiles = "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"

smiles = st.text_input(
    "Enter SMILES",
    value=example_smiles
)

if st.button("Predict"):

    try:
        result, descriptor_df = predict_antibacterial_activity(smiles)

        st.subheader("Prediction Result")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                label="Prediction",
                value=result["Prediction"]
            )

        with metric_col2:
            st.metric(
                label="Probability of Active",
                value=f"{result['Probability_Active']:.2%}"
            )

        with metric_col3:
            st.metric(
                label="Probability of Inactive",
                value=f"{result['Probability_Inactive']:.2%}"
            )

        if result["Prediction"] == "Active":
            st.success("This compound is predicted to have antibacterial activity.")
        else:
            st.error("This compound is predicted not to have antibacterial activity.")

        result_df = pd.DataFrame([result])

        st.download_button(
            label="Download Prediction Result",
            data=result_df.to_csv(index=False),
            file_name="single_prediction_result.csv",
            mime="text/csv"
        )

        with st.expander("Show selected descriptors used by model"):
            st.dataframe(descriptor_df)

    except Exception as e:
        st.error(f"Error: {e}")


st.markdown("---")


# =========================
# Batch prediction
# =========================

st.header("📄 Batch Prediction from CSV")

st.write(
    "Upload a CSV file containing a column named `SMILES`."
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        input_df = pd.read_csv(uploaded_file)

        if "SMILES" not in input_df.columns:
            st.error("CSV file must contain a column named 'SMILES'.")

        else:
            st.subheader("Uploaded Data")
            st.dataframe(input_df.head())

            if st.button("Run Batch Prediction"):

                batch_results = []

                progress_bar = st.progress(0)

                for index, row in input_df.iterrows():

                    smiles_value = row["SMILES"]

                    try:
                        result, descriptor_df = predict_antibacterial_activity(
                            smiles_value
                        )

                        batch_results.append(result)

                    except Exception as e:
                        batch_results.append(
                            {
                                "SMILES": smiles_value,
                                "Prediction": "Error",
                                "Probability_Inactive": None,
                                "Probability_Active": None,
                                "Error": str(e)
                            }
                        )

                    progress_bar.progress(
                        (index + 1) / len(input_df)
                    )

                batch_result_df = pd.DataFrame(batch_results)

                st.subheader("Batch Prediction Results")
                st.dataframe(batch_result_df)

                st.download_button(
                    label="Download Batch Prediction Results",
                    data=batch_result_df.to_csv(index=False),
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")


st.markdown("---")

st.caption(
    "Developed for research purposes. Model predictions should be interpreted with caution."
)