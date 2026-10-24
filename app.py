import streamlit as st
import pandas as pd
import joblib

# Load our trained machine learning model
saved = joblib.load("machine_failure_model.pkl")
model = saved["model"]
columns = saved["columns"]

st.title("Industrial Machine Failure Prediction")
st.write(
    "Enter the machine's operating information "
    "to receive a prediction from our ML model."
)

st.warning(
    "Student project prototype. Predictions are experimental "
    "and should not be used as the sole basis for real maintenance decisions."
)

with st.form("machine_form"):
    machine_type = st.selectbox(
        "Machine type", ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air temperature (K)", min_value=250.0,
        max_value=350.0, value=298.0
    )

    process_temp = st.number_input(
        "Process temperature (K)", min_value=250.0,
        max_value=400.0, value=308.0
    )

    speed = st.number_input(
        "Rotational speed (rpm)", min_value=0,
        max_value=5000, value=1500
    )

    torque = st.number_input(
        "Torque (Nm)", min_value=0.0,
        max_value=200.0, value=40.0
    )

    wear = st.number_input(
        "Tool wear (minutes)", min_value=0,
        max_value=300, value=100
    )

    submitted = st.form_submit_button("Predict machine condition")

if submitted:
    values = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
    }

    # Arrange the inputs in the exact format used during training
    row = pd.DataFrame([[0] * len(columns)], columns=columns)

    for name, value in values.items():
        if name in row.columns:
            row.loc[0, name] = value

    type_column = "Type_" + machine_type
    if type_column in row.columns:
        row.loc[0, type_column] = 1

    prediction = int(model.predict(row)[0])

    if prediction == 1:
        st.error("The model predicts a possible machine failure.")
        st.write("Recommendation: inspect the machine and verify its condition.")
    else:
        st.success("The model predicts no failure in this record.")
        st.write("Continue routine monitoring and maintenance.")
