import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Machine Failure Prediction",
    page_icon="🏭",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "models/machine_failure_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏭 Industrial Machine Failure Prediction")

st.write(
    "Enter the machine operating parameters below to predict "
    "whether the machine is likely to experience a failure."
)

st.divider()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------

st.subheader("Machine Parameters")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"],
    index=1
)

air_temperature = st.number_input(
    "Air temperature",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.number_input(
    "Process temperature",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.number_input(
    "Rotational speed",
    min_value=1,
    max_value=5000,
    value=1500,
    step=1
)

torque = st.number_input(
    "Torque",
    min_value=0.0,
    max_value=100.0,
    value=55.0,
    step=0.1
)

tool_wear = st.number_input(
    "Tool wear",
    min_value=0.0,
    max_value=300.0,
    value=200.0,
    step=1.0
)

st.divider()

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

if process_temperature <= air_temperature:
    st.warning(
        "Process temperature should normally be higher than "
        "air temperature."
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Failure", use_container_width=True):

    if rotational_speed <= 0:
        st.error("Rotational speed must be greater than 0.")
        st.stop()

    if torque < 0:
        st.error("Torque cannot be negative.")
        st.stop()

    if tool_wear < 0:
        st.error("Tool wear cannot be negative.")
        st.stop()

    # IMPORTANT:
    # These column names MUST match the training dataset.

    row = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature": [air_temperature],
        "Process temperature": [process_temperature],
        "Rotational speed": [rotational_speed],
        "Torque": [torque],
        "Tool wear": [tool_wear]
    })

    try:
        prediction = int(model.predict(row)[0])

        probability = float(
            model.predict_proba(row)[0][1]
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.divider()

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ MACHINE FAILURE PREDICTED")

        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "The model predicts that the machine may experience "
            "a failure within the defined prediction window."
        )

    else:

        st.success("✅ NO MACHINE FAILURE PREDICTED")

        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "The model predicts that the machine is unlikely "
            "to experience a failure within the defined prediction window."
        )

    # --------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------

    with st.expander("View Input Parameters"):

        st.dataframe(
            row,
            use_container_width=True
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Industrial Machine Failure Prediction | "
    "Machine Learning Capstone Project"
)