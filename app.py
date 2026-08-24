import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Suicidal Thinking Prediction",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .app-subtitle {
        color: #5f6368;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.4rem;
    }
    .result-number {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.2rem 0 0.4rem 0;
    }
    .result-label {
        color: #5f6368;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Suicidal Thinking Probability Prediction")
st.markdown(
    '<div class="app-subtitle">A research-based tool for estimating the probability of suicidal ideation.</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("#### Before you begin")
    st.write(
        "This tool provides a model-based estimate for research and educational purposes. "
        "It is not a clinical diagnosis and should not be used as the sole basis for clinical decisions."
    )

st.markdown("### Enter information")

# 모델 불러오기
try:
    model = joblib.load("model_gbm_pipeline.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# numeric
Age = st.number_input("Age", min_value=19, max_value=100, value=30, step=1)

# categorical
sex_label = st.selectbox("Sex", ["Male", "Female"])
sex_map = {
    "Male": 1,
    "Female": 2
}
Sex = sex_map[sex_label]

region_label = st.selectbox("Region", ["Urban", "Rural"])
region_map = {
    "Urban": 1,
    "Rural": 2
}
Region = region_map[region_label]

bmi_label = st.selectbox(
    "BMI category",
    ["Underweight", "Normal", "Overweight", "Obese"]
)
bmi_map = {
    "Underweight": 1,
    "Normal": 2,
    "Overweight": 3,
    "Obese": 4
}
BMI = bmi_map[bmi_label]

education_label = st.selectbox(
    "Education",
    ["High school or below", "College or above"]
)
education_map = {
    "High school or below": 1,
    "College or above": 2
}
Education = education_map[education_label]

income_label = st.selectbox(
    "Household income",
    ["Quartile 1 (lowest)", "Quartile 2", "Quartile 3", "Quartile 4 (highest)"]
)
income_map = {
    "Quartile 1 (lowest)": 1,
    "Quartile 2": 2,
    "Quartile 3": 3,
    "Quartile 4 (highest)": 4
}
Household_income = income_map[income_label]

smoking_label = st.selectbox(
    "Smoking status",
    ["Non-smoker", "Smoker"]
)
smoking_map = {
    "Non-smoker": 0,
    "Smoker": 1
}
Smoking_status = smoking_map[smoking_label]

drink_label = st.selectbox(
    "Alcohol consumption (days/month)",
    ["<2", "2-4", "≥5"]
)
drink_map = {
    "<2": 1,
    "2-4": 2,
    "≥5": 3
}
Drink_frequency = drink_map[drink_label]

stress_label = st.selectbox(
    "Stress status",
    ["Low", "Moderate", "High", "Severe"]
)
stress_map = {
    "Low": 4,
    "Moderate": 3,
    "High": 2,
    "Severe": 1
}
Stress_status = stress_map[stress_label]

depressive_label = st.selectbox(
    "Depressive symptoms",
    ["No", "Yes"]
)
depressive_map = {
    "No": 0,
    "Yes": 1
}
Depressive_symptoms = depressive_map[depressive_label]

living_alone_label = st.selectbox(
    "Living alone",
    ["No", "Yes"]
)
living_alone_map = {
    "No": 2,
    "Yes": 1
}
Living_alone = living_alone_map[living_alone_label]

employment_label = st.selectbox(
    "Employment status",
    ["Unemployed", "Employed"]
)
employment_map = {
    "Unemployed": 0,
    "Employed": 1
}
Employment_status = employment_map[employment_label]

# 학습 데이터 순서와 동일하게 생성
input_df = pd.DataFrame([{
    "Age": Age,
    "Sex": Sex,
    "Region": Region,
    "BMI": BMI,
    "Education": Education,
    "Household_income": Household_income,
    "Smoking_status": Smoking_status,
    "Drink_frequency": Drink_frequency,
    "Stress_status": Stress_status,
    "Depressive_symptoms": Depressive_symptoms,
    "Living_alone": Living_alone,
    "Employment_status": Employment_status
}])

if st.button("Predict probability", type="primary", use_container_width=True):
    try:
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.markdown("## Prediction result")

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="result-label">Model-estimated probability</div>
                <div class="result-number">{prob:.1%}</div>
                """,
                unsafe_allow_html=True
            )
            st.caption(
                "This value represents the probability estimated by the model based on the information entered."
            )

        with st.container(border=True):
            st.markdown("#### How to interpret this result")
            st.write(
                "A higher estimated probability indicates that the entered characteristics are more similar to patterns "
                "associated with suicidal ideation in the model-development data. However, this estimate should not be "
                "interpreted as a diagnosis or as confirmation of suicidal ideation."
            )

            if prob >= 0.50:
                st.write(
                    "This estimate is above the model's default 0.50 classification threshold. "
                    "This threshold is statistical rather than clinical and should not be used alone to determine individual risk."
                )
            else:
                st.write(
                    "This estimate is below the model's default 0.50 classification threshold. "
                    "A value below this threshold does not rule out suicidal ideation or the need for further assessment."
                )

        with st.container(border=True):
            st.markdown("#### Important limitations")
            st.write(
                "The model may produce both false-positive and false-negative predictions. "
                "A higher estimated probability does not confirm suicidal ideation, and a lower estimated probability does not exclude it. "
                "Predictions may also vary across populations and settings."
            )

        with st.container(border=True):
            st.markdown("#### Additional guidance")
            st.write(
                "This application is intended for research and educational purposes and does not replace a clinical interview, "
                "professional judgment, or a comprehensive suicide-risk assessment. If there are concerns about suicidal thoughts "
                "or personal safety, professional assessment should be sought regardless of the predicted probability."
            )

        with st.expander("View information used for this prediction"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
