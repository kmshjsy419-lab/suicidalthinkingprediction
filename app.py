import streamlit as st
import pandas as pd
import joblib

st.title("Suicidal Thinking Probability Prediction")
st.write("Enter the following information:")

# 모델 불러오기
try:
    model = joblib.load("model_gbm.pkl")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()
    
# numeric
Age = st.number_input("Age", 19, 100, 30)

# ===== categorical input (label -> code) =====

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
    ["Smoker", "Non-smoker"]
)
smoking_map = {
    "Smoker": 1,
    "Non-smoker": 2
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
    ["Severe", "High", "Moderate", "Low"]
)
stress_map = {
    "Severe": 1,
    "High": 2,
    "Moderate": 3,
    "Low": 4
}
Stress_status = stress_map[stress_label]

depressive_label = st.selectbox(
    "Depressive symptoms",
    ["Yes","No"]
)
depressive_map = {
    "Yes": 1,
    "No": 0
}
Depressive_symptoms = depressive_map[depressive_label]

living_alone_label = st.selectbox(
    "Living alone",
    ["Yes","No"]
)
living_alone_map = {
    "Yes": 1,
    "No": 2
}
Living_alone = living_alone_map[living_alone_label]

employment_label = st.selectbox(
    "Employment status",
    ["Employed", "Unemployed"]
)
employment_map = {
    "Employed": 1,
    "Unemployed": 2
}
Employment_status = employment_map[employment_label]

# 데이터프레임 생성
input_df = pd.DataFrame([{
    'Sex': Sex,
    'Region': Region,
    'BMI': BMI,
    'Education': Education,
    'Household_income': Household_income,
    'Smoking_status': Smoking_status,
    'Drink_frequency': Drink_frequency,
    'Stress_status': Stress_status,
    'Depressive_symptoms': Depressive_symptoms,
    'Living_alone': Living_alone,
    'Employment_status': Employment_status,
    'Age': Age
}])

if st.button("Predict probability"):
    try:
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("Predicted probability of suicidal thinking")
        st.write(f"{prob:.2%}")

        st.write("Input values sent to the model:")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
