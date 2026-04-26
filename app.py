
import streamlit as st
import pandas as pd
import pickle

st.title("Loan Approval Prediction App")

with open("model.pkl", "rb") as file:
    saved = pickle.load(file)

model = saved["model"]
scaler = saved["scaler"]
feature_columns = saved["feature_columns"]
numeric_features = saved["numeric_features_for_scaling"]
threshold = saved["threshold"]

st.write("Enter applicant information:")

reason = st.selectbox("Reason", [
    "debt_conslidation", "credit_card_refinancing", "home_improvement",
    "major_purchase", "cover_an_unexpected_cost", "other"
])
requested_loan_amount = st.number_input("Requested Loan Amount", min_value=0.0, value=25000.0)
fico_score = st.number_input("FICO Score", min_value=300.0, max_value=850.0, value=680.0)
fico_group = st.selectbox("FICO Score Group", ["poor", "fair", "good", "very_good", "excellent"])
employment_status = st.selectbox("Employment Status", ["full_time", "part_time", "unemployed"])
employment_sector = st.selectbox("Employment Sector", [
    "Unknown", "financials", "information_technology", "health_care", "industrials",
    "real_estate", "materials", "utilities", "energy", "consumer_staples",
    "communication_services", "consumer_discretionary"
])
monthly_income = st.number_input("Monthly Gross Income", min_value=0.0, value=6000.0)
monthly_housing = st.number_input("Monthly Housing Payment", min_value=0.0, value=1500.0)
bankrupt = st.selectbox("Ever Bankrupt or Foreclosed?", [0, 1])
lender = st.selectbox("Lender", ["A", "B", "C"])

input_df = pd.DataFrame([{
    "Reason": reason,
    "Requested_Loan_Amount": requested_loan_amount,
    "FICO_score": fico_score,
    "Fico_Score_group": fico_group,
    "Employment_Status": employment_status,
    "Employment_Sector": employment_sector,
    "Monthly_Gross_Income": monthly_income,
    "Monthly_Housing_Payment": monthly_housing,
    "Ever_Bankrupt_or_Foreclose": bankrupt,
    "Lender": lender
}])

input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
input_encoded[numeric_features] = scaler.transform(input_encoded[numeric_features])

prob = model.predict_proba(input_encoded)[:, 1][0]
prediction = int(prob >= threshold)

st.metric("Predicted Approval Probability", f"{prob:.2%}")
st.write("Prediction:", "Approved" if prediction == 1 else "Denied")
st.caption(f"Classification threshold: {threshold:.3f}")
