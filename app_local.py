import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "artifacts/credit_score_LogisticRegression.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Credit Score App", layout="centered")
st.title("📊 Prediksi Credit Score")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        annual_income = st.number_input("Annual Income", value=50000.0)
        salary = st.number_input("Monthly Inhand Salary", value=4000.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", value=1)
        num_credit_card = st.number_input("Num Credit Cards", value=1)
        interest_rate = st.number_input("Interest Rate (%)", value=5)
        num_loans = st.number_input("Num Loans", value=1)
        total_emi = st.number_input("Total EMI per month", value=100.0)
        credit_util = st.number_input("Credit Utilization Ratio", value=30.0)
        monthly_bal = st.number_input("Monthly Balance", value=500.0)
        delay_days = st.number_input("Delay from Due Date", value=0)
    with col2:
        num_delayed = st.number_input("Num Delayed Payments", value=0)
        credit_history = st.number_input("Credit History Age (Months)", value=12.0)
        credit_inquiries = st.number_input("Credit Inquiries", value=1.0)
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
        min_payment = st.selectbox("Payment of Min Amount", ["Yes", "No"])
        payment_behaviour = st.selectbox("Payment Behaviour", ["Low_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Large_value_payments"])
        occupation = st.text_input("Occupation", "Engineer")
        month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
        changed_limit = st.number_input("Changed Credit Limit", value=10.0)
        outstanding_debt = st.number_input("Outstanding Debt", value=500.0)
        amount_invested = st.number_input("Amount invested monthly", value=100.0)

    submit = st.form_submit_button("Prediksi")

if submit:
    input_data = pd.DataFrame([{
        "Age": age, "Annual_Income": annual_income, "Monthly_Inhand_Salary": salary,
        "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_loans,
        "Delay_from_due_date": delay_days, "Num_of_Delayed_Payment": num_delayed,
        "Credit_History_Age": credit_history, "Num_Credit_Inquiries": credit_inquiries,
        "Credit_Mix": credit_mix, "Payment_of_Min_Amount": min_payment,
        "Payment_Behaviour": payment_behaviour, "Occupation": occupation, "Month": month,
        "Total_EMI_per_month": total_emi, "Credit_Utilization_Ratio": credit_util,
        "Monthly_Balance": monthly_bal, "Changed_Credit_Limit": changed_limit,
        "Outstanding_Debt": outstanding_debt, "Amount_invested_monthly": amount_invested
    }])

    # predict menggunakan artifacts
    prediction = model.predict(input_data)
    target_map = {0: "Poor", 1: "Standard", 2: "Good"}
    result = target_map.get(int(prediction[0]), "Unknown")
    
    st.success(f"Hasil Prediksi: **{result}**")
