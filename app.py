import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px
history_file = "prediction_history.csv"

# =====================================================
# LOAD MODEL AND SCALER
# =====================================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")


# =====================================================
# PAGE TITLE
# =====================================================

st.title("Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether the customer will churn."
)


# =====================================================
# CUSTOMER DETAILS
# =====================================================

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)


monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=500.0,
    value=70.0
)


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=840.0
)


senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)


partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)


dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)


contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)


internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)


# =====================================================
# PREDICT BUTTON
# =====================================================

predict_button = st.button(
    "Predict Churn",
    key="predict_churn_button"
)


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    # Convert Yes/No values into numbers

    senior_citizen_value = (
        1 if senior_citizen == "Yes" else 0
    )

    partner_value = (
        1 if partner == "Yes" else 0
    )

    dependents_value = (
        1 if dependents == "Yes" else 0
    )


    # Convert Contract into numbers

    contract_value = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }[contract]


    # Convert Internet Service into numbers

    internet_service_value = {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    }[internet_service]


    # =================================================
    # CREATE INPUT DATA
    # =================================================

    input_data = pd.DataFrame([[
        tenure,
        monthly_charges,
        total_charges,
        senior_citizen_value,
        partner_value,
        dependents_value,
        contract_value,
        internet_service_value
    ]])


    # =================================================
    # SCALE INPUT
    # =================================================

    input_scaled = scaler.transform(input_data)


    # =================================================
    # MAKE PREDICTION
    # =================================================

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]


    # =================================================
    # SAVE PREDICTION HISTORY
    # =================================================

    history_file = "prediction_history.csv"
# SAVE PREDICTION HISTORY
    history_file = "prediction_history.csv"
# SAVE PREDICTION HISTORY
    history_file = "prediction_history.csv"

    new_record = pd.DataFrame({
        "Member ID": [
            len(pd.read_csv(history_file)) + 1
            if os.path.exists(history_file) else 1
        ],
        "Date": [datetime.now().strftime("%Y-%m-%d")],
        "Tenure": [tenure],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Contract": [contract],
        "Internet Service": [internet_service],
        "Probability": [probability * 100],
        "Prediction": [
            "Churn" if prediction == 1 else "Not Churn"
        ]
    })

    if os.path.exists(history_file):
        new_record.to_csv(
            history_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_record.to_csv(history_file, index=False)
    # =================================================
    # PREDICTION RESULT
    # =================================================

    st.write(
        f"### Churn Probability: {probability * 100:.2f}%"
    )


    if prediction == 1:

        st.error(
            "🔴 Customer is likely to CHURN"
        )

    else:

        st.success(
            "🟢 Customer is NOT likely to CHURN"
        )


    # =================================================
    # CHURN RISK
    # =================================================

    st.progress(float(probability))


    if probability >= 0.5:

        st.warning(
            "⚠️ High churn risk"
        )

    else:

        st.info(
            "🟦 Low churn risk"
        )


    # =================================================
    # CHURN GRAPHS - SIDE BY SIDE
    # =================================================

    st.subheader("Churn Analysis")


    col1, col2 = st.columns(2)


    # =================================================
    # PIE CHART
    # =================================================

    with col1:

        st.subheader("📊 Churn Probability")


        pie_data = pd.DataFrame({
            "Status": [
                "Churn",
                "Not Churn"
            ],

            "Probability": [
                probability * 100,
                (1 - probability) * 100
            ]
        })


        fig_pie = px.pie(
            pie_data,
            names="Status",
            values="Probability",
            hole=0.4
        )


        fig_pie.update_layout(
            title="Customer Churn Probability",
            showlegend=True
        )


        st.plotly_chart(
            fig_pie,
            width="stretch"
        )


    # =================================================
    # BAR GRAPH
    # =================================================

    with col2:

        st.subheader("📊 Churn Prediction Graph")


        chart_data = pd.DataFrame({

            "Status": [
                "Churn",
                "Not Churn"
            ],

            "Probability": [
                probability * 100,
                (1 - probability) * 100
            ]
        })


        fig_bar = px.bar(
            chart_data,
            x="Status",
            y="Probability",
            text="Probability",
            title="Churn Prediction"
        )


        fig_bar.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )


        fig_bar.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="Prediction",
            yaxis_range=[0, 100]
        )


        st.plotly_chart(
            fig_bar,
            width="stretch"
        )


# =====================================================
# CUSTOMER CHURN HISTORY
# =====================================================

st.subheader("📅 Customer Churn History")


history_file = "prediction_history.csv"


if os.path.exists(history_file):

    history_df = pd.read_csv(history_file)


    # Convert Date column

    history_df["Date"] = pd.to_datetime(
        history_df["Date"],
        errors="coerce"
    )


    # Remove invalid dates

    history_df = history_df.dropna(
        subset=["Date"]
    )


    # =================================================
    # SELECT TIME PERIOD
    # =================================================

    period = st.selectbox(
        "Select Time Period",
        [
            "Daily",
            "Weekly",
            "Monthly"
        ]
    )


    # =================================================
    # DAILY
    # =================================================

    if period == "Daily":

        history_df["Period"] = (
            history_df["Date"]
            .dt.strftime("%Y-%m-%d")
        )


        graph_data = (
            history_df
            .groupby(
                ["Period", "Prediction"]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )


        graph_title = "Daily Customer Churn"


    # =================================================
    # WEEKLY
    # =================================================

    elif period == "Weekly":

        history_df["Period"] = (
            history_df["Date"]
            .dt.to_period("W")
            .astype(str)
        )


        graph_data = (
            history_df
            .groupby(
                ["Period", "Prediction"]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )


        graph_title = "Weekly Customer Churn"


    # =================================================
    # MONTHLY
    # =================================================

    else:

        history_df["Period"] = (
            history_df["Date"]
            .dt.to_period("M")
            .astype(str)
        )


        graph_data = (
            history_df
            .groupby(
                ["Period", "Prediction"]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )


        graph_title = "Monthly Customer Churn"


    # =================================================
    # HISTORY GRAPH
    # =================================================

    fig_history = px.bar(
        graph_data,
        x="Period",
        y="Customers",
        color="Prediction",
        barmode="group",
        text="Customers",
        title=graph_title
    )


    fig_history.update_traces(
        textposition="outside"
    )


    fig_history.update_layout(
        xaxis_title="Date / Period",
        yaxis_title="Number of Customers"
    )


    st.plotly_chart(
        fig_history,
        width="stretch"
    )


    # =================================================
    # HISTORY TABLE
    # =================================================

    st.subheader("Prediction Records")


    display_history = history_df[
        [
            "Date",
            "Probability",
            "Prediction"
        ]
    ].copy()


    display_history["Date"] = (
        display_history["Date"]
        .dt.strftime("%Y-%m-%d")
    )


    display_history["Probability"] = (
        display_history["Probability"]
        .round(2)
    )


    st.dataframe(
        display_history,
        width="stretch"
    )


else:

    st.info(
        "No customer prediction history available yet."
    )

    # =====================================================
# SEARCH MEMBER
# =====================================================

st.subheader("🔎 Search Member")

if os.path.exists(history_file):

    member_history = pd.read_csv(history_file)

    if "Member ID" in member_history.columns:

        member_number = st.number_input(
            "Enter Member Number",
            min_value=1,
            max_value=len(member_history),
            value=1,
            step=1
        )

        member_history["Member ID"] = pd.to_numeric(
    member_history["Member ID"], errors="coerce"
)

member_data = member_history[
    member_history["Member ID"] == int(member_number)
]

if not member_data.empty:
    customer = member_data.iloc[0]

    st.subheader(
        f"👤 Member {int(member_number)} Details"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Date:** {customer['Date']}")

        st.write(
            f"**Tenure:** {customer['Tenure']} months"
        )

        st.write(
            f"**Monthly Charges:** {customer['Monthly Charges']}"
        )

        st.write(
            f"**Total Charges:** {customer['Total Charges']}"
        )

        st.write(
            f"**Senior Citizen:** {customer['Senior Citizen']}"
        )

        st.write(
            f"**Partner:** {customer['Partner']}"
        )

        with col2:

                st.write(
                    f"**Dependents:** {customer['Dependents']}"
                )

                st.write(
                    f"**Contract:** {customer['Contract']}"
                )

                st.write(
                    f"**Internet Service:** {customer['Internet Service']}"
                )

                st.write(
                    f"**Churn Probability: {float(customer['Probability']):.2f}%**"
                )

                if customer["Prediction"] == "Churn":

                    st.error(
                        "🔴 Customer is likely to CHURN"
                    )

                else:

                    st.success(
                        "🟢 Customer is NOT likely to CHURN"
                    )

else:
    st.warning("Member not found.")

    st.info(
        "Please delete the old prediction_history.csv and make a new prediction."
    )