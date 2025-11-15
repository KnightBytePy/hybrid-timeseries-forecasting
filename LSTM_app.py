import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load model and scaler
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("xgb_walmart_model.pkl")
    scaler = joblib.load("scaler_walmart.pkl")
    return model, scaler

model, scaler = load_model()

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Walmart Sales Forecast (XGBoost)", layout="centered")

st.title("📊 Walmart Weekly Sales Forecast — Mini App")
st.markdown(
    "Upload a dataset with the same feature structure used in training "
    "and get **predicted weekly sales** instantly using the trained XGBoost model."
)

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.write(df.head())

    # Use all numeric features
    X = df.select_dtypes(include=[np.number])
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    df["Predicted_Weekly_Sales"] = preds

    st.success("✅ Predictions generated successfully!")
    st.write(df[["Predicted_Weekly_Sales"]].head())

    # Visualization
    st.subheader("📈 Predicted Weekly Sales Trend")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Predicted_Weekly_Sales"], color="royalblue", linewidth=2)
    ax.set_xlabel("Observations")
    ax.set_ylabel("Predicted Sales ($)")
    ax.set_title("Predicted Weekly Sales (XGBoost Model)")
    st.pyplot(fig)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Predictions as CSV",
        data=csv,
        file_name="predicted_sales.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Upload your CSV file to start forecasting.")