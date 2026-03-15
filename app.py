import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="AI Carbon Emission Analytics",
    layout="wide"
)

st.title("AI Carbon Emission Analytics Platform")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("owid-co2-data.csv")

    # Handle divide by zero safely
    df["carbon_intensity"] = df["co2"] / df["primary_energy_consumption"]
    df["carbon_intensity"] = df["carbon_intensity"].replace(
        [float("inf"), -float("inf")], None
    )

    return df


df = load_data()

# -----------------------------
# Load ML Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("co2_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Go to",
    [
        "Dashboard Overview",
        "Global Trends",
        "Top Emitters",
        "Carbon Intensity",
        "AI Predictor"
    ]
)

# -----------------------------
# Dashboard Overview
# -----------------------------
if page == "Dashboard Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Countries", df["country"].nunique())
    col2.metric("Years Covered", df["year"].nunique())
    col3.metric("Latest Year", int(df["year"].max()))

# -----------------------------
# Global Emission Trends
# -----------------------------
elif page == "Global Trends":

    st.header("Global CO₂ Emissions Trend")

    global_co2 = df.groupby("year")["co2"].sum()

    fig, ax = plt.subplots()
    ax.plot(global_co2.index, global_co2.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions")
    ax.set_title("Global CO₂ Emissions Over Time")

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Top CO₂ Emitters
# -----------------------------
elif page == "Top Emitters":

    st.header("Top CO₂ Emitting Countries")

    latest_year = df["year"].max()

    top_emitters = (
        df[df["year"] == latest_year]
        .sort_values("co2", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()

    ax.barh(top_emitters["country"], top_emitters["co2"])
    ax.set_xlabel("CO₂ Emissions")
    ax.set_title("Top 10 CO₂ Emitting Countries")
    ax.invert_yaxis()

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Carbon Intensity Analysis
# -----------------------------
elif page == "Carbon Intensity":

    st.header("Carbon Intensity Analysis")

    latest_year = df["year"].max()

    intensity_data = (
        df[df["year"] == latest_year]
        .sort_values("carbon_intensity", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()

    ax.barh(intensity_data["country"], intensity_data["carbon_intensity"])
    ax.set_xlabel("Carbon Intensity")
    ax.set_title("Countries with Highest Carbon Intensity")
    ax.invert_yaxis()

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# AI Prediction Tool
# -----------------------------
elif page == "AI Predictor":

    st.header("AI CO₂ Emission Predictor")

    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Year", 1965, 2035, 2022)
        population = st.number_input("Population", value=100000000)

    with col2:
        gdp = st.number_input("GDP (USD)", value=1000000000000)
        energy = st.number_input("Primary Energy Consumption (TWh)", value=10000)

    if st.button("Predict CO₂ Emissions"):

        input_data = [[year, population, gdp, energy]]

        prediction = model.predict(input_data)[0]

        if prediction > 1000:
            gt = prediction / 1000
            st.success(f"Predicted CO₂ Emissions: {gt:.2f} Gigatonnes (GtCO₂)")
        else:
            st.success(f"Predicted CO₂ Emissions: {prediction:.2f} Million Tonnes (MtCO₂)")

        st.info("CO₂ emissions are measured in Million Tonnes (MtCO₂) or Gigatonnes (GtCO₂).")