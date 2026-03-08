import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Carbon Emission Dashboard",
    layout="wide"
)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("owid-co2-data.csv")

# create carbon intensity
df["carbon_intensity"] = df["co2"] / df["primary_energy_consumption"]

# load trained model
model = pickle.load(open("co2_model.pkl","rb"))

# -------------------------------
# Sidebar Navigation
# -------------------------------
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

# -------------------------------
# Dashboard Overview
# -------------------------------
if page == "Dashboard Overview":

    st.title("AI Carbon Emission Analytics Platform")

    st.subheader("Key Climate Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric("Countries", df["country"].nunique())
    col2.metric("Years Covered", df["year"].nunique())
    col3.metric("Latest Year", int(df["year"].max()))

# -------------------------------
# Global Trends
# -------------------------------
elif page == "Global Trends":

    st.title("Global CO₂ Emissions Trend")

    global_co2 = df.groupby("year")["co2"].sum().reset_index()

    fig = px.line(
        global_co2,
        x="year",
        y="co2",
        title="Global CO₂ Emissions Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Top Emitters
# -------------------------------
elif page == "Top Emitters":

    st.title("Top CO₂ Emitting Countries")

    latest_year = df["year"].max()

    top_emitters = (
        df[df["year"] == latest_year]
        .sort_values("co2", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_emitters,
        x="co2",
        y="country",
        orientation="h",
        title="Top 10 CO₂ Emitting Countries"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Carbon Intensity Analysis
# -------------------------------
elif page == "Carbon Intensity":

    st.title("Carbon Intensity Analysis")

    latest_year = df["year"].max()

    high_intensity = (
        df[df["year"] == latest_year]
        .sort_values("carbon_intensity", ascending=False)
        .head(10)
    )

    fig = px.bar(
        high_intensity,
        x="carbon_intensity",
        y="country",
        orientation="h",
        title="Countries with Highest Carbon Intensity"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# AI Prediction Tool
# -------------------------------
elif page == "AI Predictor":

    st.title("AI CO₂ Emission Predictor")

    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Year", 1965, 2035, 2022)
        population = st.number_input("Population", value=100000000)

    with col2:
        gdp = st.number_input("GDP", value=1000000000000)
        energy = st.number_input("Primary Energy Consumption", value=10000)

    if st.button("Predict CO₂ Emissions"):

        input_data = [[year, population, gdp, energy]]

        prediction = model.predict(input_data)

        co2_value = prediction[0]

        if co2_value > 1000:
            gt = co2_value / 1000
            st.success(f"Predicted CO₂ Emissions: {gt:.2f} Gigatonnes (GtCO₂)")
        else:
            st.success(f"Predicted CO₂ Emissions: {co2_value:.2f} Million Tonnes (MtCO₂)")

        st.info("Units: CO₂ emissions are measured in Million Tonnes (MtCO₂) or Gigatonnes (GtCO2)")