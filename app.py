import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import geopandas as gpd

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="AI Carbon Emission Analytics",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("owid-co2-data.csv")

# Safe carbon intensity calculation
df["carbon_intensity"] = df["co2"] / df["primary_energy_consumption"]
df["carbon_intensity"] = df["carbon_intensity"].replace([float("inf"), -float("inf")], None)

# Load ML model
model = pickle.load(open("co2_model.pkl", "rb"))

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
        "Emission Map",
        "AI Predictor"
    ]
)

# -----------------------------
# Dashboard Overview
# -----------------------------
if page == "Dashboard Overview":

    st.title("AI Carbon Emission Analytics Platform")

    col1, col2, col3 = st.columns(3)

    col1.metric("Countries", df["country"].nunique())
    col2.metric("Years Covered", df["year"].nunique())
    col3.metric("Latest Year", int(df["year"].max()))

# -----------------------------
# Global Emission Trends
# -----------------------------
elif page == "Global Trends":

    st.title("Global CO₂ Emissions Trend")

    global_co2 = df.groupby("year")["co2"].sum()

    fig, ax = plt.subplots()
    ax.plot(global_co2.index, global_co2.values)

    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions")
    ax.set_title("Global CO₂ Emissions Over Time")

    st.pyplot(fig)

# -----------------------------
# Top CO2 Emitters
# -----------------------------
elif page == "Top Emitters":

    st.title("Top CO₂ Emitting Countries")

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

    st.pyplot(fig)

# -----------------------------
# Carbon Intensity Analysis
# -----------------------------
elif page == "Carbon Intensity":

    st.title("Carbon Intensity Analysis")

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

    st.pyplot(fig)

# -----------------------------
# Global Emission Map
# -----------------------------
elif page == "Emission Map":

    st.title("Global CO₂ Emission Map")

    latest_year = df["year"].max()
    map_data = df[df["year"] == latest_year]

    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    merged = world.merge(map_data, how="left", left_on="ISO_A3", right_on="iso_code")

    fig, ax = plt.subplots(figsize=(12,6))

    merged.plot(
        column="co2",
        cmap="Reds",
        linewidth=0.5,
        ax=ax,
        edgecolor="black",
        legend=True
    )

    ax.set_title("Global CO₂ Emissions by Country")

    st.pyplot(fig)

# -----------------------------
# AI Prediction Tool
# -----------------------------
elif page == "AI Predictor":

    st.title("AI CO₂ Emission Predictor")

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