# 🌍 GeoWatt ZH – Geothermal Potential Explorer for Canton Zürich

**Author:** Daniel Herrera  
**Date:** 03/05/2025  

# <img src="images/banner_geowatt.png" alt="GeoWatt ZH Banner"/>

---

## 1. Project Overview

**GeoWatt ZH** is an interactive, Streamlit-based web application that allows users to estimate shallow geothermal potential at any point within the boundaries of the Canton of Zürich. Built with a combination of machine learning, GIS analysis, and official energy data, the app helps inform sustainable heating strategies and preliminary system planning.

Key features include:

- 🔍 **Location-Based Analysis**: Users can search for any address in Zürich and receive regulatory and geothermal information for that point.
- 📊 **Heat Yield Prediction**: Uses a trained XGBoost model to estimate the expected heat output based on probe configuration.
- 🎯 **Optimization Tool**: Suggests optimal depth and probe count to maximize energy yield per probe while discouraging oversized systems.
- 🗺️ **Interactive Map**: Visualizes surrounding installations, subsurface restrictions, and estimated potential via dynamic layers.
- 💰 **Financial Incentive Estimator**: Calculates approximate subsidies based on the 2025 cantonal energy program.

➡️ **[Live App](https://geowatt-zh.streamlit.app/)**  

---

## 2. Repository Structure

- 📁 **`assets/`**: Visual assets and UI screenshots.
- 📁 **`models/`**: PKL model files.
- 📁 **`data/`**: GeoJSON data files, including depth regulations and borehole data.
- 📁 **`scripts/`**: Modular code for geocoding, prediction, optimization, and UI components.
- 📄 **`main.py`**: Main Streamlit app.
- 📄 **`requirements.txt`**: List of Python packages required to run the app locally.
- 📄 **`README.md`**: This file, providing an overview of the project.

---

## 3. How It Works

1. 🔎 **User selects a location** using the address search bar (powered by OpenStreetMap Nominatim).
2. 🧮 **GIS queries** determine local drilling restrictions and depth limitations via Zürich's public maps.
3. 🧠 **Features are engineered** for the ML model based on nearby boreholes, elevation, and legal depth.
4. ⚡ **Predicted yield is returned**, with optional unit conversions and performance benchmarking for the input project settings.
5. 📈 **Optional optimization** proposes a configuration for maximizing system efficiency.
6. 💸 **Estimated subsidies** are shown based on the predicted heat yield and bonus eligibility.

---

## 4. Data Sources

GeoWatt ZH is built entirely on publicly available datasets and services, including:

- 🗺️ **Wärmenutzungsatlas Zürich (GIS Portal)** – depth limitations, restrictions, and borehole data.
- 🌐 **OpenStreetMap Nominatim API** – address and coordinate conversion.

---

## 5. Running the App Locally

To launch the app locally:

```bash
git clone https://github.com/leinadher/GeoWattZH.git
cd GeoWattZH
pip install -r requirements.txt
streamlit run app.py
