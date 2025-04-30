from scripts.yield_optimizer import optimize_yield
from scripts.ui_components import show_performance_comparison
from scripts.predict_energy import predict_energy_yield
import streamlit as st

@st.fragment
def show_yield_ui(features, zh_geothermal_probes_gdf):
    st.session_state.setdefault("selected_depth", int(features["depth_max"] / 2))
    st.session_state.setdefault("gesamtsondenzahl", 5)

    # Apply optimized values only before widgets are rendered
    if st.session_state.get("apply_optimization", False):
        st.session_state.selected_depth = st.session_state.optimized_depth
        st.session_state.gesamtsondenzahl = st.session_state.optimized_probes
        st.session_state.run_yield = True
        st.session_state.apply_optimization = False  # reset flag

    st.markdown("### 🔋 Heat Yield Estimation")

    st.slider(
        "Select probe depth (m)",
        min_value=10,
        max_value=int(features["depth_max"]),
        step=5,
        key="selected_depth"
    )

    st.slider(
        "Select number of probes",
        min_value=1,
        max_value=10,
        step=1,
        key="gesamtsondenzahl"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("⚡ Estimate Heat Yield"):
            st.session_state.run_yield = True

    with col2:
        if st.button("🎯 Optimize Configuration"):
            with st.spinner("⏳ Optimizing..."):
                result = optimize_yield(features)

                st.session_state.optimized_depth = result["depth"]
                st.session_state.optimized_probes = result["probes"]
                st.session_state.optimized_yield = result["yield"]

                # Set a flag to trigger update on rerun
                st.session_state.apply_optimization = True

                st.rerun()

    if st.session_state.get("run_yield", False):
        bottom_elevation = features.get("elevation") - st.session_state.selected_depth
        features_for_model = {
            "Gesamtsondenzahl": st.session_state.gesamtsondenzahl,
            "count_100m": features.get("count_100m"),
            "nearest_borehole_dist": features.get("nearest_borehole_dist"),
            "Sondentiefe": st.session_state.selected_depth,
            "bottom_elevation": bottom_elevation
        }

        with st.spinner("⏳ Processing result..."):
            prediction = predict_energy_yield(features_for_model)
            st.session_state.prediction = prediction
            
            # Save parameters from the prediction:
            st.session_state.prediction_sondenzahl = st.session_state.gesamtsondenzahl
            st.session_state.prediction_depth = st.session_state.selected_depth

            # Reset yield process to avoid reloading UI
            st.session_state.run_yield = False

    if "prediction" in st.session_state and st.session_state.prediction is not None:
        conversion_option = st.selectbox(
            "Select unit for estimated yield:",
            options=[
                "Instantaneous Power (kW)",
                "Daily Energy (kWh/day)",
                "Annual Energy (kWh/year)",
                "Annual Energy (MWh/year)"
            ],
            index=0
        )

        pred_kw = st.session_state.get("prediction", 0)
        full_load_hours = 2000
        if conversion_option == "Instantaneous Power (kW)":
            converted = f"{pred_kw:,.1f} kW"
        elif conversion_option == "Daily Energy (kWh/day)":
            converted = f"{pred_kw * 24:,.1f} kWh"
        elif conversion_option == "Annual Energy (kWh/year)":
            converted = f"{pred_kw * full_load_hours:,.0f} kWh"
        else:
            converted = f"{(pred_kw * full_load_hours) / 1000:,.1f} MWh"

        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Yield", value=converted)
        col2.metric(label="Depth", value=f"{st.session_state.get('prediction_depth', 0)} m")
        col3.metric(label="# Probes", value=st.session_state.get("prediction_sondenzahl", 0))

        with st.expander("📊 Performance Comparison", expanded=False):
            show_performance_comparison(
                pred_kw=st.session_state.prediction,
                depth=st.session_state.prediction_depth,
                sondenzahl=st.session_state.prediction_sondenzahl,
                zh_geothermal_probes_gdf=zh_geothermal_probes_gdf
            )

        if "show_financial" not in st.session_state:
            st.session_state.show_financial = False

        with st.expander("💰 Financial Incentives", expanded=False):
            st.markdown(
                "Based on the [2025 Förderprogramm](https://www.zh.ch/de/umwelt-tiere/energie/energiefoerderung.html) "
                "for Erdwärmesonden:"
            )

            pred_kw = st.session_state.prediction

            # Base subsidy calculation
            if pred_kw <= 15:
                base_subsidy = 6800
            else:
                base_subsidy = 6800 + 420 * (pred_kw - 15)
            base_subsidy = round(base_subsidy)

            # Optional bonus
            capped_kw = min(pred_kw, 70)
            bonus_subsidy = 3000 + 100 * capped_kw
            bonus_subsidy = round(bonus_subsidy)

            total = base_subsidy + bonus_subsidy

            # Display nicely
            st.markdown(f"""
            - **Base subsidy:** CHF **{base_subsidy:,}**
            - **Optional bonus** (frost-free regeneration system): CHF **{bonus_subsidy:,}**
            - **Estimated total** (if eligible for bonus): CHF **{total:,}**

            _Exact amount subject to approval. Bonus requires additional technical components._
            """)