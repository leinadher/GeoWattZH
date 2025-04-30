import streamlit as st
import io
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

def load_svg_icon(path: str) -> str:
    with open(path, "r") as file:
        return file.read()

def show_performance_comparison(pred_kw, depth, sondenzahl, zh_geothermal_probes_gdf):
    if depth is None or sondenzahl is None:
        st.warning("Missing input values for comparison.")
        return

    similar_sites = zh_geothermal_probes_gdf[
        (zh_geothermal_probes_gdf["Sondentiefe"].between(depth - 20, depth + 20)) &
        (zh_geothermal_probes_gdf["Gesamtsondenzahl"] == sondenzahl)
    ].copy()

    if similar_sites.empty:
        st.info("No comparable boreholes found in canton records.")
        return

    percentile = percentileofscore(similar_sites["Waermeentnahme"], pred_kw)
    st.markdown(f"""
    Your predicted yield of **{pred_kw:.1f} kW** outperforms **{percentile:.0f}%**
    of installations in Zürich with **{sondenzahl} probes** and **depths similar (±20 m) to {depth}  m**.
    """)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(similar_sites["Waermeentnahme"], bins=25, color="grey", edgecolor="white", alpha=0.7)
    ax.axvline(pred_kw, color=(91/255, 144/255, 247/255), linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Heat Yield (kW)")
    ax.set_ylabel("Number of Sites")
    ax.grid(True, color="#dddddd", linestyle="-", linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("")
    ax.legend().set_visible(False)

    svg_buffer = io.StringIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
    st.components.v1.html(f"<div style='text-align:center'>{svg_buffer.getvalue()}</div>", height=500)

### SYSTEM COMPARISON ###

def show_performance_comparison(pred_kw, depth, sondenzahl, zh_geothermal_probes_gdf):
    from scipy.stats import percentileofscore
    import matplotlib.pyplot as plt

    if depth is None or sondenzahl is None:
        st.warning("Missing input values for comparison.")
        return

    similar_sites = zh_geothermal_probes_gdf[
        (zh_geothermal_probes_gdf["Sondentiefe"].between(depth - 20, depth + 20)) &
        (zh_geothermal_probes_gdf["Gesamtsondenzahl"] == sondenzahl)
    ].copy()

    if similar_sites.empty:
        st.info("No comparable boreholes found in canton records.")
        return

    percentile = percentileofscore(similar_sites["Waermeentnahme"], pred_kw)

    st.markdown(f"""
    Your predicted yield of **{pred_kw:.1f} kW** outperforms **{percentile:.0f}%**
    of installations in Zürich with **{sondenzahl} probes** and **depths similar (±20 m) to {depth}  m**.
    """)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(similar_sites["Waermeentnahme"], bins=25, color="grey", edgecolor="white", alpha=0.7)
    ax.axvline(pred_kw, color=(91/255, 144/255, 247/255), linewidth=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Heat Yield (kW)")
    ax.set_ylabel("Number of Sites")
    ax.grid(True, color="#dddddd", linestyle="-", linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("")
    ax.legend().set_visible(False)
    
    # Save as SVG in memory and show
    svg_buffer = io.StringIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
    svg_data = svg_buffer.getvalue()
    plt.close(fig)
    st.components.v1.html(f"<div style='text-align:center'>{svg_data}</div>", height=500)
