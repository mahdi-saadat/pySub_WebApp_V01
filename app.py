# -*- coding: utf-8 -*-
"""
pySub – Subsidence Assessment Tool (Streamlit Web App)
Features:
1. Vertical displacement calculation
2. Interactive contour plot with user-controlled min, max, and interval
3. No re-run of calculations needed when adjusting contour display
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker

# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(page_title="pySub – Subsidence Assessment", layout="centered")
st.title("pySub – Subsidence Assessment Tool")
st.caption("Numerical subsidence prediction for longwall mining")
st.success("Modules imported successfully")

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
with st.sidebar:
    st.header("Panel & Geotechnical Parameters")
    panel_width = st.number_input("Panel width (m)", 50.0, 1000.0, 500.0)
    panel_length = st.number_input("Panel length (m)", 50.0, 5000.0, 3000.0)
    depth_of_cover_input = st.number_input("Depth of cover (m)", 50.0, 1000.0, 250.0)
    extraction_thickness = st.number_input("Extraction thickness (m)", 1.0, 10.0, 4.2)
    lw_azimuth_angle = st.number_input("Longwall Azimuth (deg)", 0.0, 90.0, 90.0)
    percentage_hard_rock = st.number_input("Hard Rock (%)", 10.0, 100.0, 30.0)
    st.markdown("---")
    run_model = st.button("▶ Run Subsidence Assessment")

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def rotate_point(point, angle_degrees, center):
    """Rotate a point counter-clockwise around a given center by a specified angle."""
    angle_radians = np.radians(angle_degrees)
    x_translated = point[0] - center[0]
    y_translated = point[1] - center[1]
    x_rotated = x_translated * np.cos(angle_radians) - y_translated * np.sin(angle_radians)
    y_rotated = x_translated * np.sin(angle_radians) + y_translated * np.cos(angle_radians)
    return (x_rotated + center[0], y_rotated + center[1])

def get_subsidence_factor(calculated_ratio, hard_rock_percentage):
    data = {
        'W/h': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        0.1: [0.64, 0.69, 0.71, 0.72, 0.73, 0.74, 0.74, 0.74, 0.75, 0.75, 0.75, 0.75, 0.75, 0.76, 0.76],
        0.2: [0.59, 0.63, 0.65, 0.66, 0.67, 0.68, 0.68, 0.68, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69],
        0.3: [0.51, 0.55, 0.57, 0.58, 0.58, 0.59, 0.59, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        0.4: [0.42, 0.46, 0.47, 0.48, 0.49, 0.49, 0.49, 0.49, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        0.5: [0.34, 0.36, 0.38, 0.38, 0.39, 0.39, 0.39, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
        0.6: [0.26, 0.28, 0.29, 0.30, 0.30, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31],
        0.7: [0.21, 0.22, 0.23, 0.23, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24],
        0.8: [0.16, 0.18, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19]
    }
    df = pd.DataFrame(data).set_index('W/h')
    closest_ratio = min(df.index, key=lambda x: abs(x - calculated_ratio))
    closest_col = min([float(c) for c in df.columns], key=lambda x: abs(x - hard_rock_percentage))
    return df.at[closest_ratio, closest_col]

def calculate_subsidence(panel_width, panel_length, extraction_thick, percentage_hard_rock, depth_of_cover, grid_resolution=100):
    avg_depth = depth_of_cover
    w_h_ratio = round(panel_width / avg_depth, 1)
    hr_fraction = percentage_hard_rock / 100
    subs_factor = get_subsidence_factor(w_h_ratio, hr_fraction)
    s_max = extraction_thick * subs_factor

    # Grid
    x_buffer, y_buffer = 100, 100
    x_values = np.linspace(-x_buffer, panel_length + x_buffer, grid_resolution)
    y_values = np.linspace(-y_buffer, panel_width + y_buffer, grid_resolution)
    X, Y = np.meshgrid(x_values, y_values)

    # Inflection point
    if w_h_ratio >= 1.2:
        inf_point = 0.2 * avg_depth
    else:
        inf_point = avg_depth * (-2.1702 * (w_h_ratio**4) + 7.2849 * (w_h_ratio**3) - 9.1824 * (w_h_ratio**2) + 5.3794 * w_h_ratio - 1.1308)

    R = panel_width / 2  # Simplified major influence radius
    Sxy = -s_max * (
        0.5 * (erf(np.sqrt(np.pi) * (inf_point - Y) / R) + erf(np.sqrt(np.pi) * (-panel_width + inf_point + Y) / R)) *
        0.5 * (erf(np.sqrt(np.pi) * (inf_point - X) / R) + erf(np.sqrt(np.pi) * (-panel_length + inf_point + X) / R))
    )
    return X, Y, Sxy

def plot_vertical_displacement(X, Y, Sxy, interval, user_min=None, user_max=None):
    # Contour levels
    panel_min = Sxy.min() if user_min is None else user_min
    panel_max = Sxy.max() if user_max is None else user_max
    levels = np.arange(panel_min, panel_max + interval, interval)
    cmap = plt.get_cmap('gist_rainbow')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    contour = ax.contourf(X, Y, Sxy.T, levels=levels, cmap=cmap, norm=norm, alpha=0.85)
    cbar = plt.colorbar(contour, ax=ax, ticks=levels[::max(1, int(len(levels)/10))])
    cbar.set_label('Vertical Displacement [m]', fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.1)
    ax.set_aspect('equal')
    return fig

# -------------------------------------------------
# Run model
# -------------------------------------------------
if run_model:
    st.info("Calculating subsidence...")
    X, Y, Sxy = calculate_subsidence(panel_width, panel_length, extraction_thickness, percentage_hard_rock, depth_of_cover_input)

    # Display automatic min, max
    auto_min, auto_max = Sxy.min(), Sxy.max()

    st.sidebar.markdown("---")
    st.sidebar.header("Contour Customization")
    user_set_limits = st.sidebar.checkbox("Enable manual contour limits", value=False)
    interval = st.sidebar.number_input("Contour interval [m]", min_value=0.01, max_value=1.0, value=0.25, step=0.01, format="%.2f")
    if user_set_limits:
        user_min = st.sidebar.number_input("Min displacement [m]", value=float(np.round(auto_min, 2)))
        user_max = st.sidebar.number_input("Max displacement [m]", value=float(np.round(auto_max, 2)))
    else:
        user_min, user_max = None, None

    fig = plot_vertical_displacement(X, Y, Sxy, interval, user_min, user_max)
    st.pyplot(fig)
