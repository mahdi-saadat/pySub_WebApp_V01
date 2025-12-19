# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 20:03:04 2025
Subsidence Assessment Tool with Contour Customization (Vertical Displacement)
@author: Mahdi Saadat
"""
import streamlit as st
from collections import defaultdict
import numpy as np
from scipy.special import erf
import pandas as pd
import os
import ezdxf
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm

# -------------------------------------------------
# Page config (ONLY ONCE)
# -------------------------------------------------
st.set_page_config(
    page_title="pySub – Subsidence Assessment",
    layout="wide"  # changed to wide for side-by-side layout
)

# -------------------------------------------------
# Initialize session state
# -------------------------------------------------
if 'vertical_displacement_data' not in st.session_state:
    st.session_state.vertical_displacement_data = None
if 'contour_manual' not in st.session_state:
    st.session_state.contour_manual = False
if 'contour_interval' not in st.session_state:
    st.session_state.contour_interval = 0.25
if 'contour_min' not in st.session_state:
    st.session_state.contour_min = -2.5
if 'contour_max' not in st.session_state:
    st.session_state.contour_max = 0.0

# -------------------------------------------------
# Basic UI
# -------------------------------------------------
st.title("pySub – Subsidence Assessment Tool")
st.caption("Numerical subsidence prediction for longwall mining")
st.success("Modules imported successfully")

# -------------------------------------------------
# Inputs
# -------------------------------------------------
with st.sidebar:
    st.header("Panel & Geotechnical Parameters")
    panel_width = st.number_input("Panel width (m)", 50.0, 1000.0, 500.0)
    panel_length = st.number_input("Panel length (m)", 50.0, 5000.0, 3000.0)
    depth_of_cover_input = st.number_input("Depth of cover (m)", 50.0, 1000.0, 250.0)
    extraction_thickness = st.number_input("Extraction thickness (m)", 1.0, 10.0, 4.20)
    lw_azimuth_angle = st.number_input("Longwall Azimuth (deg)", 0.0, 90.0, 90.0)
    percentage_hard_rock = st.number_input("Hard Rock (%)", 10.0, 100.0, 30.0)
    st.markdown("---")
    run_model = st.button("▶ Run Subsidence Assessment")

# -------------------------------------------------
# Helper functions (same as original)
# -------------------------------------------------

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
    available_ratios = df.index.tolist()
    closest_ratio = min(available_ratios, key=lambda x: abs(x - calculated_ratio))
    available_cols = [float(c) for c in df.columns]
    closest_col = min(available_cols, key=lambda x: abs(x - hard_rock_percentage))
    return df.at[closest_ratio, closest_col]

# Depth of cover setup
data = [{"Panel ID": 1, "Panel ID LW": "LW03", "Start": depth_of_cover_input, "End": depth_of_cover_input, "Seam": "Seam A"}]
df_doc = pd.DataFrame(data)

grid_point = 10
gradient_dict = {}
def calculate_gradient(start, end, num_points=grid_point):
    return np.linspace(start, end, num_points)

for index, row in df_doc.iterrows():
    lw_id = row['Panel ID']
    gradient = calculate_gradient(row['Start'], row['End'])
    gradient_dict[lw_id] = gradient

inflection_points_dict = {}
for ip, igrad in enumerate(gradient_dict.values()):
    lw_id = 1
    inflection_points_list = []
    current_panel_id = ip + 1
    current_row = df_doc[df_doc['Panel ID'] == current_panel_id]
    if not current_row.empty:
        mystart = current_row['Start'].values[0]
        myend = current_row['End'].values[0]
    else:
        continue
    avg_doc = (mystart + myend) / 2
    i_width = panel_width
    w_h_ratio = round(i_width / avg_doc, 1)
    if w_h_ratio >= 1.2:
        inf_point = igrad * 0.2
    else:
        inf_point = np.round(
            igrad * (-2.1702 * (w_h_ratio**4) + 7.2849 * (w_h_ratio**3) - 9.1824 * (w_h_ratio**2) + 5.3794 * w_h_ratio - 1.1308), 
            3
        )
    inflection_points_list.append(inf_point)
    inflection_points_dict[lw_id] = inflection_points_list

beta_angle_dict = {}
major_influence_radius_dict = {}
m_to_ft = 3.28084
ft_to_m = 0.3048
doc_counter = 0

for panel_id, gradient in gradient_dict.items():
    major_influence_radius_list = []
    gradient_ft = gradient * m_to_ft
    beta_angle = 58.89 + 0.03089 * gradient_ft - 0.0000184 * (gradient_ft ** 2)
    beta_angle_radians = np.radians(beta_angle)
    beta_angle_dict[doc_counter] = beta_angle
    major_influence_radius = np.round(gradient / np.tan(beta_angle_radians), 2)
    major_influence_radius_list.append(major_influence_radius)
    major_influence_radius_dict[panel_id] = major_influence_radius_list
    doc_counter += 1

# -------------------------------------------------
# Subsidence calculation functions (only vertical displacement used for caching)
# -------------------------------------------------
global_resolution = 100

def calculate_subsidence(lw_panel_id, panel_width, panel_length, extraction_thick, percentage_hard_rock, depth_of_cover, grid_resolution=100):
    my_panel_id = lw_panel_id
    myrow = df_doc[df_doc['Panel ID'] == my_panel_id]
    if not myrow.empty:
        mystart = myrow['Start'].values[0]
        myend = myrow['End'].values[0]
    else:
        raise ValueError(f"Panel ID {my_panel_id} not found.")
    average_depth_of_cover = (mystart + myend) / 2
    x_buffer = 100
    y_buffer = 100
    x_values_limit = np.linspace(0 - x_buffer, panel_length + x_buffer, grid_resolution)
    y_values_limit = np.linspace(0 - y_buffer, panel_width + y_buffer, grid_resolution)
    w_h_rat = round(panel_width / average_depth_of_cover, 1)
    hr_percentage = percentage_hard_rock / 100
    subsidence_factor = get_subsidence_factor(w_h_rat, hr_percentage)
    s_max = round(extraction_thick * subsidence_factor, 1)
    X, Y = np.meshgrid(x_values_limit, y_values_limit)
    Sxy = np.zeros_like(X)
    inflection_point_array = np.interp(
        np.arange(grid_resolution),
        np.linspace(0, grid_resolution - 1, len(inflection_points_dict[1][0])),
        inflection_points_dict[1][0]
    )
    major_influence_radius_array = np.interp(
        np.arange(grid_resolution),
        np.linspace(0, grid_resolution - 1, len(major_influence_radius_dict[1][0])),
        major_influence_radius_dict[1][0]
    )
    for i, x in enumerate(x_values_limit):
        c = inflection_point_array[i]
        R = major_influence_radius_array[i]
        for j, y in enumerate(y_values_limit):
            Sxy[i, j] = -s_max * (
                0.5 * (erf(np.sqrt(np.pi) * (c - y) / R) +
                       erf(np.sqrt(np.pi) * (-panel_width + c + y) / R))
            ) * (
                0.5 * (erf(np.sqrt(np.pi) * (c - x) / R) +
                       erf(np.sqrt(np.pi) * (-panel_length + c + x) / R))
            )
    return X, Y, Sxy

# Horizontal displacement / strain / tilt functions remain (used as-is for other plots)
horizontal_strain_coeff = 0.15

def calculate_horizontal_displacement(...):  # (as in original)
    ...

def calculate_horizontal_strain(...):  # (as in original)
    ...

def calculate_tilt(...):  # (as in original)
    ...

# -------------------------------------------------
# Rotation function
# -------------------------------------------------
def rotate_point(point, angle_degrees, center):
    angle_radians = np.radians(angle_degrees)
    x_translated = point[0] - center[0]
    y_translated = point[1] - center[1]
    x_rotated = x_translated * np.cos(angle_radians) - y_translated * np.sin(angle_radians)
    y_rotated = x_translated * np.sin(angle_radians) + y_translated * np.cos(angle_radians)
    return (x_rotated + center[0], y_rotated + center[1])

# -------------------------------------------------
# Plotting functions (only vertical displacement modified for dynamic contours)
# -------------------------------------------------
cmap_method = 'gist_rainbow'
contour_transparancy = 0.85
all_panel_min_x = [0]
all_panel_min_y = [0]

def plot_vertical_displacement_dynamic(X, Y, Sxy, all_panel_min_x, all_panel_min_y, lw_azimuth_angle,
                                       interval, vmin, vmax):
    panel_min_x = all_panel_min_x[0]
    panel_min_y = all_panel_min_y[0]
    X_shifted = X + panel_min_x
    Y_shifted = Y + panel_min_y
    rotation_center = (panel_min_x, panel_min_y)
    lw_rotation_angle = 90.0 - lw_azimuth_angle
    rotated_coords = [
        rotate_point((x, y), lw_rotation_angle, rotation_center)
        for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())
    ]
    rotated_X = np.array([c[0] for c in rotated_coords]).reshape(X.shape)
    rotated_Y = np.array([c[1] for c in rotated_coords]).reshape(Y.shape)

    levels = np.arange(
        np.floor(vmin / interval) * interval,
        np.ceil(vmax / interval) * interval + interval,
        interval
    )
    tick_positions = levels[::max(1, len(levels) // 10)]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap(cmap_method)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    contour = ax.contourf(
        rotated_X, rotated_Y, Sxy.T,
        levels=levels, cmap=cmap, norm=norm,
        alpha=contour_transparancy, extend='both'
    )
    ax.contour(rotated_X, rotated_Y, Sxy.T, levels=levels, colors="k", linewidths=0.2, alpha=0.5)

    cbar = plt.colorbar(contour, ticks=tick_positions)
    cbar.set_label('Vertical Displacement [m]', fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')
    ax.set_xlim(-100, panel_length + 100)
    ax.set_ylim(-100, panel_width + 100)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    return fig

# Other plot functions (unchanged)
def plot_vertical_displacement_3D(all_panels_data, all_panel_min_x, all_panel_min_y):
    # (as in original)
    ...

def plot_horizontal_displacement(...): ...
def plot_horizontal_strain(...): ...
def plot_tilt(...): ...

# -------------------------------------------------
# Run model
# -------------------------------------------------
if run_model:
    with st.spinner("Running subsidence model..."):
        try:
            # Compute vertical displacement once and cache
            X, Y, Sxy = calculate_subsidence(
                lw_panel_id=1,
                panel_width=panel_width,
                panel_length=panel_length,
                extraction_thick=extraction_thickness,
                percentage_hard_rock=percentage_hard_rock,
                depth_of_cover=depth_of_cover_input
            )
            auto_min = float(np.floor(Sxy.min() * 10) / 10)
            auto_max = float(np.ceil(Sxy.max() * 10) / 10)
            st.session_state.vertical_displacement_data = {"X": X, "Y": Y, "Sxy": Sxy}
            st.session_state.contour_min = auto_min
            st.session_state.contour_max = auto_max

            # 3D Plot
            st.subheader("3D Subsidence Surface")
            all_panels_data = [(X, Y, Sxy)]
            fig_3d = plot_vertical_displacement_3D(all_panels_data, all_panel_min_x, all_panel_min_y)
            st.pyplot(fig_3d, use_container_width=True)

            # Surface Response Parameters
            st.subheader("Surface Response Parameters")
            col1, col2 = st.columns([3, 1])  # plot | control
            col3, col4 = st.columns(2)

            # --- Vertical Displacement with Right Control ---
            with col2:
                st.markdown("**Contour Settings**")
                manual = st.checkbox("Manual limits", value=False)
                if manual:
                    interval = st.number_input(
                        "Interval [m]", min_value=0.01, max_value=1.0,
                        value=st.session_state.contour_interval, step=0.05, format="%.2f"
                    )
                    cmin = st.number_input("Min [m]", value=auto_min, step=0.1, format="%.2f")
                    cmax = st.number_input("Max [m]", value=auto_max, step=0.1, format="%.2f")
                    st.session_state.contour_interval = interval
                    st.session_state.contour_min = cmin
                    st.session_state.contour_max = cmax
                    st.session_state.contour_manual = True
                else:
                    st.session_state.contour_manual = False
                    st.session_state.contour_interval = 0.25
                    st.session_state.contour_min = auto_min
                    st.session_state.contour_max = auto_max

            with col1:
                data = st.session_state.vertical_displacement_data
                if st.session_state.contour_manual:
                    vmin = st.session_state.contour_min
                    vmax = st.session_state.contour_max
                    interval = st.session_state.contour_interval
                else:
                    vmin = auto_min
                    vmax = auto_max
                    interval = 0.25
                fig = plot_vertical_displacement_dynamic(
                    data["X"], data["Y"], data["Sxy"],
                    all_panel_min_x, all_panel_min_y, lw_azimuth_angle,
                    interval, vmin, vmax
                )
                st.pyplot(fig, use_container_width=True)

            # --- Other Plots (unchanged) ---
            with col3:
                all_panels_data = []
                Xh, Yh, Sh = calculate_horizontal_displacement(
                    lw_panel_id=1,
                    panel_width=panel_width,
                    panel_length=panel_length,
                    extraction_thick=extraction_thickness,
                    percentage_hard_rock=percentage_hard_rock,
                    depth_of_cover=depth_of_cover_input
                )
                all_panels_data.append((Xh, Yh, Sh))
                st.markdown("**Horizontal Displacement**")
                fig = plot_horizontal_displacement(all_panels_data, all_panel_min_x, all_panel_min_y)
                st.pyplot(fig, use_container_width=True)

            with col4:
                all_panels_data = []
                Xt, Yt, St = calculate_tilt(
                    lw_panel_id=1,
                    panel_width=panel_width,
                    panel_length=panel_length,
                    extraction_thick=extraction_thickness,
                    percentage_hard_rock=percentage_hard_rock,
                    depth_of_cover=depth_of_cover_input
                )
                all_panels_data.append((Xt, Yt, St))
                st.markdown("**Tilt**")
                fig = plot_tilt(all_panels_data, all_panel_min_x, all_panel_min_y)
                st.pyplot(fig, use_container_width=True)

            # Horizontal Strain (add in new row if needed)
            st.markdown("**Horizontal Strain**")
            Xs, Ys, Ss = calculate_horizontal_strain(
                lw_panel_id=1,
                panel_width=panel_width,
                panel_length=panel_length,
                extraction_thick=extraction_thickness,
                percentage_hard_rock=percentage_hard_rock,
                depth_of_cover=depth_of_cover_input
            )
            fig = plot_horizontal_strain([(Xs, Ys, Ss)], all_panel_min_x, all_panel_min_y)
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Click **▶ Run Subsidence Assessment** to start.")