# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 20:03:04 2025
[Description of the module or script]

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
# Page config (ONLY ONCE, FIRST THING)
# -------------------------------------------------
st.set_page_config(
    page_title="pySub – Subsidence Assessment",
    layout="centered"
)

# -------------------------------------------------
# Basic UI (render immediately)
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
    #------------------------------------------------------
    st.markdown("---")
    with st.expander("📊 Vertical Displacement Contour Customization"):
        use_manual_limits = st.checkbox("Set custom contour levels", value=False)
        interval_input = st.number_input("Contour interval [m]", min_value=0.01, max_value=2.0, value=0.25, step=0.05, format="%.2f")
        # Placeholder values; actual defaults will come from data during plotting
        panel_min_input = st.number_input("Min displacement [m]", value=0.0, step=0.1, format="%.2f")
        panel_max_input = st.number_input("Max displacement [m]", value=0.0, step=0.1, format="%.2f")

#----------------------------------------------------------------- Core Subsidence Calculations

def get_subsidence_factor(calculated_ratio, hard_rock_percentage):
    """
    Calculates W/h and retrieves the corresponding value from the table.
    
    Parameters:
    W (float): Width value
    h (float): Height value
    top_row_val (float): The column header to look up (0.1, 0.2, etc.)
    """
    
    # 1. Define the table data
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

    # 2. Find the closest W/h index in the table
    # This prevents errors if W/h is something like 0.92
    available_ratios = df.index.tolist()
    closest_ratio = min(available_ratios, key=lambda x: abs(x - calculated_ratio))

    # 3. Find the closest column header
    available_cols = [float(c) for c in df.columns]
    closest_col = min(available_cols, key=lambda x: abs(x - hard_rock_percentage))

    # 4. Return the value
    return df.at[closest_ratio, closest_col]

#--------------------- Depth of Cover ----------------

"""
    This depth of cover can be modified later
"""
# Your initial data
data = [
    {
        "Panel ID": 1,
        "Panel ID LW": "LW03",
        "Start": depth_of_cover_input,
        "End": depth_of_cover_input,
        "Seam": "Seam A"
    }
]

df_doc = pd.DataFrame(data)
#--------------------- End of Depth of Cover ----------------


# Define your grid points
grid_point = 10 

# Initialize your storage dictionary
gradient_dict = {}

def calculate_gradient(start, end, num_points=grid_point):
    return np.linspace(start, end, num_points)


# Populate the dictionary with gradients
for index, row in df_doc.iterrows():
    lw_id = row['Panel ID']
    gradient = calculate_gradient(row['Start'], row['End'])
    gradient_dict[lw_id] = gradient

# Dictionary to store inflection points with Panel ID as the key
inflection_points_dict = {}
#for ip, (lw_id, igrad) in zip(range(len(all_panel_widths)), gradient_dict.items()):
for ip, igrad in enumerate(gradient_dict.values()):
    lw_id = 1
    inflection_points_list = []
    current_panel_id = ip + 1
    current_row = df_doc[df_doc['Panel ID'] == current_panel_id]
    
    if not current_row.empty:
        mystart = current_row['Start'].values[0]
        myend = current_row['End'].values[0]   
    else:
        #print(f"Panel ID {current_panel_id} not found in the dataframe.")
        continue  # Skip this panel if it isn't found
    
    avg_doc = (mystart + myend) / 2
    
    i_width = panel_width
    w_h_ratio = round(i_width / avg_doc, 1)
    #print(f"Panel width: {i_width} , Average DOC: {avg_doc}, W/H Ratio: {w_h_ratio}, Panel_id: {lw_id}, Panel_id: {ip}")
    if w_h_ratio >= 1.2:
        
        inf_point = igrad * 0.2
    else:
        inf_point = np.round(
            igrad * (-2.1702 * (w_h_ratio**4) + 7.2849 * (w_h_ratio**3) - 9.1824 * (w_h_ratio**2) + 5.3794 * w_h_ratio - 1.1308), 
            3
        )
    # Append inf_point to the list
    inflection_points_list.append(inf_point)
    # Store the list in the dictionary
    inflection_points_dict[lw_id] = inflection_points_list
    
    print(f"W/h : {w_h_ratio}, Inflection Point: {inflection_points_list}")

beta_angle_dict = {}
major_influence_radius_dict = {}
m_to_ft = 3.28084  # meters to feet
ft_to_m = 0.3048   # feet to meters

doc_counter = 0
# Calculate beta_angle and major influence radius for each depth of cover
for panel_id, gradient in gradient_dict.items():
    
    major_influence_radius_list = []  # Initialize a new list for each panel ID
    # Calculate beta_angle (angle of major influence) in degrees
    gradient_ft = gradient * m_to_ft

    beta_angle = 58.89 + 0.03089 * gradient_ft - 0.0000184 * (gradient_ft ** 2)

    # Convert beta_angle from degrees to radians
    beta_angle_radians = np.radians(beta_angle)
    
    # Store beta_angle value in the dictionary
    beta_angle_dict[doc_counter] = beta_angle
    
    # Calculate major influence radius
    major_influence_radius = np.round(gradient / np.tan(beta_angle_radians), 2)
    major_influence_radius_list.append(major_influence_radius)
    # Store major influence radius in the dictionary
    major_influence_radius_dict[panel_id] = major_influence_radius_list
    doc_counter +=1

# --------------------------------------------------------------------------------------------------
# Create Subsidence parameters calculation functions
# --------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                        # Vertical Dsipalcement
def calculate_subsidence(lw_panel_id, panel_width, panel_length, extraction_thick, percentage_hard_rock, depth_of_cover,grid_resolution=100):
    global my_panel_id
    my_panel_id = lw_panel_id
    myrow = df_doc[df_doc['Panel ID'] == my_panel_id]
    
    #If the row exists, extract Start and End values
    if not myrow.empty:
        mystart = myrow['Start'].values[0]
        myend = myrow['End'].values[0]   
    else:
        print(f"Panel ID {my_panel_id} not found in the dataframe.")
    
    average_depth_of_cover = (mystart+myend)/2
    
    # Define buffers
    x_buffer = 100#0.85 * panel_length
    y_buffer = 100#1.5 * panel_width
    
    # Define x and y ranges
    global x_values_limit
    global y_values_limit
    x_values_limit = np.linspace(0 - x_buffer, panel_length + x_buffer, grid_resolution)
    y_values_limit = np.linspace(0 - y_buffer, panel_width + y_buffer, grid_resolution)
    

    w_h_rat = round(panel_width / average_depth_of_cover, 1)
    hr_percentage = percentage_hard_rock/100
    subsidence_factor = get_subsidence_factor(w_h_rat,hr_percentage)
    
    # Calculate Smax, Maximum Subsidence [m]
    s_max = round(extraction_thick * subsidence_factor, 1)
    X, Y = np.meshgrid(x_values_limit, y_values_limit)
    Sxy = np.zeros_like(X)
    
    #inflection_point_list = inflection_points_dict[lw_panel_id]
    major_influence_radius_array = major_influence_radius_dict[lw_panel_id]
    
    # Convert inflection points to match grid resolution
    inflection_point_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(inflection_points_dict[1][0])), 
        inflection_points_dict[1][0]
    )
    
    major_influence_radius_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(major_influence_radius_dict[1][0])), 
        major_influence_radius_dict[1][0]
    )

    
    # Iterate over x and y values
    for i, x in enumerate(x_values_limit):
        inflection_point_to_edge_conservative = inflection_point_array[i]
        major_influence_radius = major_influence_radius_array[i]
        for j, y in enumerate(y_values_limit):
            
            x = x_values_limit[i]
            y = y_values_limit[j]
            
            Sxy[i, j] = -s_max * (
                0.5 * (erf(np.sqrt(np.pi) * (inflection_point_to_edge_conservative - y) / major_influence_radius) +
                       erf(np.sqrt(np.pi) * (-panel_width + inflection_point_to_edge_conservative + y) / major_influence_radius))
            ) * (
                0.5 * (erf(np.sqrt(np.pi) * (inflection_point_to_edge_conservative - x) / major_influence_radius) +
                       erf(np.sqrt(np.pi) * (-panel_length + inflection_point_to_edge_conservative + x) / major_influence_radius))
            )
            
    return X, Y, Sxy

#----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                        # Horizontal Dsipalcement


horizontal_strain_coeff = 0.15

def calculate_horizontal_displacement(lw_panel_id, panel_width, panel_length, extraction_thick, percentage_hard_rock, depth_of_cover,grid_resolution=100):
    global my_panel_id
    my_panel_id = lw_panel_id
    myrow = df_doc[df_doc['Panel ID'] == my_panel_id]
    
    #If the row exists, extract Start and End values
    if not myrow.empty:
        mystart = myrow['Start'].values[0]
        myend = myrow['End'].values[0]   
    else:
        print(f"Panel ID {my_panel_id} not found in the dataframe.")
    
    average_depth_of_cover = (mystart+myend)/2
    
    # Define buffers
    x_buffer = 100#0.85 * panel_length
    y_buffer = 100#1.5 * panel_width
    
    # Define x and y ranges
    global x_values_limit
    global y_values_limit
    x_values_limit = np.linspace(0 - x_buffer, panel_length + x_buffer, grid_resolution)
    y_values_limit = np.linspace(0 - y_buffer, panel_width + y_buffer, grid_resolution)
    

    w_h_rat = round(panel_width / average_depth_of_cover, 1)
    hr_percentage = percentage_hard_rock/100
    subsidence_factor = get_subsidence_factor(w_h_rat,hr_percentage)
    
    # Calculate Smax, Maximum Subsidence [m]
    s_max = round(extraction_thick * subsidence_factor, 1)
    X, Y = np.meshgrid(x_values_limit, y_values_limit)
    Uxy = np.zeros_like(X)  # horizontal displacement along x
    Vxy = np.zeros_like(Y)  # horizontal displacement along y
    Sxy = np.zeros_like(X)
    
    #inflection_point_list = inflection_points_dict[lw_panel_id]
    major_influence_radius_array = major_influence_radius_dict[lw_panel_id]
    
    # Convert inflection points to match grid resolution
    inflection_point_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(inflection_points_dict[1][0])), 
        inflection_points_dict[1][0]
    )
    
    major_influence_radius_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(major_influence_radius_dict[1][0])), 
        major_influence_radius_dict[1][0]
    )

    # Loop over x and y
    for i, x in enumerate(x_values_limit):
        inflection_point_to_edge = inflection_point_array[i]
        major_influence_radius = major_influence_radius_array[i]
        horizontal_strain_factor = horizontal_strain_coeff * major_influence_radius  # can include negative sign below
        
        for j, y in enumerate(y_values_limit):
            c = inflection_point_to_edge
            R = major_influence_radius
            W = panel_width
            L = panel_length
            
            # Smoothing factors
            A = 0.5 * (erf(np.sqrt(np.pi)*(c - y)/R) + erf(np.sqrt(np.pi)*(-W + c + y)/R))
            B = 0.5 * (erf(np.sqrt(np.pi)*(c - x)/R) + erf(np.sqrt(np.pi)*(-L + c + x)/R))
            
            # Partial derivatives (tilts)
            dBdx = (s_max / R) * (np.exp(-np.pi*(c - x)**2 / R**2) - np.exp(-np.pi*(-L + c + x)**2 / R**2))
            dAdy = (s_max / R) * (np.exp(-np.pi*(c - y)**2 / R**2) - np.exp(-np.pi*(-W + c + y)**2 / R**2))
            
            # Horizontal displacement components (include negative sign for compression inside panel)
            Uxy[i,j] = - horizontal_strain_factor * (A * dBdx)
            Vxy[i,j] = - horizontal_strain_factor * (B * dAdy)
    
    # Horizontal displacement magnitude for contour plotting
    Sxy = np.sqrt(Uxy**2 + Vxy**2)
    return X, Y, Sxy

#----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                        # Horizontal strain
                                                        
def calculate_horizontal_strain(lw_panel_id, panel_width, panel_length, extraction_thick, percentage_hard_rock, depth_of_cover,grid_resolution=100):
    global my_panel_id
    my_panel_id = lw_panel_id
    myrow = df_doc[df_doc['Panel ID'] == my_panel_id]
    
    #If the row exists, extract Start and End values
    if not myrow.empty:
        mystart = myrow['Start'].values[0]
        myend = myrow['End'].values[0]   
    else:
        print(f"Panel ID {my_panel_id} not found in the dataframe.")
    
    average_depth_of_cover = (mystart+myend)/2
    
    # Define buffers
    x_buffer = 100#0.85 * panel_length
    y_buffer = 100#1.5 * panel_width
    
    # Define x and y ranges
    global x_values_limit
    global y_values_limit
    x_values_limit = np.linspace(0 - x_buffer, panel_length + x_buffer, grid_resolution)
    y_values_limit = np.linspace(0 - y_buffer, panel_width + y_buffer, grid_resolution)
    

    w_h_rat = round(panel_width / average_depth_of_cover, 1)
    hr_percentage = percentage_hard_rock/100
    subsidence_factor = get_subsidence_factor(w_h_rat,hr_percentage)
    
    # Calculate Smax, Maximum Subsidence [m]
    s_max = round(extraction_thick * subsidence_factor, 1)
    X, Y = np.meshgrid(x_values_limit, y_values_limit)
    Uxy = np.zeros_like(X)  # horizontal displacement along x
    Vxy = np.zeros_like(Y)  # horizontal displacement along y
    Sxy = np.zeros_like(X)
    
    #inflection_point_list = inflection_points_dict[lw_panel_id]
    major_influence_radius_array = major_influence_radius_dict[lw_panel_id]
    
    # Convert inflection points to match grid resolution
    inflection_point_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(inflection_points_dict[1][0])), 
        inflection_points_dict[1][0]
    )
    
    major_influence_radius_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(major_influence_radius_dict[1][0])), 
        major_influence_radius_dict[1][0]
    )


    # Compute horizontal displacement field
    for i, x in enumerate(x_values_limit):
        inflection_point_to_edge_conservative = inflection_point_array[i]
        major_influence_radius = major_influence_radius_array[i]
        horizontal_strain_factor = horizontal_strain_coeff * major_influence_radius
        for j, y in enumerate(y_values_limit):
            c = inflection_point_to_edge_conservative
            R = major_influence_radius
            W = panel_width
            L = panel_length
            
            A = 0.5 * (erf(np.sqrt(np.pi)*(c - y)/R) + erf(np.sqrt(np.pi)*(-W + c + y)/R))
            B = 0.5 * (erf(np.sqrt(np.pi)*(c - x)/R) + erf(np.sqrt(np.pi)*(-L + c + x)/R))
            
            dBdx = (s_max / R) * (np.exp(-np.pi*(c - x)**2 / R**2) - np.exp(-np.pi*(-L + c + x)**2 / R**2))
            dAdy = (s_max / R) * (np.exp(-np.pi*(c - y)**2 / R**2) - np.exp(-np.pi*(-W + c + y)**2 / R**2))
            
            # Signed vector components
            Uxy[i,j] = -horizontal_strain_factor * (A * dBdx)  # x-component
            Vxy[i,j] = -horizontal_strain_factor * (B * dAdy)  # y-component
    
    # Define grid spacing
    dx = x_values_limit[1] - x_values_limit[0]
    dy = y_values_limit[1] - y_values_limit[0]
    
    # Compute 2D strain components
    ex = np.gradient(Uxy, dx, axis=0) * 1e3     # du/dx
    ey = np.gradient(Vxy, dy, axis=1) * 1e3     # dv/dy
    
    
    mean = 0.5 * (ex + ey)
    
    # """
    # #-----------------------
    # # Calculate other Strain Parameters
    # #----------------------
    # #========== Other Strain Parameters [Plot if Needed]
    # # exy_shear = 0.5 * (np.gradient(Uxy, dy, axis=1) + np.gradient(Vxy, dx, axis=0))  # shear strain
    # # rad = np.sqrt(((ex - ey) * 0.5)**2 + exy_shear**2)
    # # e1 = mean + rad   # major principal strain
    # # e2 = mean - rad   # minor principal strain
    # # effective_strain = np.sqrt(ex**2 + ey**2 + exy_shear**2)
    # # max_shear_strain = rad
    # # exy = 0.5 * (np.gradient(Uxy, dy, axis=1) + np.gradient(Vxy, dx, axis=0)) * 1e3  # Shear strain (epsilon_xy)
    # #==========
    # """
    Sxy = mean

    return X, Y, Sxy

#----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                        # Tilt
def calculate_tilt(lw_panel_id, panel_width, panel_length, extraction_thick, percentage_hard_rock, depth_of_cover,grid_resolution=100):
    global my_panel_id
    my_panel_id = lw_panel_id
    myrow = df_doc[df_doc['Panel ID'] == my_panel_id]
    
    #If the row exists, extract Start and End values
    if not myrow.empty:
        mystart = myrow['Start'].values[0]
        myend = myrow['End'].values[0]   
    else:
        print(f"Panel ID {my_panel_id} not found in the dataframe.")
    
    average_depth_of_cover = (mystart+myend)/2
    
    # Define buffers
    x_buffer = 100#0.85 * panel_length
    y_buffer = 100#1.5 * panel_width
    
    # Define x and y ranges
    global x_values_limit
    global y_values_limit
    x_values_limit = np.linspace(0 - x_buffer, panel_length + x_buffer, grid_resolution)
    y_values_limit = np.linspace(0 - y_buffer, panel_width + y_buffer, grid_resolution)
    

    w_h_rat = round(panel_width / average_depth_of_cover, 1)
    hr_percentage = percentage_hard_rock/100
    subsidence_factor = get_subsidence_factor(w_h_rat,hr_percentage)
    
    # Calculate Smax, Maximum Subsidence [m]
    s_max = round(extraction_thick * subsidence_factor, 1)
    X, Y = np.meshgrid(x_values_limit, y_values_limit)
    Sxy = np.zeros_like(X)
    Tilt = np.zeros_like(X)
    #inflection_point_list = inflection_points_dict[lw_panel_id]
    major_influence_radius_array = major_influence_radius_dict[lw_panel_id]
    
    # Convert inflection points to match grid resolution
    inflection_point_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(inflection_points_dict[1][0])), 
        inflection_points_dict[1][0]
    )
    
    major_influence_radius_array = np.interp(
        np.arange(global_resolution), 
        np.linspace(0, global_resolution-1, len(major_influence_radius_dict[1][0])), 
        major_influence_radius_dict[1][0]
    )

    
    
    # Iterate over x and y values
    for i, x in enumerate(x_values_limit):
        inflection_point_to_edge_conservative = inflection_point_array[i]
        major_influence_radius = major_influence_radius_array[i]
        #print(f"major influence radius: {major_influence_radius}")
        #horizontal_strain_coefficient = strain_coefficient * major_influence_radius
        for j, y in enumerate(y_values_limit):
            
            # Inside your loops over x and y:
            c = inflection_point_to_edge_conservative
            R = major_influence_radius
            W = panel_width
            L = panel_length
            
            # 1D smoothing factors (sum of erf terms)
            A = 0.5 * (erf(np.sqrt(np.pi)*(c - y)/R) + erf(np.sqrt(np.pi)*(-W + c + y)/R))
            B = 0.5 * (erf(np.sqrt(np.pi)*(c - x)/R) + erf(np.sqrt(np.pi)*(-L + c + x)/R))
            
            # Partial derivatives (tilts along x and y)
            dBdx = (s_max / R) * (np.exp(-np.pi*(c - x)**2 / R**2) - np.exp(-np.pi*(-L + c + x)**2 / R**2))
            dAdy = (s_max / R) * (np.exp(-np.pi*(c - y)**2 / R**2) - np.exp(-np.pi*(-W + c + y)**2 / R**2))
            
            # 2D tilt magnitude (mm/m)
            Tilt[i, j] = np.sqrt((A * dBdx)**2 + (B * dAdy)**2) * 1000
    
    Sxy = Tilt
    
    return X, Y, Sxy


# Define global grid resolution
global_resolution = 100  # Adjust this for finer/coarser grids
all_panels_data = []

all_panel_widths = [panel_width]
all_panel_lengths = [panel_length]
for i in range(len(all_panel_widths)):
    X, Y, Sxy = calculate_subsidence(
        lw_panel_id=i+1,
        panel_width=panel_width,
        panel_length=panel_length,
        extraction_thick=extraction_thickness,
        percentage_hard_rock=percentage_hard_rock,
        depth_of_cover=depth_of_cover_input
    )
    all_panels_data.append((X, Y, Sxy))



#------------------------------------------------ Rotation ----------------------
def rotate_point(point, angle_degrees, center):
    """Rotate a point counter-clockwise around a given center by a specified angle."""
    angle_radians = np.radians(angle_degrees)
    # Translate point to origin
    x_translated = point[0] - center[0]
    y_translated = point[1] - center[1]
    
    # Perform the rotation (counter-clockwise)
    x_rotated = x_translated * np.cos(angle_radians) - y_translated * np.sin(angle_radians)
    y_rotated = x_translated * np.sin(angle_radians) + y_translated * np.cos(angle_radians)
    
    # Translate back to the original center
    return (x_rotated + center[0], y_rotated + center[1])

cmap_method = 'gist_rainbow'
contour_transparancy = 0.85

all_panel_min_x = [0]
all_panel_min_y = [0]


# -------------------------------------------------
# Create Plot Functions
# -------------------------------------------------


#======================================================================================================================================================
# def plot_vertical_displacement(all_panels_data, all_panel_min_x, all_panel_min_y):
    
#     # Define the center point for rotation
#     # *** FIX: Rotation center must be the bottom-left corner of the current panel's coordinate system. ***
#     # rotation_center = center#(619925.17, 7594941.26) # Commented out global placeholder

#     for i, panel_data in enumerate(all_panels_data):
#         X, Y, mySxy = panel_data
        
#         # Calculate min and max for each panel's subsidence
#         panel_min_subsidence = round(mySxy.min(),1)
#         panel_max_subsidence = round(mySxy.max(),1)
#         levels = np.linspace(panel_min_subsidence, panel_max_subsidence, 2000)
#         tick_positions = np.linspace(panel_min_subsidence, panel_max_subsidence, 10)

#         # Create a new figure for each panel
#         fig, ax = plt.subplots(figsize=(10, 5))

#         # Shift X and Y by panel's min_x and min_y (This creates the global coordinates)
#         X_shifted = X + all_panel_min_x[i]
#         Y_shifted = Y + all_panel_min_y[i]
        
#         # *** The fix: Set the rotation center to the bottom-left corner of the current panel in global coordinates ***
#         # This point is (all_panel_min_x[i], all_panel_min_y[i]) in the shifted system.
#         rotation_center = (all_panel_min_x[i], all_panel_min_y[i])

#         # Rotate all points by lw_rotation_angle degrees
#         # The rotation is now anchored correctly to the bottom-left corner of the panel in global coordinates.
#         lw_rotation_angle = 90.0 - lw_azimuth_angle
#         rotated_coords = [rotate_point((x, y), lw_rotation_angle, rotation_center) 
#                           for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())]
#         rotated_X = np.array([coord[0] for coord in rotated_coords]).reshape(X.shape)
#         rotated_Y = np.array([coord[1] for coord in rotated_coords]).reshape(Y.shape)

#         # Plot the subsidence contours for this panel using the custom levels
#         contour = ax.contourf(rotated_X, rotated_Y, mySxy.T, levels=levels, cmap=cmap_method, alpha=contour_transparancy, 
#                               vmin=panel_min_subsidence, vmax=panel_max_subsidence)
        
#         # Add colorbar
#         cbar = plt.colorbar(contour, label='Vertical Displacement [m]', ticks=tick_positions)
#         cbar.set_label('Vertical Displacement [m]', fontsize=8, fontweight='bold')
#         cbar.ax.yaxis.set_tick_params(labelsize=8)
#         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#         # Set labels and title for the plot
#         ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
#         ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')

#         # Format axes
#         ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
#         ax.xaxis.get_major_formatter().set_useOffset(False)
#         ax.xaxis.get_major_formatter().set_scientific(False)
#         ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
#         ax.yaxis.get_major_formatter().set_useOffset(False)
#         ax.yaxis.get_major_formatter().set_scientific(False)

#         plt.xticks(fontsize=10, rotation=45)
#         plt.yticks(fontsize=10)
        
#         ax.set_xlim(-100, panel_length+100)  
#         ax.set_ylim(-100, panel_width+100) 

#         # Grid and aspect ratio
#         ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
#         ax.set_aspect('equal')
        
#     return fig

# interval = 0.25
# def plot_vertical_displacement(all_panels_data, all_panel_min_x, all_panel_min_y):

#     for i, panel_data in enumerate(all_panels_data):
#         X, Y, mySxy = panel_data

#         # -------------------------------------------------
#         # DISCRETE LEVEL DEFINITION
#         # -------------------------------------------------
#         panel_min_subsidence = np.floor(mySxy.min() / interval) * interval
#         panel_max_subsidence = np.ceil(mySxy.max() / interval) * interval

#         levels = np.arange(panel_min_subsidence,
#                            panel_max_subsidence + interval,
#                            interval)

#         cmap = plt.get_cmap(cmap_method)
#         norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

#         tick_positions = levels[::max(1, int(len(levels) / 10))]

#         # -------------------------------------------------
#         # FIGURE
#         # -------------------------------------------------
#         fig, ax = plt.subplots(figsize=(10, 5))

#         # Shift to global coordinates
#         X_shifted = X + all_panel_min_x[i]
#         Y_shifted = Y + all_panel_min_y[i]

#         # Rotation center (bottom-left of panel)
#         rotation_center = (all_panel_min_x[i], all_panel_min_y[i])
#         lw_rotation_angle = 90.0 - lw_azimuth_angle

#         # Rotate coordinates
#         rotated_coords = [
#             rotate_point((x, y), lw_rotation_angle, rotation_center)
#             for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())
#         ]

#         rotated_X = np.array([c[0] for c in rotated_coords]).reshape(X.shape)
#         rotated_Y = np.array([c[1] for c in rotated_coords]).reshape(Y.shape)

#         # -------------------------------------------------
#         # CONTOUR PLOT (DISCRETE)
#         # -------------------------------------------------
#         contour = ax.contourf(
#             rotated_X,
#             rotated_Y,
#             mySxy.T,
#             levels=levels,
#             cmap=cmap,
#             norm=norm,
#             alpha=contour_transparancy,
#             extend='both'
#         )
#         # -------------------------------------------------
#         # COLORBAR
#         # -------------------------------------------------
#         cbar = plt.colorbar(contour, ticks=tick_positions)
#         cbar.set_label('Vertical Displacement [m]', fontsize=8, fontweight='bold')
#         cbar.ax.tick_params(labelsize=8)
#         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
#         ax.contour(X, Y, mySxy.T, colors="k", linewidths=0.4, alpha=0.6)


#         # -------------------------------------------------
#         # AXIS FORMATTING
#         # -------------------------------------------------
#         ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
#         ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')

#         ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
#         ax.xaxis.get_major_formatter().set_useOffset(False)
#         ax.xaxis.get_major_formatter().set_scientific(False)

#         ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
#         ax.yaxis.get_major_formatter().set_useOffset(False)
#         ax.yaxis.get_major_formatter().set_scientific(False)

#         plt.xticks(fontsize=10, rotation=45)
#         plt.yticks(fontsize=10)

#         ax.set_xlim(-100, panel_length+100)  
#         ax.set_ylim(-100, panel_width+100) 

#         ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
#         ax.set_aspect('equal')

#     return fig

def plot_vertical_displacement(all_panels_data, all_panel_min_x, all_panel_min_y,
                               use_manual_limits=False,
                               user_min=0.0,
                               user_max=0.0,
                               user_interval=0.25):
    for i, panel_data in enumerate(all_panels_data):
        X, Y, mySxy = panel_data

        # -------------------------------------------------
        # DISCRETE LEVEL DEFINITION (auto or manual)
        # -------------------------------------------------
        auto_min = np.floor(mySxy.min() / user_interval) * user_interval
        auto_max = np.ceil(mySxy.max() / user_interval) * user_interval

        if use_manual_limits and user_min < user_max:
            panel_min_subsidence = user_min
            panel_max_subsidence = user_max
            interval = user_interval
        else:
            panel_min_subsidence = auto_min
            panel_max_subsidence = auto_max
            interval = user_interval

        levels = np.arange(panel_min_subsidence,
                           panel_max_subsidence + interval,
                           interval)

        cmap = plt.get_cmap(cmap_method)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        tick_positions = levels[::max(1, int(len(levels) / 10))]

        # -------------------------------------------------
        # FIGURE
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))

        # Shift to global coordinates
        X_shifted = X + all_panel_min_x[i]
        Y_shifted = Y + all_panel_min_y[i]

        # Rotation center (bottom-left of panel)
        rotation_center = (all_panel_min_x[i], all_panel_min_y[i])
        lw_rotation_angle = 90.0 - lw_azimuth_angle

        # Rotate coordinates
        rotated_coords = [
            rotate_point((x, y), lw_rotation_angle, rotation_center)
            for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())
        ]

        rotated_X = np.array([c[0] for c in rotated_coords]).reshape(X.shape)
        rotated_Y = np.array([c[1] for c in rotated_coords]).reshape(Y.shape)

        # -------------------------------------------------
        # CONTOUR PLOT (DISCRETE)
        # -------------------------------------------------
        contour = ax.contourf(
            rotated_X,
            rotated_Y,
            mySxy.T,
            levels=levels,
            cmap=cmap,
            norm=norm,
            alpha=contour_transparancy,
            extend='both'
        )
        ax.contour(X, Y, mySxy.T, colors="k", linewidths=0.4, alpha=0.6)

        # -------------------------------------------------
        # COLORBAR
        # -------------------------------------------------
        cbar = plt.colorbar(contour, ticks=tick_positions)
        cbar.set_label('Vertical Displacement [m]', fontsize=8, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # -------------------------------------------------
        # AXIS FORMATTING
        # -------------------------------------------------
        ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
        ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')

        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.get_major_formatter().set_scientific(False)

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_scientific(False)

        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)

        ax.set_xlim(-100, panel_length+100)  
        ax.set_ylim(-100, panel_width+100) 

        ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
        ax.set_aspect('equal')

    return fig
#======================================================================================================================================================

def plot_horizontal_displacement(all_panels_data, all_panel_min_x, all_panel_min_y):
    
    # Define the center point for rotation
    # *** FIX: Rotation center must be the bottom-left corner of the current panel's coordinate system. ***
    # rotation_center = center#(619925.17, 7594941.26) # Commented out global placeholder

    for i, panel_data in enumerate(all_panels_data):
        X, Y, mySxy = panel_data
        
        # Calculate min and max for each panel's subsidence
        panel_min_subsidence = round(mySxy.min(),1)
        panel_max_subsidence = round(mySxy.max(),1)
        levels = np.linspace(panel_min_subsidence, panel_max_subsidence, 2000)
        tick_positions = np.linspace(panel_min_subsidence, panel_max_subsidence, 10)

        # Create a new figure for each panel
        fig, ax = plt.subplots(figsize=(10, 5))

        # Shift X and Y by panel's min_x and min_y (This creates the global coordinates)
        X_shifted = X + all_panel_min_x[i]
        Y_shifted = Y + all_panel_min_y[i]
        
        # *** The fix: Set the rotation center to the bottom-left corner of the current panel in global coordinates ***
        # This point is (all_panel_min_x[i], all_panel_min_y[i]) in the shifted system.
        rotation_center = (all_panel_min_x[i], all_panel_min_y[i])

        # Rotate all points by lw_rotation_angle degrees
        # The rotation is now anchored correctly to the bottom-left corner of the panel in global coordinates.
        lw_rotation_angle = 90.0 - lw_azimuth_angle
        rotated_coords = [rotate_point((x, y), lw_rotation_angle, rotation_center) 
                          for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())]
        rotated_X = np.array([coord[0] for coord in rotated_coords]).reshape(X.shape)
        rotated_Y = np.array([coord[1] for coord in rotated_coords]).reshape(Y.shape)

        # Plot the subsidence contours for this panel using the custom levels
        contour = ax.contourf(rotated_X, rotated_Y, mySxy.T, levels=levels, cmap=cmap_method, alpha=contour_transparancy, 
                              vmin=panel_min_subsidence, vmax=panel_max_subsidence)
        
        # Add colorbar
        cbar = plt.colorbar(contour, label='Horizontal Displacement [m]', ticks=tick_positions)
        cbar.set_label('Horizontal Displacement [m]', fontsize=8, fontweight='bold')
        cbar.ax.yaxis.set_tick_params(labelsize=8)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Set labels and title for the plot
        ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
        ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')

        # Format axes
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_scientific(False)

        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        
        ax.set_xlim(-100, panel_length+100)  
        ax.set_ylim(-100, panel_width+100) 

        # Grid and aspect ratio
        ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
        ax.set_aspect('equal')
        
    return fig


#======================================================================================================================================================


def plot_horizontal_strain(all_panels_data, all_panel_min_x, all_panel_min_y):
    
    # Define the center point for rotation
    # *** FIX: Rotation center must be the bottom-left corner of the current panel's coordinate system. ***
    # rotation_center = center#(619925.17, 7594941.26) # Commented out global placeholder

    for i, panel_data in enumerate(all_panels_data):
        X, Y, mySxy = panel_data
        
        # Calculate min and max for each panel's subsidence
        panel_min_subsidence = round(mySxy.min(),1)
        panel_max_subsidence = round(mySxy.max(),1)
        levels = np.linspace(panel_min_subsidence, panel_max_subsidence, 2000)
        tick_positions = np.linspace(panel_min_subsidence, panel_max_subsidence, 10)

        # Create a new figure for each panel
        fig, ax = plt.subplots(figsize=(10, 5))

        # Shift X and Y by panel's min_x and min_y (This creates the global coordinates)
        X_shifted = X + all_panel_min_x[i]
        Y_shifted = Y + all_panel_min_y[i]
        
        # *** The fix: Set the rotation center to the bottom-left corner of the current panel in global coordinates ***
        # This point is (all_panel_min_x[i], all_panel_min_y[i]) in the shifted system.
        rotation_center = (all_panel_min_x[i], all_panel_min_y[i])

        # Rotate all points by lw_rotation_angle degrees
        # The rotation is now anchored correctly to the bottom-left corner of the panel in global coordinates.
        lw_rotation_angle = 90.0 - lw_azimuth_angle
        rotated_coords = [rotate_point((x, y), lw_rotation_angle, rotation_center) 
                          for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())]
        rotated_X = np.array([coord[0] for coord in rotated_coords]).reshape(X.shape)
        rotated_Y = np.array([coord[1] for coord in rotated_coords]).reshape(Y.shape)

        # Plot the subsidence contours for this panel using the custom levels
        contour = ax.contourf(rotated_X, rotated_Y, mySxy.T, levels=levels, cmap=cmap_method, alpha=contour_transparancy, 
                              vmin=panel_min_subsidence, vmax=panel_max_subsidence)
        
        # Add colorbar
        cbar = plt.colorbar(contour, label='Horizontal Strain [mm/m]', ticks=tick_positions)
        cbar.set_label('Horizontal Strain [mm/m]', fontsize=8, fontweight='bold')
        cbar.ax.yaxis.set_tick_params(labelsize=8)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Set labels and title for the plot
        ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
        ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')

        # Format axes
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_scientific(False)

        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        
        ax.set_xlim(-100, panel_length+100)  
        ax.set_ylim(-100, panel_width+100) 

        # Grid and aspect ratio
        ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
        ax.set_aspect('equal')
        
    return fig
#======================================================================================================================================================


#======================================================================================================================================================


def plot_tilt(all_panels_data, all_panel_min_x, all_panel_min_y):
    
    # Define the center point for rotation
    # *** FIX: Rotation center must be the bottom-left corner of the current panel's coordinate system. ***
    # rotation_center = center#(619925.17, 7594941.26) # Commented out global placeholder

    for i, panel_data in enumerate(all_panels_data):
        X, Y, mySxy = panel_data
        
        # Calculate min and max for each panel's subsidence
        panel_min_subsidence = round(mySxy.min(),1)
        panel_max_subsidence = round(mySxy.max(),1)
        levels = np.linspace(panel_min_subsidence, panel_max_subsidence, 2000)
        tick_positions = np.linspace(panel_min_subsidence, panel_max_subsidence, 10)

        # Create a new figure for each panel
        fig, ax = plt.subplots(figsize=(10, 5))

        # Shift X and Y by panel's min_x and min_y (This creates the global coordinates)
        X_shifted = X + all_panel_min_x[i]
        Y_shifted = Y + all_panel_min_y[i]
        
        # *** The fix: Set the rotation center to the bottom-left corner of the current panel in global coordinates ***
        # This point is (all_panel_min_x[i], all_panel_min_y[i]) in the shifted system.
        rotation_center = (all_panel_min_x[i], all_panel_min_y[i])

        # Rotate all points by lw_rotation_angle degrees
        # The rotation is now anchored correctly to the bottom-left corner of the panel in global coordinates.
        lw_rotation_angle = 90.0 - lw_azimuth_angle
        rotated_coords = [rotate_point((x, y), lw_rotation_angle, rotation_center) 
                          for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())]
        rotated_X = np.array([coord[0] for coord in rotated_coords]).reshape(X.shape)
        rotated_Y = np.array([coord[1] for coord in rotated_coords]).reshape(Y.shape)

        # Plot the subsidence contours for this panel using the custom levels
        contour = ax.contourf(rotated_X, rotated_Y, mySxy.T, levels=levels, cmap=cmap_method, alpha=contour_transparancy, 
                              vmin=panel_min_subsidence, vmax=panel_max_subsidence)
        
        # Add colorbar
        cbar = plt.colorbar(contour, label='Tilt [mm/m]', ticks=tick_positions)
        cbar.set_label('Tilt [mm/m]', fontsize=8, fontweight='bold')
        cbar.ax.yaxis.set_tick_params(labelsize=8)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Set labels and title for the plot
        ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold')
        ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold')

        # Format axes
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.get_major_formatter().set_scientific(False)

        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        
        ax.set_xlim(-100, panel_length+100)  
        ax.set_ylim(-100, panel_width+100) 

        # Grid and aspect ratio
        ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
        ax.set_aspect('equal')
        
    return fig
#======================================================================================================================================================


def plot_vertical_displacement_3D(all_panels_data, all_panel_min_x, all_panel_min_y):

    for i, panel_data in enumerate(all_panels_data):
        X, Y, mySxy = panel_data

        panel_min_subsidence = round(mySxy.min(), 1)
        panel_max_subsidence = round(mySxy.max(), 1)

        # ----------------- Coordinate transform -----------------
        X_shifted = X + all_panel_min_x[i]
        Y_shifted = Y + all_panel_min_y[i]
        rotation_center = (all_panel_min_x[i], all_panel_min_y[i])
        lw_rotation_angle = 90.0 - lw_azimuth_angle

        rotated_coords = [
            rotate_point((x, y), lw_rotation_angle, rotation_center)
            for x, y in zip(X_shifted.flatten(), Y_shifted.flatten())
        ]

        rotated_X = np.array([c[0] for c in rotated_coords]).reshape(X.shape)
        rotated_Y = np.array([c[1] for c in rotated_coords]).reshape(Y.shape)

        Z = mySxy.T

        # ----------------- FIGURE STYLE -----------------
        fig = plt.figure(figsize=(13, 6), facecolor='#f7f7f7')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#f7f7f7')

        # Remove pane fills (clean look)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False


        # ----------------- SURFACE -----------------
        surf = ax.plot_surface(
            rotated_X,
            rotated_Y,
            Z,
            cmap='Spectral',               # perceptually uniform
            vmin=panel_min_subsidence,
            vmax=panel_max_subsidence,
            linewidth=0.15,               # subtle mesh
            edgecolor='k',
            antialiased=True,
            shade=True
        )

        # ----------------- GROUND PLANE (Z = 0) -----------------
        ax.plot_surface(
            rotated_X,
            rotated_Y,
            np.zeros_like(Z),
            color='lightgrey',
            alpha=0.15,
            linewidth=0,
            zorder=0
        )
        
        

        # ----------------- VIEW -----------------
        ax.view_init(elev=25, azim=45)
        ax.dist = 9  # camera distance (smaller = closer)

        # ----------------- LABELS -----------------
        ax.set_xlabel('Easting [m]', fontsize=10, fontweight='bold', labelpad=10)
        ax.set_ylabel('Northing [m]', fontsize=10, fontweight='bold', labelpad=10)
        ax.set_zlabel('V(x,y) [m]', fontsize=10, fontweight='bold', labelpad=8)

        # ----------------- LIMITS -----------------
        ax.set_xlim(-100, panel_length+100)  
        ax.set_ylim(-100, panel_width+100) 
        # ax.set_zlim(panel_min_subsidence, panel_max_subsidence)

        # ----------------- TICKS & FORMAT -----------------
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            axis.set_tick_params(labelsize=8, pad=3)

        ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # ----------------- ASPECT -----------------
        ax.set_box_aspect((1, 1, 0.35))  # visually pleasing exaggeration

        # ----------------- COLORBAR -----------------
        cbar = fig.colorbar(
            surf,
            ax=ax,
            shrink=0.55,
            aspect=18,
            pad=0.08
        )
        cbar.set_label('Vertical Displacement [m]', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        # ----------------- TITLE -----------------
        ax.set_title(
            f'3D Subsidence Surface – Panel {i+1}',
            fontsize=11,
            fontweight='bold',
            pad=12
        )

        # ----------------- SAVE -----------------
        plt.tight_layout()
    return fig




# -------------------------------------------------
# Run model
# -------------------------------------------------
# ---------------------------------
# Run all models ONCE
# ---------------------------------

if run_model:
    with st.spinner("Running subsidence model..."):
        try:
            st.subheader("3D Subsidence Surface")
            
            all_panels_data = []
            for i in range(len(all_panel_widths)):
                X, Y, Sxy = calculate_subsidence(
                    lw_panel_id=1,
                    panel_width=all_panel_widths[i],
                    panel_length=all_panel_lengths[i],
                    extraction_thick=extraction_thickness,
                    percentage_hard_rock=percentage_hard_rock,
                    depth_of_cover=depth_of_cover_input
                )
                all_panels_data.append((X, Y, Sxy))
            
            fig_3d = plot_vertical_displacement_3D(
                all_panels_data, all_panel_min_x, all_panel_min_y
            )
            st.pyplot(fig_3d, use_container_width=True)

            st.subheader("Surface Response Parameters")

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                all_panels_data = []
                for i in range(len(all_panel_widths)):
                    X, Y, Sxy = calculate_subsidence(
                        lw_panel_id=1,
                        panel_width=all_panel_widths[i],
                        panel_length=all_panel_lengths[i],
                        extraction_thick=extraction_thickness,
                        percentage_hard_rock=percentage_hard_rock,
                        depth_of_cover=depth_of_cover_input
                    )
                    all_panels_data.append((X, Y, Sxy))
                st.markdown("**Vertical Displacement**")
                # fig = plot_vertical_displacement(
                #     all_panels_data, all_panel_min_x, all_panel_min_y
                # )
                # Get actual data range for smart defaults in UI (optional improvement)
                all_Sxy = [panel[2] for panel in all_panels_data]
                global_min = min(s.min() for s in all_Sxy)
                global_max = max(s.max() for s in all_Sxy)
                
                # Only use user inputs if manual mode is on; otherwise, auto is used inside function
                fig = plot_vertical_displacement(
                    all_panels_data, all_panel_min_x, all_panel_min_y,
                    use_manual_limits=use_manual_limits,
                    user_min=panel_min_input,
                    user_max=panel_max_input,
                    user_interval=interval_input
                )
                st.pyplot(fig, use_container_width=True)

            with col2:
                all_panels_data = []
                for i in range(len(all_panel_widths)):
                    X, Y, Sxy = calculate_horizontal_displacement(
                        lw_panel_id=1,
                        panel_width=all_panel_widths[i],
                        panel_length=all_panel_lengths[i],
                        extraction_thick=extraction_thickness,
                        percentage_hard_rock=percentage_hard_rock,
                        depth_of_cover=depth_of_cover_input
                    )
                    all_panels_data.append((X, Y, Sxy))
                st.markdown("**Horizontal Displacement**")
                fig = plot_horizontal_displacement(
                    all_panels_data, all_panel_min_x, all_panel_min_y
                )
                st.pyplot(fig, use_container_width=True)

            with col3:
                st.markdown("**Horizontal Strain**")
                all_panels_data = []
                for i in range(len(all_panel_widths)):
                    X, Y, Sxy = calculate_horizontal_strain(
                        lw_panel_id=1,
                        panel_width=all_panel_widths[i],
                        panel_length=all_panel_lengths[i],
                        extraction_thick=extraction_thickness,
                        percentage_hard_rock=percentage_hard_rock,
                        depth_of_cover=depth_of_cover_input
                    )
                    all_panels_data.append((X, Y, Sxy))
                fig = plot_horizontal_strain(
                    all_panels_data, all_panel_min_x, all_panel_min_y
                )
                st.pyplot(fig, use_container_width=True)

            with col4:
                st.markdown("**Tilt**")
                all_panels_data = []
                for i in range(len(all_panel_widths)):
                    X, Y, Sxy = calculate_tilt(
                        lw_panel_id=1,
                        panel_width=all_panel_widths[i],
                        panel_length=all_panel_lengths[i],
                        extraction_thick=extraction_thickness,
                        percentage_hard_rock=percentage_hard_rock,
                        depth_of_cover=depth_of_cover_input
                    )
                    all_panels_data.append((X, Y, Sxy))
                fig = plot_tilt(
                    all_panels_data, all_panel_min_x, all_panel_min_y
                )
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
