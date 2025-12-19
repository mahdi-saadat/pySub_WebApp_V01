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

import tempfile

def save_uploaded_file(uploaded_file):
    """
    Save a Streamlit uploaded file to a temporary folder
    and return the file path.
    """
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    return None


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
st.write("App started successfully")

st.success("Modules imported successfully")

st.markdown("### Enter panel and geotechnical parameters")

# -------------------------------------------------
# Inputs
# -------------------------------------------------
panel_width = st.number_input("Panel width (m)", 50.0, 1000.0, 270.0)
panel_length = st.number_input("Panel length (m)", 50.0, 5000.0, 1000.0)
depth_of_cover = st.number_input("Depth of cover (m)", 50.0, 1000.0, 115.0)
extraction_thickness = st.number_input("Extraction thickness (m)", 1.0, 10.0, 4.20)
lw_azimuth_angle = st.number_input("Longwall Panel Azimuth", 0.0, 90.0, 90.0)
percentage_hard_rock = st.number_input("Hard Rock Percentage", 10.0, 100.0, 30.0)

        # DXF Inputs:
#-------------------------------------------
uploaded_panel_dxf = st.file_uploader(
    "Upload panel DXF",
    type=["dxf"]
)
uploaded_parts_dxf = st.file_uploader(
    "Upload Mine Plane and Structures DXF",
    type=["dxf"]
)
#-------------------------------------------
panel_dxf_path = save_uploaded_file(uploaded_panel_dxf)
parts_dxf_path = save_uploaded_file(uploaded_parts_dxf)

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

#---------------------------------------------------------------------- Start DXF Analysis ---------------------------------------------------------------------

# Global variables to store aggregated results
all_panel_widths = []
all_panel_lengths = []
all_pillar_spacings = []

def calculate_dimensions(vertices):
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    if x_coords and y_coords:
        length = max(x_coords) - min(x_coords)
        width = max(y_coords) - min(y_coords)
        #print(f"✔Pane width: {width}")
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        return length, width, min_x, max_x, min_y, max_y
    else:
        return None, None, None, None, None, None

def find_panel_lines(lines):
    panels_temp = []
    line_dict = defaultdict(list)
    for line in lines:
        p1, p2 = line.dxf.start, line.dxf.end
        line_dict[(p1.x, p1.y)].append((p2.x, p2.y))
        line_dict[(p2.x, p2.y)].append((p1.x, p1.y))
    
    visited = set()
    for start, ends in line_dict.items():
        if start in visited:
            continue
        panel = [start]
        visited.add(start)
        current = start
        while len(panel) < 4:
            for end in line_dict[current]:
                if end not in visited:
                    panel.append(end)
                    visited.add(end)
                    current = end
                    break
        
        # Ensure the panel is a rectangle by checking the number of vertices
        if len(panel) == 4:
            panels_temp.append(panel)
    
    # Calculate dimensions and sort panels based on min_y
    global sorted_panels
    panels = [(panel, calculate_dimensions(panel)) for panel in panels_temp]
    panels_sorted = sorted(panels, key=lambda x: x[1][4])  # Sorting by min_y
    sorted_panels = [panel[0] for panel in panels_sorted]
    return sorted_panels

def calculate_pillar_spacing(panels):
    min_y_values = []
    max_y_values = []
    for panel in panels:
        _, _, _, _, min_y, max_y = calculate_dimensions(panel)
        min_y_values.append(min_y)
        max_y_values.append(max_y)
    
    pillar_spacings = [abs(min_y_values[i + 1] - max_y_values[i]) for i in range(len(min_y_values) - 1)]
    
    # Add the width of the last panel as the spacing next to it
    if panels:
        last_panel_width = max_y_values[-1] - min_y_values[-1]
        pillar_spacings.append(last_panel_width)
    
    return pillar_spacings


def process_dxf_files(file_path):
    global all_panel_widths, all_panel_lengths, all_panel_min_x, all_panel_max_x, all_panel_min_y, all_panel_max_y, all_pillar_spacings

    all_panel_widths = []
    all_panel_lengths = []
    all_panel_min_x = []
    all_panel_max_x = []
    all_panel_min_y = []
    all_panel_max_y = []
    all_pillar_spacings = []

    if file_path is None:
        return

    try:
        doc = ezdxf.readfile(file_path)
    except IOError:
        st.error(f"Cannot open {file_path}.")
        return

    msp = doc.modelspace()
    lines = list(msp.query("LINE"))

    panels = find_panel_lines(lines)
    
    for panel in panels:
        length, width, min_x, max_x, min_y, max_y = calculate_dimensions(panel)
        all_panel_widths.append(width)
        all_panel_lengths.append(length)
        all_panel_min_x.append(min_x)
        all_panel_max_x.append(max_x)
        all_panel_min_y.append(min_y)
        all_panel_max_y.append(max_y)

    all_pillar_spacings.extend(calculate_pillar_spacing(panels))


# Set the directory containing DXF files

# Process all DXF files in the directory
process_dxf_files(panel_dxf_path)

#---------------------------------------------------------------------- End DXF Analysis ---------------------------------------------------------------------

ploting_panels = []
def process_dxf_files(directory):

    # Loop through all DXF files in the specified directory
    for filename in os.listdir(directory):
            # Process panels in this DXF file
            if filename.endswith(".dxf"):
                dxf_path = os.path.join(directory, filename)
                #dxf_letter = filename.split('_')[0][-1] 
                #print(dxf_letter)
                try:
                    doc = ezdxf.readfile(dxf_path)
                except IOError:
                    print(f"Error: Cannot open {dxf_path}. Check if the file exists.")
                    continue

                msp = doc.modelspace()
                lines = list(msp.query("LINE"))
            temp_panel = find_panel_lines(lines)
            ploting_panels.append(temp_panel)

# Rotation function for panels
def rotate_panel(panel, angle_deg, ref_point=(0, 0)):
    """Rotate the panel's coordinates by a specified angle around a reference point."""
    theta = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    
    # Translate panel coordinates to rotate around reference point
    panel_rotated = []
    for x, y in panel:
        x_shifted, y_shifted = x - ref_point[0], y - ref_point[1]  # Translate to reference point
        x_rotated, y_rotated = rotation_matrix @ np.array([x_shifted, y_shifted])
        panel_rotated.append((x_rotated + ref_point[0], y_rotated + ref_point[1]))  # Translate back
    
    return panel_rotated

# Function to plot panel outlines with rotation
def plot_panels_dxf(ax, panels, angle_deg):
    for panel in panels:
        # Rotate the panel by the specified angle
        rotated_panel = rotate_panel(panel, angle_deg)

        # Plot the rotated panel
        x_coords, y_coords = zip(*rotated_panel)
        x_coords += (x_coords[0],)  # Close the polygon
        y_coords += (y_coords[0],)
        ax.plot(x_coords, y_coords, color='black', linewidth=1.5)

#----------------------------------------------------------------------
process_dxf_files(uploaded_panel_dxf)

#------------------------------------------------------- DXF Processing for LW Geometries

def rotate_point_for_LW(point, angle, center):
    # (This function remains the same)
    angle_rad = np.radians(angle)
    x, y = point
    cx, cy = center
    x_translated = x - cx
    y_translated = y - cy
    x_rotated = x_translated * np.cos(angle_rad) - y_translated * np.sin(angle_rad)
    y_rotated = x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)
    x_final = x_rotated + cx
    y_final = y_rotated + cy
    return (x_final, y_final)

def plot_rotated_dxf_LW_lines_check(directory, angle, ax=None, line_color='k'):
    # Added line_color argument, default is 'k' (black)
    if ax is None:
        fig, ax = plt.subplots()

    all_x_coords = []
    all_y_coords = []
    
    # Pass 1: Collect all coordinates to determine the Min/Max bounds
    for filename in os.listdir(directory):
        if filename.endswith(".dxf"):
            dxf_path = os.path.join(directory, filename)
            try:
                doc = ezdxf.readfile(dxf_path)
            except (IOError, ezdxf.DXFStructureError) as e:
                print(f"Skipping {dxf_path} due to error: {e}")
                continue

            msp = doc.modelspace()
            lines = msp.query("LINE")

            for line in lines:
                all_x_coords.extend([line.dxf.start.x, line.dxf.end.x])
                all_y_coords.extend([line.dxf.start.y, line.dxf.end.y])

    # Calculate the rotation center (bottom-left corner)
    if all_x_coords and all_y_coords:
        center_x = min(all_x_coords)
        center_y = min(all_y_coords)
        rotation_center = (center_x, center_y)
        #print(f"Rotation Center for DXF in {directory}: {rotation_center}")

        # Pass 2: Plot the rotated lines
        for filename in os.listdir(directory):
            if filename.endswith(".dxf"):
                dxf_path = os.path.join(directory, filename)
                try:
                    doc = ezdxf.readfile(dxf_path)
                except (IOError, ezdxf.DXFStructureError) as e:
                    continue

                msp = doc.modelspace()
                lines = msp.query("LINE")

                for line in lines:
                    start_rotated = rotate_point_for_LW((line.dxf.start.x, line.dxf.start.y), angle, rotation_center)
                    end_rotated = rotate_point_for_LW((line.dxf.end.x, line.dxf.end.y), angle, rotation_center)
                    
                    # *** FIX: Use line_color variable here ***
                    ax.plot([start_rotated[0], end_rotated[0]], 
                            [start_rotated[1], end_rotated[1]], 
                            color=line_color, linestyle='-', linewidth=1.5) # Increased linewidth slightly for visibility

        ax.set_aspect('equal')
    else:
        print(f"No DXF lines found in directory: {directory}")

#===================================================================================================================================================


def rotate_point_for_LW(point, angle, center):
    # (This function remains the same)
    angle_rad = np.radians(angle)
    x, y = point
    cx, cy = center
    x_translated = x - cx
    y_translated = y - cy
    x_rotated = x_translated * np.cos(angle_rad) - y_translated * np.sin(angle_rad)
    y_rotated = x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)
    x_final = x_rotated + cx
    y_final = y_rotated + cy
    return (x_final, y_final)

def plot_rotated_dxf_LW_lines_check(directory, angle, ax=None, line_color='k', flatten_arc_segments=32):
    if ax is None:
        fig, ax = plt.subplots()

    all_x_coords = []
    all_y_coords = []
    
    def flatten_entity(entity):
        points = []
        dxftype = entity.dxftype()
        
        if dxftype == 'LINE':
            points = [(entity.dxf.start.x, entity.dxf.start.y),
                      (entity.dxf.end.x, entity.dxf.end.y)]
        
        elif dxftype == 'LWPOLYLINE':
            # LWPOLYLINE: 2D polyline, vertices() is a method
            points = [(pt[0], pt[1]) for pt in entity.vertices()]
            if entity.closed:
                points.append(points[0])
        
        elif dxftype == 'POLYLINE':
            # POLYLINE: vertices is a LIST of vertex entities (not a method!)
            # Each vertex is a DXF entity with .dxf.location (a Vec3)
            try:
                vertex_points = []
                for vertex in entity.vertices:  # ← NO parentheses! It's a list
                    loc = vertex.dxf.location
                    vertex_points.append((float(loc.x), float(loc.y)))
                if vertex_points:
                    points = vertex_points
                    # Check if closed: POLYLINE uses flags
                    if entity.is_closed:
                        points.append(points[0])
            except Exception as e:
                print(f"Warning: Failed to parse POLYLINE: {e}")
                return []
        
        elif dxftype == 'ARC':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = float(entity.dxf.radius)
            start_angle = np.radians(entity.dxf.start_angle)
            end_angle = np.radians(entity.dxf.end_angle)
            if end_angle < start_angle:
                end_angle += 2 * np.pi
            angles = np.linspace(start_angle, end_angle, flatten_arc_segments)
            points = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]
        
        elif dxftype == 'CIRCLE':
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = float(entity.dxf.radius)
            angles = np.linspace(0, 2 * np.pi, flatten_arc_segments)
            points = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]
        
        else:
            return []  # Unsupported entity

        return points

    # Pass 1: Collect all coordinates
    for filename in os.listdir(directory):
        if filename.endswith(".dxf"):
            dxf_path = os.path.join(directory, filename)
            try:
                doc = ezdxf.readfile(dxf_path)
            except (IOError, ezdxf.DXFStructureError) as e:
                print(f"Skipping {dxf_path} due to error: {e}")
                continue

            msp = doc.modelspace()
            for entity in msp:
                if entity.dxftype() in {'LINE', 'LWPOLYLINE', 'POLYLINE', 'ARC', 'CIRCLE'}:
                    pts = flatten_entity(entity)
                    if pts:
                        xs, ys = zip(*pts)
                        all_x_coords.extend(xs)
                        all_y_coords.extend(ys)

    if not (all_x_coords and all_y_coords):
        print(f"No supported DXF entities found in directory: {directory}")
        return

    # Global rotation center (bottom-left)
    rotation_center = (min(all_x_coords), min(all_y_coords))
    print(f"Rotation Center for DXF in {directory}: {rotation_center}")

    # Pass 2: Plot rotated geometry
    for filename in os.listdir(directory):
        if filename.endswith(".dxf"):
            dxf_path = os.path.join(directory, filename)
            try:
                doc = ezdxf.readfile(dxf_path)
            except (IOError, ezdxf.DXFStructureError) as e:
                print(f"Skipping {dxf_path} during plotting: {e}")
                continue

            msp = doc.modelspace()
            for entity in msp:
                if entity.dxftype() not in {'LINE', 'LWPOLYLINE', 'POLYLINE', 'ARC', 'CIRCLE'}:
                    continue
                pts = flatten_entity(entity)
                if not pts:
                    continue
                rotated_pts = [rotate_point_for_LW(p, angle, rotation_center) for p in pts]
                x_vals, y_vals = zip(*rotated_pts)
                ax.plot(x_vals, y_vals, color=line_color, linestyle='-', linewidth=1.0)

    ax.set_aspect('equal')


#--------------------- Depth of Cover ----------------

"""
    This depth of cover can be modified later
"""
# Your initial data
data = [
    {
        "Panel ID": 1,
        "Panel ID LW": "LW03",
        "Start": 230,
        "End": 130,
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
for ip, (lw_id, igrad) in zip(range(len(all_panel_widths)), gradient_dict.items()):
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
    
    i_width = all_panel_widths[ip]
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
    
    inflection_point_list = inflection_points_dict[lw_panel_id]
    major_influence_radius_array = major_influence_radius_dict[lw_panel_id]
    
    # Iterate over x and y values
    for i, x in enumerate(x_values_limit):
        inflection_point_to_edge_conservative = inflection_point_list[0][i]
        major_influence_radius = major_influence_radius_array[0][i]
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

# Define global grid resolution
global_resolution = 100  # Adjust this for finer/coarser grids
all_panels_data = []

for i in range(len(all_panel_widths)):
    X, Y, Sxy = calculate_subsidence(
        lw_panel_id=i+1,
        panel_width=all_panel_widths[i],
        panel_length=all_panel_lengths[i],
        extraction_thick=extraction_thickness,
        percentage_hard_rock=percentage_hard_rock,
        depth_of_cover=depth_of_cover
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


def plot_vertical_displacement(all_panels_data, all_panel_min_x, all_panel_min_y, ploting_panels):
    
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

        # Plot the panel boundaries (if needed)
        plot_rotated_dxf_LW_lines_check(uploaded_parts_dxf, 0.0, ax)
        
        # Add colorbar
        cbar = plt.colorbar(contour, label='Vertical Displacement [m]', ticks=tick_positions)
        cbar.set_label('Vertical Displacement [m]', fontsize=8, fontweight='bold')
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
        
        ax.set_xlim(-100, 1100)  # Setting xlim based on panel boundaries
        ax.set_ylim(-100, 370) 

        # Grid and aspect ratio
        ax.grid(True, color='gray', linestyle='--', linewidth=0.1)
        ax.set_aspect('equal')
        
    return fig

# -------------------------------------------------
# Run model
# -------------------------------------------------
if st.button("Run Subsidence Assessment"):
    if panel_dxf_path is None or parts_dxf_path is None:
        st.error("Please upload both Panel DXF and Parts DXF files.")
    else:
        with st.spinner("Running subsidence model..."):
            try:
                # Step 1: Process DXF
                process_dxf_files(panel_dxf_path)

                # Step 2: Calculate subsidence for all panels
                all_panels_data = []
                for i in range(len(all_panel_widths)):
                    X, Y, Sxy = calculate_subsidence(
                        lw_panel_id=i+1,
                        panel_width=all_panel_widths[i],
                        panel_length=all_panel_lengths[i],
                        extraction_thick=extraction_thickness,
                        percentage_hard_rock=percentage_hard_rock,
                        depth_of_cover=depth_of_cover
                    )
                    all_panels_data.append((X, Y, Sxy))

                # Step 3: Plot results
                fig = plot_vertical_displacement(all_panels_data, all_panel_min_x, all_panel_min_y, ploting_panels)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

