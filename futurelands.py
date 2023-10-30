# -*- coding: utf-8 -*-
"""
Created on Oct 2023

@author: Angelos Alamanos
"""
#####  Data Preprocessing - Estimate the Transition Probability Matrices
# Using the Change Matrices (tabulate areas) from ArcGIS (replace with your own values)

import numpy as np

# Example change matrix for a specific year change (e.g., from 2006 to 2011)
change_matrix_06to11 = np.array([
    [58110300, 39600, 17100, 55800, 444600],   # From Water (1)
    [0, 78462900, 0, 0, 0],  # From Urban (2)
    [8100, 5400, 359100, 0, 900],   # From Barren Land (3)
    [119700, 107100, 0, 79761600, 509400],     # From Forest (4)
    [431100, 1272600, 18000, 252000, 477891900]   # From Crops (5)
])

change_matrix_11to16 = np.array([
    [58081500, 30600, 29700, 87300, 440100],   # From Water (1)
    [0, 79887600, 0, 0, 0],  # From Urban (2)
    [17100, 5400, 369900, 1800, 0],   # From Barren Land (3)
    [118800, 77400, 0, 79321500, 551700],     # From Forest (4)
    [382500, 1329300, 2700, 109800, 477022500]   # From Crops (5)
])

change_matrix_16to21 = np.array([
    [58503600, 27000, 36900, 28800, 3600],   # From Water (1)
    [900, 81312300, 0, 900, 16200],  # From Urban (2)
    [3600, 8100, 373500, 17100, 0],   # From Barren Land (3)
    [29700, 165600, 3600, 79142400, 179100],     # From Forest (4)
    [192600, 1146600, 734400, 13500, 475927200]   # From Crops (5)
])

# Normalize the Change Matrices
total_transitions1 = np.sum(change_matrix_06to11, axis=1)
normalized_matrix1 = change_matrix_06to11 / total_transitions1[:, np.newaxis]

total_transitions2 = np.sum(change_matrix_11to16, axis=1)
normalized_matrix2 = change_matrix_11to16 / total_transitions2[:, np.newaxis]

total_transitions3 = np.sum(change_matrix_16to21, axis=1)
normalized_matrix3 = change_matrix_16to21 / total_transitions3[:, np.newaxis]


# Step 4: Construct the Transition Probability Matrix
transition_probability_matrix1 = normalized_matrix1
transition_probability_matrix2 = normalized_matrix2
transition_probability_matrix3 = normalized_matrix3

np.set_printoptions(precision=3, suppress=True)

# Display the resulting transition probability matrix
np.set_printoptions(precision=4, suppress=True)
print(transition_probability_matrix1)
print(transition_probability_matrix2)
print(transition_probability_matrix3)



##################### 2016 to 2021 #############################

import numpy as np
import os
import rasterio
from rasterio.transform import from_origin

# Define the data directory and file paths
data_directory = r'D:\your\path'
land_use_2016 = r'D:\your\path\cedar16reclas1.tif'

# Define the transition probabilities matrix manually
transition_probs = np.array([
    [0.998, 0.001, 0.001, 0.000, 0.000],
    [0.000, 1.000, 0.000, 0.000, 0.000],
    [0.009, 0.020, 0.928, 0.043, 0.000],
    [0.000, 0.002, 0.001, 0.995, 0.002],
    [0.000, 0.002, 0.002, 0.000, 0.996]
   ])

# Check if the probabilities are valid (between 0 and 1 and sum to 1 for each row)
if (transition_probs < 0).any() or (transition_probs > 1).any() or not np.allclose(transition_probs.sum(axis=1), 1):
    raise ValueError("Invalid transition probabilities.") 

# Load the 2016 land use map using rasterio
with rasterio.open(land_use_2016) as src:
    current_land_use_map_2016 = src.read(1)
    transform = src.transform  # Get the spatial transform from the input raster

# Define a function to apply the transition based on probabilities
def apply_transition(land_use, transition_probs):
    new_land_use = np.copy(land_use)
    rows, cols = land_use.shape
    for row in range(rows):
        for col in range(cols):
            current_category = int(land_use[row, col])
            if 1 <= current_category <= 5:
                transition_probs_normalized = transition_probs[current_category - 1]
                new_category = np.argmax(np.random.multinomial(1, transition_probs_normalized))
                new_land_use[row, col] = new_category + 1
    return new_land_use

# Apply the transition to the 2021 land use map
predicted_land_use_2021 = apply_transition(current_land_use_map_2016, transition_probs)

# Get the shape of the predicted_land_use_2021 array
rows, cols = predicted_land_use_2021.shape

# Save the predicted land use map for 2021 with spatial reference information
output_raster_path = os.path.join(data_directory, 'predicted_land_use_2021.tif')
with rasterio.open(output_raster_path, 'w', driver='GTiff', width=cols, height=rows, count=1, dtype=rasterio.int32, crs=src.crs, transform=transform) as dst:
    dst.write(predicted_land_use_2021, 1)

print("Prediction completed. The result is saved to:", output_raster_path)


###################  Repeat the same script for all time-steps needed  ##################
###########################  e.g.   2021 to 2026,  etc.  ################################
# It is recommended that you paste the above script as many times as necessary
# rather than always editing the one above, having thus a 'secure' template for the process.



###########################  VALIDATION   ##########################################

import geopandas as gpd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
import numpy as np

# Paths to truth and predicted point feature classes
truth_path = r'your\path\validactual21.shp'
predicted_path = r'your\path\validpred21.shp'


# Load the truth and predicted point feature classes
truth_data = gpd.read_file(truth_path)
predicted_data = gpd.read_file(predicted_path)

# Round spatial coordinates to 6 decimal places to ensure correct merging
truth_data['rounded_x'] = truth_data.geometry.x.round(6)
truth_data['rounded_y'] = truth_data.geometry.y.round(6)
predicted_data['rounded_x'] = predicted_data.geometry.x.round(6)
predicted_data['rounded_y'] = predicted_data.geometry.y.round(6)

# Generate a unique identifier based on spatial coordinates to ensure that the same points are compared
truth_data['unique_id'] = truth_data.geometry.apply(lambda geom: f'{geom.x:.6f}_{geom.y:.6f}')
predicted_data['unique_id'] = predicted_data.geometry.apply(lambda geom: f'{geom.x:.6f}_{geom.y:.6f}')

# Merge the datasets based on the 'unique_id' column
merged_data = truth_data.merge(predicted_data, on='unique_id', how='inner') #merge on geometry or on unique_id or on pointid

# Check if the datasets have the same number of points
if len(truth_data) != len(predicted_data):
    print("Warning: The number of points in truth and predicted datasets is not the same.")
    
# Check if there are common points
if len(merged_data) == 0:
    print("Error: There are no common points between the truth and predicted datasets.")
else:
    # Calculate accuracy metrics
    truth_labels = merged_data['grid_code_x'].astype(int)
    predicted_labels = merged_data['grid_code_y'].astype(int)
    
    
# Check the datasets and their column names in the data 
print(truth_data)
print(predicted_data)
print(merged_data)
# To ensure that we are comparing the correct ones and the merged file looks as expected
print(truth_data.columns)
print(predicted_data.columns)
print(merged_data.columns)


# Plot the truth_data and predicted_data points together, to ensure they are the same points
import matplotlib.pyplot as plt
ax = truth_data.plot(color='blue', label='Truth Data')
predicted_data.plot(ax=ax, color='red', label='Predicted Data')
plt.legend()
plt.show()

# Accuracy metrics
accuracy = accuracy_score(truth_labels, predicted_labels)
mae = mean_absolute_error(truth_labels, predicted_labels)
rmse = mean_squared_error(truth_labels, predicted_labels, squared=False)  # Use squared=False for RMSE
kappa = cohen_kappa_score(truth_labels, predicted_labels)
confusion = confusion_matrix(truth_labels, predicted_labels)

# Calculate precision, recall, and F1-score
classification_rep = classification_report(truth_labels, predicted_labels)

# Print the accuracy metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("Cohen's Kappa:", kappa)

print("Confusion Matrix:")
print(confusion)

# Print precision, recall, and F1-score
print("Classification Report:")
print(classification_rep)




######################  PLOT THE MAPS   ################################

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd

# Paths to the shapefiles for each year
shapefile_paths = [
    r'your\path\predicted26polygons.shp',
    r'your\path\predicted31polygons.shp',
    r'your\path\predicted36polygons.shp',
    r'your\path\predicted41polygons.shp',
    r'your\path\predicted46polygons.shp',
    r'Dyour\path\predicted51polygons.shp',
]

# Corresponding years
years = [2026, 2031, 2036, 2041, 2046, 2051]

# Land use categories and their colors
land_use_colors = {
    0: 'white',
    1: 'blue',
    2: 'gray',
    3: 'black',
    4: 'darkred',
    5: 'lightgreen',
}


# Land use labels for the legend
land_use_labels = {
    0: 'No data',
    1: 'Water',
    2: 'Urban',
    3: 'Barren Land',
    4: 'Forest',
    5: 'Crops',
}

# Create a figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loop through shapefiles and corresponding years
for i, (shapefile_path, year) in enumerate(zip(shapefile_paths, years)):
    # Read the shapefile using GeoPandas
    gdf = gpd.read_file(shapefile_path)

    # Get the axis for the current subplot
    ax = axes[i // 3, i % 3]

    # Plot the GeoDataFrame with specified colors and legend
    for land_use_code, color in land_use_colors.items():
        gdf[gdf['gridcode'] == land_use_code].plot(ax=ax, color=color)

    # Set title and remove axes
    ax.set_title(f'Year {year}', fontsize=16, fontweight='bold')
    ax.axis('off')

# Add a single legend for the 2051 map
ax_legend = axes[1, 2].inset_axes([0.85, 0.05, 0.35, 0.25])
ax_legend.axis('off')
for land_use_code, color in land_use_colors.items():
    land_use_label = {
        1: 'Water',
        2: 'Urban',
        3: 'Barren Land',
        4: 'Forest',
        5: 'Crops',
    }.get(land_use_code, f'Land Use {land_use_code}')
    ax_legend.add_patch(plt.Rectangle((0, (land_use_code - 1) * 0.2), 0.2, 0.2, color=color))
    ax_legend.annotate(land_use_label, (0.25, (land_use_code - 1) * 0.2), fontsize=12)

# Adjust subplot layout and remove gaps between subplots
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

# Show the figure
plt.show()




#########################  PLOT THE PREDICTED AREAS  ############################

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

# Paths to the shapefiles for each year
shapefile_paths = [
    r'your\path\predicted26polygons.shp',
    r'your\path\predicted31polygons.shp',
    r'your\path\predicted36polygons.shp',
    r'your\path\predicted41polygons.shp',
    r'your\path\predicted46polygons.shp',
    r'Dyour\path\predicted51polygons.shp',
    ]

# Corresponding years
years = [2026, 2031, 2036, 2041, 2046, 2051]

# Land use categories
land_use_categories = {
    0: 'No data',
    1: 'Water',
    2: 'Urban',
    3: 'Barren Land',
    4: 'Forest',
    5: 'Crops',
}

# Create an empty DataFrame to store land use category counts over time
land_use_counts = pd.DataFrame(columns=land_use_categories.values())

# Loop through shapefiles and corresponding years
for shapefile_path, year in zip(shapefile_paths, years):
    # Read the shapefile using GeoPandas
    gdf = gpd.read_file(shapefile_path)

    # Count the number of pixels for each land use category
    category_counts = {land_use_categories[i]: (gdf['gridcode'] == i).sum() for i in land_use_categories.keys()}

    # Append the counts to the DataFrame
    land_use_counts = land_use_counts.append(category_counts, ignore_index=True)

# Set up the plot
plt.figure(figsize=(10, 6))

# Define the bar colors
colors = [land_use_colors[col] for col in land_use_counts.columns]

# Create the bar plot
land_use_counts.plot(kind='bar', stacked=True, color=colors)

plt.title("Land Use Evolution Over Time")
plt.xlabel("Year")
plt.ylabel("Land Use Area (Number of Pixels)")

# Show the plot
plt.show()


####################  Plot the areas as stacked bars ###########################

import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from your data  (replace with the predicted areas)
data = {
    'Year': [2011, 2016, 2021, 2026, 2031, 2036, 2041, 2046, 2051],
    'Water': [58.67, 58.60, 58.73, 58.62, 58.72, 58.74, 58.77, 58.81, 58.75],
    'Urban': [79.89, 81.33, 82.66, 84.29, 85.83, 87.39, 88.90, 90.55, 90.55],
    'Barren Land': [0.39, 0.40, 1.15, 1.71, 1.65, 1.59, 1.55, 1.53, 1.58],
    'Forest': [80.07, 79.52, 79.20, 78.86, 78.20, 77.54, 76.86, 76.16, 76.08],
    'Crops': [478.85, 478.01, 476.13, 474.38, 473.46, 472.60, 471.79, 470.83, 470.91]
}

df = pd.DataFrame(data)

# Create the stacked bar plot
plt.figure(figsize=(10, 6))

# Define colors for the land use classes
colors = ['blue', 'gray', 'orange', 'red', 'lightgreen']

# Initialize the bottom values for the bars
bottom = [0] * len(df['Year'])

# Plot stacked bars for each land use class
for i, col in enumerate(df.columns[1:]):
    plt.bar(df['Year'], df[col], label=col, color=colors[i], bottom=bottom)
    bottom = [bottom[j] + df[col][j] for j in range(len(df['Year']))]

# Add labels, title, legend, and grid
plt.xlabel('Year')
plt.ylabel('Areas (km2)')
plt.title('Land Uses Over Time')
plt.legend(loc='upper left')
plt.grid(True)

# Set the x-axis ticks
plt.xticks(df['Year'])

# Show the plot
plt.show()
