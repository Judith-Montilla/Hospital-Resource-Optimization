# Hospital Resource Optimization Using Predictive Modeling and Linear Programming

# Objective:
# This case study aims to optimize hospital resources (beds and staff) based on predicted patient length of stay.
# The analysis uses Random Forest for length of stay prediction and linear programming for resource optimization across multiple facilities.

# Key Points:
# - End-to-end analysis covering data preprocessing, predictive modeling (Random Forest), and resource optimization using linear programming.
# - Dataset: Hospital admissions data, including features such as patient demographics and health conditions.
# - Techniques: Random Forest for length of stay prediction and linear programming for resource optimization.
# - Performance Evaluation: The Random Forest model achieved an R² score of 0.217. 
# - Insights: Facility E had the highest resource demand, requiring 90 beds and 44 staff, while other facilities required significantly fewer resources.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = r"C:\Users\JUDIT\Desktop\Data Sets\LengthOfStay.csv"
df = pd.read_csv(file_path)

# Step 2: Data Cleaning
# Convert admission and discharge dates to datetime format
df['vdate'] = pd.to_datetime(df['vdate'], errors='coerce')
df['discharged'] = pd.to_datetime(df['discharged'], errors='coerce')

# Handle missing values by using median for numerical variables and mode for categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Step 3: Feature Selection for Predictive Modeling
# Select the features and target variable for predicting length of stay
features = ['gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum',
            'psychologicaldisordermajor', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 
            'pulse', 'respiration', 'secondarydiagnosisnonicd9']
df['gender'] = df['gender'].map({'F': 0, 'M': 1})
X = df[features]
y = df['lengthofstay']

# Step 4: Data Preprocessing
# Split the data into training and testing sets and standardize the numerical features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Predictive Modeling
# Train a Random Forest model to predict length of stay
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Step 6: Model Evaluation
# Evaluate the model's performance
y_pred = rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance:\nMSE: {mse}\nR² Score: {r2}")

# Step 7: Resource Optimization Using Linear Programming
# Define the number of available beds and staff for optimization
beds = 100
staff = 50

# Predict the length of stay for the entire dataset
predicted_los = rf_model.predict(scaler.transform(X))

# Create variables for the number of beds and staff allocated to each facility
facilities = df['facid'].unique()
facility_beds = LpVariable.dicts("Beds", facilities, lowBound=0, cat='Integer')
facility_staff = LpVariable.dicts("Staff", facilities, lowBound=0, cat='Integer')

# Define the linear programming problem
problem = LpProblem("Resource_Optimization", LpMaximize)

# Maximize the total allocation of beds and staff across all facilities
problem += lpSum([facility_beds[i] + facility_staff[i] for i in facilities])

# Set the constraints for total beds and staff available
problem += lpSum([facility_beds[i] for i in facilities]) <= beds
problem += lpSum([facility_staff[i] for i in facilities]) <= staff

# Ensure that each facility receives at least half the average predicted length of stay in beds, and a quarter in staff
for i in facilities:
    avg_los_facility = np.mean(predicted_los[df['facid'] == i])
    problem += facility_beds[i] >= avg_los_facility / 2
    problem += facility_staff[i] >= avg_los_facility / 4

# Solve the linear programming problem
problem.solve()

# Display the optimized allocation of beds and staff
print("Optimized Bed Allocation:")
for i in facilities:
    print(f"Facility {i}: {facility_beds[i].varValue} beds")
print("Optimized Staff Allocation:")
for i in facilities:
    print(f"Facility {i}: {facility_staff[i].varValue} staff")

# Step 8: Visualization
# Create a summary DataFrame with the optimized resource allocation
summary_df = pd.DataFrame({
    'Facility': facilities,
    'Optimized Beds': [facility_beds[i].varValue for i in facilities],
    'Optimized Staff': [facility_staff[i].varValue for i in facilities]
})

# Plot the resource allocation across facilities
plt.figure(figsize=(10, 6))
summary_df.plot(x='Facility', y=['Optimized Beds', 'Optimized Staff'], kind='bar')
plt.title("Optimized Hospital Resource Allocation")
plt.xlabel("Facility")
plt.ylabel("Number of Resources")

# Save the plot as an image
output_image_path = r"C:\Users\JUDIT\Desktop\Data Sets\optimized_hospital_resource_allocation.png"
plt.savefig(output_image_path)
plt.show()

# Step 9: Conclusion:
# The Random Forest model predicted patient length of stay with an R² score of 0.217.
# Linear programming was used to allocate beds and staff efficiently across facilities.
# Facility E required the most resources, receiving 90 beds and 44 staff, while other facilities required significantly fewer resources.
# This analysis demonstrates the potential for optimizing hospital resources using predictive modeling and optimization techniques.

# Future Work:
# - Model Improvement: Further exploration of more advanced models, such as XGBoost or Gradient Boosting, could improve the prediction accuracy for patient length of stay.
# - Feature Engineering: Incorporating additional features, such as patient readmission history or hospital department information, could enhance the predictive power of the model.
# - Real-Time Resource Management: Integrating real-time data streams from hospital systems could enable dynamic adjustment of resource allocation based on actual patient loads.
# - Scenario Analysis: Additional scenario analyses could simulate varying levels of resource availability (e.g., during flu season or COVID-19 spikes) to ensure hospitals are prepared for surges in patient demand.
