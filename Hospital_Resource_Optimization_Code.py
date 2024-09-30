# Hospital Resource Optimization Using Predictive Modeling and Linear Programming

# Objective:
# This case study aims to optimize hospital resources (beds and staff) based on predicted patient length of stay (LOS).
# The analysis uses XGBoost for length of stay prediction and linear programming for hospital resource optimization 
# to improve patient outcomes and streamline hospital operations.

# Key Points:
# - End-to-end analysis covering data preprocessing, predictive modeling (XGBoost), and hospital resource optimization.
# - Dataset: Hospital admissions data, including features such as patient demographics and health conditions.
# - Techniques: XGBoost for length of stay prediction and linear programming for hospital resource optimization.
# - Performance Evaluation: The XGBoost model achieved an R² score of 0.2524.
# - Insights: Facility E had the highest resource demand, requiring 90 beds and 44 staff, while other facilities required significantly fewer resources.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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

# Step 3: Feature Selection and Engineering
# Select the features and target variable for predicting length of stay (LOS)
features = ['gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum',
            'psychologicaldisordermajor', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 
            'pulse', 'respiration', 'secondarydiagnosisnonicd9']
df['gender'] = df['gender'].map({'F': 0, 'M': 1})  # Gender mapping for model processing
X = df[features]
y = df['lengthofstay']

# Apply log transformation to length of stay if skewed
if df['lengthofstay'].skew() > 1:
    y = np.log(df['lengthofstay'] + 1)  # Add 1 to avoid log(0)

# Step 4: Data Preprocessing
# Convert data types to reduce memory usage
X = X.astype(np.float32)

# Split the data into training and testing sets and standardize the numerical features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data to ensure uniformity for model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Predictive Modeling (XGBoost)
# Train an XGBoost model to predict length of stay (LOS) to help hospitals with resource allocation
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Step 6: Model Evaluation
# Evaluate the model's performance using Mean Squared Error (MSE) and R² Score
y_pred_xgb = xgb_model.predict(X_test_scaled)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost Model Performance:\nMSE: {mse_xgb}\nR² Score: {r2_xgb}")

# Feature importance for XGBoost (helps identify key factors affecting length of stay)
importance = xgb_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features)
plt.title('Feature Importance for XGBoost Model (Impact on LOS)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(r"C:\Users\JUDIT\Desktop\Data Sets\xgb_feature_importance.png")
plt.show()

# Residual plot to check model errors (useful for identifying if there's bias in predictions)
residuals = y_test - y_pred_xgb
sns.residplot(x=y_pred_xgb, y=residuals)
plt.title("Residual Plot (Model Errors)")
plt.savefig(r"C:\Users\JUDIT\Desktop\Data Sets\xgb_residual_plot.png")
plt.show()

# Q-Q plot to assess if residuals are normally distributed
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot (Check for Normality in Residuals)")
plt.savefig(r"C:\Users\JUDIT\Desktop\Data Sets\xgb_qq_plot.png")
plt.show()

# Step 7: Hospital Resource Optimization Using Linear Programming
# Define the number of available beds and staff for optimization based on LOS predictions
beds = 100
staff = 50

# Predict the length of stay for the entire dataset in smaller chunks to avoid memory errors
chunk_size = 1000  # Process 1,000 rows at a time
predicted_los = []
for i in range(0, X.shape[0], chunk_size):  # Loop through chunks
    X_chunk = X[i:i + chunk_size]
    scaled_chunk = scaler.transform(X_chunk)
    predicted_los_chunk = xgb_model.predict(scaled_chunk)
    predicted_los.extend(predicted_los_chunk)

predicted_los = np.array(predicted_los)

# Create variables for the number of beds and staff allocated to each facility
facilities = df['facid'].unique()
facility_beds = LpVariable.dicts("Beds", facilities, lowBound=0, cat='Integer')
facility_staff = LpVariable.dicts("Staff", facilities, lowBound=0, cat='Integer')

# Define the linear programming problem for hospital resource optimization
problem = LpProblem("Resource_Optimization", LpMaximize)

# Maximize the total allocation of beds and staff across all facilities to improve patient care
problem += lpSum([facility_beds[i] + facility_staff[i] for i in facilities])

# Set the constraints for total beds and staff available
problem += lpSum([facility_beds[i] for i in facilities]) <= beds
problem += lpSum([facility_staff[i] for i in facilities]) <= staff

# Add constraints for facility bed and staff allocation based on average LOS predictions
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
# The XGBoost model predicted patient length of stay with an R² score of 0.2524.
# Linear programming was used to allocate beds and staff efficiently across facilities.
# Facility E required the most resources, receiving 90 beds and 44 staff, while other facilities required significantly fewer resources.
# This analysis demonstrates the potential for optimizing hospital resources using predictive modeling and optimization techniques.

# Future Work:
# - Model Improvement: Further exploration of more advanced models, such as Gradient Boosting or neural networks, could improve the prediction accuracy for patient length of stay.
# - Feature Engineering: Incorporating additional features, such as patient readmission history or hospital department information, could enhance the predictive power of the model.
# - Real-Time Resource Management: Integrating real-time data streams from hospital systems could enable dynamic adjustment of resource allocation based on actual patient loads.
# - Scenario Analysis: Additional scenario analyses could simulate varying levels of resource availability (e.g., during flu season or pandemic spikes) to ensure hospitals are prepared for surges in patient demand.
