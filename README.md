# Hospital Resource Optimization Using Predictive Modeling and Linear Programming

## Objective:
Develop a predictive model to estimate patient length of stay and use linear programming to optimize the allocation of hospital resources (beds and staff) across multiple facilities. The focus is on maximizing operational efficiency and improving patient care by ensuring that resources are allocated in proportion to predicted patient demand.

## Key Points:
- End-to-end analysis covering data preprocessing, predictive modeling, and resource optimization.
- **Dataset**: Hospital admissions data, including patient demographics and health conditions.
- **Techniques**: Random Forest for predicting patient length of stay and linear programming for resource optimization.
- **Performance Evaluation**: The Random Forest model achieved an R² score of 0.217. Resources were optimized based on predicted length of stay using linear programming.
- **Insights**: Facility E had the highest demand, requiring 90 beds and 44 staff, while other facilities had significantly lower resource requirements.

## Dataset Description:
### Source:
Hospital admissions data including key patient metrics and resource needs.

### Features:
- **Patient Metrics**: Gender, dialysis status, asthma, iron deficiency, psychological disorders, glucose levels, blood urea nitrogen, creatinine, BMI, pulse, respiration rate, and secondary diagnosis.
- **Target Variable**: Length of stay (number of days in the hospital).

## Methodology Overview:

### Data Cleaning and Preprocessing:
- **Datetime Conversion**: Converted admission and discharge dates to datetime format for analysis.
- **Missing Values**: Replaced missing values using median imputation for numerical variables and mode imputation for categorical variables.
- **Scaling**: Standardized the numerical features using `StandardScaler` to normalize the data for predictive modeling.
- **Data Split**: Split the dataset into 80% training and 20% testing sets for model evaluation.

### Predictive Modeling:
- **Random Forest**: Used Random Forest as the predictive model to estimate patient length of stay based on health metrics.
- **Hyperparameter Tuning**: Applied GridSearchCV to fine-tune the model's hyperparameters for improved performance.

### Linear Programming for Resource Optimization:
- **Objective**: Linear programming was applied to optimize the allocation of hospital resources (beds and staff) across multiple facilities, ensuring that resources met patient demand.

## Key Findings:

### Model Performance:
- **Random Forest**:
  - **R² Score**: 0.217
  - **Mean Squared Error**: 4.29
  - The model captures some variability in length of stay but leaves room for further improvement.

### Resource Optimization Results:
- Facility E had the highest demand, requiring **90 beds** and **44 staff**.
- Other facilities (A, B, C, D) required significantly fewer resources (between 2 and 3 beds, and 1-2 staff each).

### Significant Predictors:
- **Blood Urea Nitrogen** and **BMI** were key predictors of patient length of stay, indicating patients with higher levels often require longer hospital stays.
- **Respiration rate** and **pulse** were also identified as contributing factors, reflecting patients with more severe conditions.

## Visualizations:
- **Bar Chart**: Shows the optimized allocation of hospital beds and staff across different facilities.

## Business Impact:
This analysis enables healthcare administrators to optimize hospital resources based on predicted patient length of stay, leading to:
- **Improved Resource Allocation**: Ensuring that each facility has the appropriate number of beds and staff to meet patient demand, preventing over or under-utilization of resources.
- **Operational Efficiency**: Helping hospitals plan more effectively and reduce unnecessary costs by aligning staff and bed allocation with patient needs.
- **Enhanced Patient Care**: Ensuring that resources are available when needed, leading to more timely patient care and reduced wait times.

## Future Work Recommendations:
- **Model Refinement**: Explore improving the predictive model using advanced algorithms like XGBoost to increase accuracy in predicting length of stay.
- **Scenario Analysis**: Simulate various demand scenarios, such as peak times during flu season or pandemic surges, to prepare for high patient volume.
- **Real-Time Implementation**: Integrate this model with real-time data from hospital management systems to continuously adjust resource allocation based on current patient loads.
