# Hospital Resource Optimization Using Predictive Modeling and Linear Programming

## Objective:
Develop a predictive model to estimate patient length of stay and use linear programming to optimize the allocation of hospital resources (beds and staff) across multiple facilities. The focus is on maximizing operational efficiency and patient care by ensuring that resources are allocated in proportion to patient demand.

## Key Points:
- End-to-end analysis covering data preprocessing, predictive modeling, and resource optimization.
- **Dataset**: Hospital admissions data, including patient demographics and health conditions.
- **Techniques**: Random Forest for predicting patient length of stay and linear programming for resource optimization.
- **Performance Evaluation**: The Random Forest model achieved an R² score of 0.217, with further optimization of resources using linear programming.
- **Insights**: Facility E had the highest demand, requiring 90 beds and 44 staff, while other facilities had significantly lower resource requirements.

---

## Dataset Description:
### Source:
Hospital admissions data including key patient metrics and resource needs.

### Features:
- **Patient Metrics**: Gender, dialysis status, asthma, iron deficiency, psychological disorders, glucose levels, blood urea nitrogen, creatinine, BMI, pulse, respiration rate, and secondary diagnosis.
- **Target Variable**: Length of stay (number of days in the hospital).

---

## Methodology Overview:
### Data Cleaning and Preprocessing:
- **Datetime Conversion**: Converted admission and discharge dates to datetime format for analysis.
- **Missing Values**: Replaced missing values using median imputation for numerical variables and mode imputation for categorical variables.
- **Scaling**: Standardized the numerical features using StandardScaler to normalize the data for predictive modeling.
- **Data Split**: Split the dataset into 80% training and 20% testing sets for model evaluation.

### Exploratory Data Analysis:
- **Feature Distributions**: Visualized distributions of key patient features and their relationship to length of stay.
- **Correlation Analysis**: Investigated correlations between patient characteristics and length of stay to identify important predictors.

### Model Development:
#### Random Forest (Predictive Model for Length of Stay):
- Used Random Forest as the predictive model for estimating patient length of stay based on key health metrics.
- Applied hyperparameter tuning to find the optimal number of estimators for the model.

#### Linear Programming (Resource Optimization):
- Developed a linear programming model to optimize the allocation of hospital resources (beds and staff) based on predicted patient length of stay.
- Set constraints on the total number of beds and staff to ensure efficient resource distribution across multiple facilities.

---

## Key Findings:

### Model Performance:
- **Random Forest**:
  - **R² Score**: 0.217
  - **Mean Squared Error**: 4.29
  - The model captures some variability in the length of stay but leaves room for further improvement.

### Resource Optimization Results:
- Facility E had the highest demand, requiring **90 beds** and **44 staff**.
- Other facilities (A, B, C, D) required significantly fewer resources (between 2 and 3 beds, and 1-2 staff each).

### Significant Predictors:
- **Blood Urea Nitrogen** and **BMI** were key predictors of patient length of stay, indicating patients with higher levels often require longer hospital stays.
- **Respiration rate** and **pulse** were also identified as contributing factors, reflecting patients with more severe conditions.

---

## Visualizations:
- **Bar Chart**: Shows the optimized allocation of hospital beds and staff across different facilities.
- **Resource Summary**: Provides a clear view of how resources are distributed to ensure operational efficiency.

---

## Business Impact:
This analysis enables healthcare administrators to optimize hospital resources based on predicted patient length of stay, leading to:
- **Improved Resource Allocation**: Ensuring that each facility has the appropriate number of beds and staff to meet patient demand, preventing over or under-utilization of resources.
- **Operational Efficiency**: Helping hospitals plan more effectively and reduce unnecessary costs by aligning staff and bed allocation with patient needs.
- **Enhanced Patient Care**: Ensuring that resources are available when needed, leading to more timely patient care and reduced wait times.

---

## Future Work Recommendations:
### Model Refinement:
- **Model Improvement**: Further optimization of the predictive model (e.g., using XGBoost or Gradient Boosting) to increase accuracy in predicting length of stay.
- **Feature Engineering**: Introduce additional features such as patient readmission history or hospital department information to improve predictive performance.

### Scenario Analysis:
- Conduct additional scenario analyses by varying the number of available beds and staff to simulate different demand scenarios and ensure preparedness during peak times (e.g., flu season or COVID-19 spikes).

### Data Integration:
- **Real-Time Implementation**: Integrate this model with real-time data from hospital management systems to allow for continuous adjustment of resource allocation based on current patient loads.

---

## Ethical Considerations:
- **Fairness in Resource Allocation**: Ensure that the model's recommendations do not disproportionately favor certain facilities or patient groups, leading to inequities in care. Regular monitoring and updates will ensure the model remains unbiased and accurate over time.
