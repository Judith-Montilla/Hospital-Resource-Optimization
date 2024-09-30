# Hospital Resource Optimization Using Predictive Modeling and Linear Programming

## Objective
The primary objective of this project is to optimize hospital resources, such as beds and staff, based on predicted patient length of stay (LOS). By developing a predictive model using XGBoost and applying linear programming, this project aims to help hospital administrators allocate resources efficiently to improve patient outcomes and operational efficiency. The goal is to predict patient length of stay based on patient demographics and health conditions, and use those predictions to optimize resource distribution across facilities.

## Dataset Description
- **Source:** Hospital Admissions Dataset (fictional for this case study)
- **Features:**
  - **Demographics:** Gender
  - **Health Metrics:** BMI, dialysis status, asthma, iron deficiency, psychological disorder, glucose, blood urea nitrogen, creatinine, pulse, respiration
  - **Diagnosis Codes:** Secondary diagnosis codes
- **Outcome:** Length of stay (target variable).

## Methodology Overview

### Data Cleaning and Preprocessing
- Converted admission and discharge dates to datetime format.
- Handled missing values by imputing the median for numerical variables and the mode for categorical variables.
- Transformed categorical variables (gender) to numerical format.
- Applied log transformation to length of stay (LOS) to handle skewness.
- Standardized numerical features to improve model performance.

### Predictive Modeling
- Implemented an **XGBoost regression** model to predict patient length of stay.
- Evaluated model performance using Mean Squared Error (MSE) and R² score.
- Visualized residual plots and Q-Q plots to assess normality and model assumptions.
- Created a feature importance plot to show which features are the most influential in predicting length of stay.

### Optimization Using Linear Programming
- Developed a linear programming model to allocate hospital resources (beds and staff) based on predicted patient length of stay.
- The model aimed to maximize resource allocation efficiency across multiple hospital facilities, subject to constraints on total available resources.
- The optimized bed and staff allocation per facility was calculated based on the average predicted length of stay.

## Model Performance

| Metric                 | Value          |
|------------------------|----------------|
| Mean Squared Error (MSE)| 4.1023         |
| R² Score               | 0.2524         |

### Feature Importance Visualization
The XGBoost model's feature importance plot highlights the key factors that influence patient length of stay:

- **BMI**, **creatinine**, and **psychological disorders** are among the most important predictors of length of stay.
- This information is valuable for hospital administrators to identify which patient characteristics are likely to result in longer stays, allowing for proactive planning and interventions.

## Resource Allocation Results

| Facility | Optimized Bed Allocation | Optimized Staff Allocation |
|----------|--------------------------|----------------------------|
| A        | 2 beds                   | 1 staff                    |
| B        | 2 beds                   | 1 staff                    |
| C        | 3 beds                   | 2 staff                    |
| D        | 3 beds                   | 2 staff                    |
| E        | 90 beds                  | 44 staff                   |

### Optimization Insights
- **Facility E** has the highest demand, requiring the most beds and staff based on the predicted length of stay.
- Other facilities received fewer resources, reflecting their lower patient demand.
- This optimized resource allocation helps hospitals improve patient outcomes by reducing wait times and ensuring that sufficient resources are available in high-demand facilities.

## Business Impact
The predictive insights and optimization results generated by this project can help hospitals:
1. **Improve patient outcomes**: By efficiently allocating resources based on length of stay predictions, hospitals can ensure adequate staffing and bed availability, reducing wait times and improving patient care.
2. **Optimize operational efficiency**: Hospital administrators can use these predictions to better manage resource distribution, particularly during periods of high demand (e.g., flu season, pandemics).
3. **Cost reduction**: Efficient resource management can lead to cost savings by reducing the need for emergency staffing and resource shortages.

## Collaborative Experience
This model can be a valuable tool for **cross-functional healthcare teams**, such as **hospital administrators** and **clinical staff**, to make **data-driven decisions** about resource planning. By predicting patient demand and optimizing hospital resources, administrators can ensure that their facilities are well-equipped to handle patient needs efficiently, ultimately improving operational performance and patient satisfaction.

## Future Work Recommendations

- **Advanced Models**: Explore more advanced models such as **Gradient Boosting** or **neural networks** to capture more complex relationships between patient features and length of stay.
- **Real-Time Data Integration**: Integrate real-time data from hospital management systems to dynamically adjust resource allocation based on actual patient flow.
- **Scenario Analysis**: Simulate high-demand scenarios (e.g., flu season or pandemic spikes) to ensure hospitals are prepared to handle surges in patient admissions.
- **Feature Expansion**: Incorporate additional patient metrics, such as comorbidities or prior hospitalization history, to improve prediction accuracy and resource planning.

## Ethical Considerations

- **Data Privacy**: While the dataset used in this project is fictional, in real-world applications, it's critical to ensure compliance with healthcare regulations such as **HIPAA** to protect patient data privacy.
- **Fairness**: The model's predictions should be evaluated to avoid biases based on gender, race, or socioeconomic status, ensuring that all patients receive equitable care and resource allocation.
