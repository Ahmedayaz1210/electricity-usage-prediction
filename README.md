# Predicting Electricity Usage

## Project Dashboard
[![Dashboard](https://img.shields.io/badge/Visit-Dashboard-blue?style=for-the-badge&logo=vercel)](https://electricity-usage-prediction.vercel.app/)

## Project Presentation
[![Project Presentation PDF](https://img.shields.io/badge/View-Presentation-green?style=for-the-badge&logo=github)](https://github.com/Ahmedayaz1210/electricity-usage-prediction/blob/main/final_notebook%26presentation/Electricity-Usage-Prediction-Presentation.pdf)

[![Project Presentation Video (Modeling and Prediction)](https://img.shields.io/badge/View-Presentation-green?style=for-the-badge&logo=youtube)](https://youtu.be/Yx8KJ44wufU)

By: Ahmed Ayaz, Krish Patel & Dylan Patel
Date: 12/2/2024

## Important Note
This project is contained entirely in a single Jupyter notebook: `final_project/Electricity_Usage_Prediction.ipynb`. All code, analysis, visualizations, and detailed documentation can be found in this notebook. Please refer to this file for the complete project implementation.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Data Analysis](#data-analysis)
4. [Modeling and Prediction](#modeling-and-prediction)
5. [Conclusion and Insights](#conclusion-and-insights)
6. [References](#references)

## 1. Introduction

### 1.1 Project Overview
This project analyzes the impact of weather conditions—including temperature, snow, and precipitation—on electricity usage across all 50 U.S. states. Using historical weather data and electricity usage information, we investigate correlations between environmental factors and electricity demand to build predictive models for future electricity usage.

### 1.2 Key Questions
1. How do weather factors such as temperature, snow, and precipitation influence electricity usage across different states?
2. What is the relationship between weather conditions and electricity pricing trends over time?
3. How accurately can future electricity usage be predicted based on historical weather data?

### 1.3 Brief Process
1. **Data Gathering**: Collection of weather data and electricity usage data for all 50 states
2. **Data Preparation**: Cleaning, preprocessing, and integration of weather and electricity data
3. **Data Analysis**: Exploration of trends and correlations between weather conditions and electricity metrics
4. **Modeling and Prediction**: Development of predictive models for electricity usage
5. **Evaluation and Insights**: Evaluation of model performance and insights generation

### 1.4 Project Objective
The primary objective is to build a robust model that accurately predicts electricity usage based on weather patterns. This analysis helps identify key weather factors that drive electricity demand, supporting better resource planning strategies.

### 1.5 Project Importance
Understanding the link between weather conditions and electricity demand is crucial for utility companies, policymakers, and energy providers. Accurate forecasting can help optimize energy distribution and guide sustainable energy policies.

## 2. Data Preparation

### 2.1 Data Sources
1. **Electricity Dataset**: 
   - Source: Kaggle's Electricity Prices Dataset
   - Content: Detailed information on electricity prices across all sectors and U.S. states
   - Time Period: 2001-2024

2. **Weather Dataset**:
   - Source: Meteostat Python Library
   - Content: Daily average temperature, precipitation, and snowfall data
   - Coverage: All major U.S. cities

3. **City Information Dataset**:
   - Source: SimpleMaps U.S. Cities Data
   - Content: Geographic information (latitude, longitude, population)
   - Purpose: Location data for weather data retrieval

### 2.2 Data Cleaning
1. **Electricity Usage Data Cleaning**:
   - Missing value handling
   - Column standardization
   - Relevant column filtering

2. **Weather Data Cleaning**:
   - Temperature conversion (Celsius to Fahrenheit)
   - Outlier removal
   - Date format standardization

3. **Data Integration Preparation**:
   - Structure compatibility verification
   - Temporal scale alignment
   - Granularity matching

### 2.3 Feature Engineering
1. **Time-Based Features**:
   - Season categorization
   - Quarterly representation

2. **Weather Features**:
   - Temperature range calculation
   - Precipitation intensity categories
   - Binary weather indicators:
     - `is_high_temp`: Temperature > 80°F
     - `is_low_temp`: Temperature < 32°F
     - `has_precipitation`: Precipitation > 0
     - `is_high_precipitation`: Precipitation > 1 inch
     - `has_snowfall`: Snowfall > 0
     - `is_heavy_snowfall`: Snowfall > 5 inches

3. **Electricity Usage Metrics**:
   - Per capita usage calculation
   - Price-to-sales ratio

## 3. Data Analysis

### 3.1 Exploratory Data Analysis Findings
1. **Customer Distribution**:
   - Average: ~3 million customers
   - High standard deviation indicating substantial variation
   - Range influenced by state population sizes

2. **Price Analysis**:
   - Range: 3.78 to 42.76 cents per kwh
   - Average: ~10 cents per kwh
   - Significant state-specific variations

3. **Temperature Distribution**:
   - Mean: 54°F
   - Range: -3.7°F to 90.25°F
   - Balanced seasonal distribution

### 3.2 Seasonal and Weather Impact Analysis
1. **Usage Patterns**:
   - Highest usage in Summer
   - Lower usage in Spring
   - Winter usage varies by region

2. **Temperature Impact**:
   - Strong inverse relationship with usage
   - Peak usage during extreme temperatures
   - Regional variation in temperature sensitivity

## 4. Modeling and Prediction

### 4.1 Model Selection: XGBoost
Selected for its advantages:
- Handling non-linear relationships
- Robustness to overfitting
- Speed and efficiency
- Feature importance insights
- Hyperparameter tuning flexibility

### 4.2 Model Configuration
```python
XGBRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    eval_metric='rmse'
)
```

### 4.3 Modeling Approaches

#### 4.3.1 Initial Approach
Performance varied by sales volume:
- Small amounts (500-1000): 9.8% error
- Medium amounts (3,000-9,000): 3.6% error
- Large amounts (>10,000): 23.6% error

#### 4.3.2 Log Transform Approach
Implemented to handle scale differences:
- Reduced scale difference between small and large values
- Improved model's handling of large values
- Significant RMSE improvement:
  - Training: 0.98266 → 0.07672
  - Validation: 0.98030 → 0.09716

#### 4.3.3 Final Stratified Approach
Population-based strategy results:

1. **Small Population States**:
   - Average Error: 13.65%
   - Median Error: 10.24%
   - Error Range: 0.06% - 54.96%

2. **Medium Population States**:
   - Average Error: 9.05%
   - Median Error: 7.51%
   - Error Range: 0.06% - 49.22%

3. **Large Population States**:
   - Average Error: 10.58%
   - Median Error: 7.58%
   - Error Range: 0.05% - 51.77%

## 5. Conclusion and Insights

### 5.1 Key Findings
1. **Population-Based Strategy**:
   - Effective handling of varying state sizes
   - More targeted predictions for different categories
   - Better scale difference handling

2. **State-Specific Insights**:
   - Small states: Most consistent predictions
   - Medium states: Balanced performance
   - Large states: Required complex handling

3. **Seasonal Patterns**:
   - Clear summer usage peaks
   - Regional winter variations
   - Strong temperature impact

### 5.2 Limitations
1. **Data Structure**:
   - Fixed population numbers
   - Limited weather data granularity
   - Simplified seasonal indicators

2. **Geographic Factors**:
   - State-level aggregation limitations
   - Climate zone boundary exclusion
   - Missing regional interconnections

3. **External Variables**:
   - Economic factors not included
   - Policy changes not considered
   - Industrial development not tracked

### 5.3 Future Work
1. **Technical Enhancements**:
   - City-level predictions
   - Climate zone stratification
   - Region-specific models

2. **Data Expansion**:
   - Industrial usage patterns
   - Demographic trends
   - Policy change indicators

3. **Analysis Extensions**:
   - State clustering analysis
   - Cross-state dependencies
   - Extreme weather impacts

### 5.4 Setup and Usage
1. Clone the repository
2. Install required packages:
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```
3. Open and run `final_project/Electricity_Usage_Prediction.ipynb`

## 6. References

### 6.1 Data Sources
1. Electricity Dataset: [Kaggle](https://www.kaggle.com/datasets/alistairking/electricity-prices)
2. Weather Data: [Meteostat Python Library](https://dev.meteostat.net/)
3. City Information: [SimpleMaps U.S. Cities Data](https://simplemaps.com/data/us-cities)

### 6.2 Tools and Libraries
1. **Data Analysis & Machine Learning**:
   - NumPy: Numerical computing and array operations
   - Pandas: Data manipulation and analysis
   - Scikit-learn: Machine learning algorithms and evaluation metrics
   - XGBoost: Gradient boosting implementation
   - Seaborn: Statistical data visualization
   - Matplotlib: Data visualization and plotting
   - Meteostat: Weather data retrieval and processing

2. **Dashboard & Visualization**:
   - Panel: Interactive web application framework
   - Flask: Web application framework
   - Renderer: Cloud hosting service

