import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load your data
data = pd.read_csv(r"C:\Users\Dylan Patel\Desktop\electricity-usage-prediction\datasets\cleaned_electricity_usage_data.csv")

# Convert year and month to a datetime format for better handling of time-series
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))

# Set the date as the index for time-series analysis
data.set_index('date', inplace=True)

# One-hot encode the state and sector information
data = pd.get_dummies(data, columns=['stateDescription', 'sectorName'], drop_first=True)

# Enhance feature set with seasonal and lag features
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)  # Seasonal feature
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)  # Seasonal feature
data['previous_sales'] = data['sales'].shift(1)  # Lag feature for previous month sales
data.dropna(inplace=True)  # Remove rows with NaN values resulting from lag

# Define the features (X) and target variables (y)
X = data.drop(['price', 'sales'], axis=1)
y_price = data['price']
y_sales = data['sales']

# Split data into training and testing sets for price and sales
X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)
_, _, y_train_sales, y_test_sales = train_test_split(X, y_sales, test_size=0.2, random_state=42)

# Create pipelines for both price and sales predictions with Random Forest
def create_pipeline():
    return Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

price_pipeline = create_pipeline()
sales_pipeline = create_pipeline()

# Fit the models
price_pipeline.fit(X_train, y_train_price)
sales_pipeline.fit(X_train, y_train_sales)

# Make predictions
price_preds = price_pipeline.predict(X_test)
sales_preds = sales_pipeline.predict(X_test)

# Evaluate the models
price_rmse = np.sqrt(mean_squared_error(y_test_price, price_preds))
sales_rmse = np.sqrt(mean_squared_error(y_test_sales, sales_preds))

print(f"Price Prediction RMSE: {price_rmse}")
print(f"Sales Prediction RMSE: {sales_rmse}")

# Example for estimating electricity usage for December 2024
# Create features for next month based on the training features
next_month_data = {
    'year': [2024],
    'month': [12],
    'revenue': [0],  # Adjust this value as necessary
    'month_sin': [np.sin(2 * np.pi * 12 / 12)],  # December
    'month_cos': [np.cos(2 * np.pi * 12 / 12)],  # December
    'previous_sales': [0]  # Estimate based on previous data or historical average
}

# Create a DataFrame with the correct features
next_month_df = pd.DataFrame(next_month_data)

# Add one-hot encoded stateDescription columns to next_month_df
state_columns = [col for col in X_train.columns if col.startswith('stateDescription_')]
for state_col in state_columns:
    next_month_df[state_col] = 0  # Default to 0 for missing one-hot encoded features

# Set the state to predict (e.g., for Maine)
next_month_df['stateDescription_Maine'] = 1  # Example for Maine

# Add one-hot encoded sector features as needed
sector_columns = [col for col in X_train.columns if col.startswith('sectorName_')]
for sector_col in sector_columns:
    next_month_df[sector_col] = 0  # Default to 0 for missing one-hot encoded features

# Make predictions using the sales pipeline
predicted_usage = sales_pipeline.predict(next_month_df[X_train.columns])

print(f"Estimated Electricity Usage for December 2024: {predicted_usage[0]}")

"""
results:
Price Prediction RMSE: 0.3767734676249607
Sales Prediction RMSE: 331.3029925437795
Estimated Electricity Usage for December 2024: 722.7777566000003

price rsme right
sales rsme seems wrong
and so estimate sounds wrong
"""