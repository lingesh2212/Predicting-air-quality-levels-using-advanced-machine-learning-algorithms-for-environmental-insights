# Predicting-air-quality-levels-using-advanced-machine-learning-algorithms-for-environmental-insights
# air_quality_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 1. Load dataset
df = pd.read_csv('data/air_quality_data.csv')  # Make sure you have this CSV file

# 2. Preprocessing
df.dropna(inplace=True)  # Remove missing values for simplicity

# Features and Target
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
target = 'AQI'  # Or replace with 'PM2.5' or other pollutant

X = df[features]
y = df[target]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 6. Save model
joblib.dump(model, 'models/aqi_model.pkl')
print("Model saved as models/aqi_model.pkl")
