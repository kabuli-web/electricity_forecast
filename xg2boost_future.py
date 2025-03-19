import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ------------------------
# 1. Data Loading & Preprocessing
# ------------------------
df = pd.read_csv("v2_data.csv")

# Convert Date/Time to datetime and set as index
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
 # Lag in hours
df = df[~((df.index >= '2021-08-01') & (df.index <= '2024-03-01'))]

df["Price"] = abs(df["Price"])

df = df.asfreq('D')

df.ffill(inplace=True)

# print("Price Mean", df["Price"].mean())
# print("Price Max", df["Price"].max())
# print("Price Min", df["Price"].min())
# print("Price Median", df["Price"].median())

# Drop unnecessary column
# df.drop(columns=["snowfall (cm)","Consumption Total [MWh]"], inplace=True)
df = df.filter(["Price","temperature_2m (°C)","cloud_cover (%)","wind_speed_100m (km/h)"])


for lag in range(1, 14):
    df[f"Price_lag_{lag}h"] = df["Price"].shift(lag) 


# -------------------------------
# 2. Aggregate to Daily Averages
#    - Price
#    - Weather
# -------------------------------
Forecast = 24  # forecast 7 days ahead
weather_forecast = 14
price_future_cols = []
# weather_future_cols = []

for d in range(1, Forecast+1):
    # Price
    col_price = f"Price_d{d}"
    df[col_price] = df["Price"].shift(-d)
    price_future_cols.append(col_price)

    


for d in range(1, weather_forecast+1):
    # Weather
    col_temp  = f"temp_d{d}"
    col_cloud = f"cloud_d{d}"
    col_wind  = f"wind_d{d}"

    df[col_temp]  = df["temperature_2m (°C)"].shift(-d)
    df[col_cloud] = df["cloud_cover (%)"].shift(-d)
    df[col_wind]  = df["wind_speed_100m (km/h)"].shift(-d)

df.dropna(inplace=True)
# ------------------------
# 2. Prepare Features and Target Variable
# ------------------------
columns_to_drop = ["Price"] + price_future_cols
X = df.drop(columns=columns_to_drop)
y = df.filter(price_future_cols)

df.dropna(inplace=True)



# Clean up column names to remove disallowed characters
X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '') for col in X.columns]

print(X.columns )
# ------------------------
# 3. Train-Test Split
# ------------------------
# Using a fixed random_state for reproducibility


training_end = df.index.get_loc("2024-03-01")

X_train = X.iloc[:training_end, :]
y_train = y.iloc[:training_end]

X_test  = X.iloc[training_end+24:, :]
y_test  = y.iloc[training_end+24:]


# ------------------------
# 4. Build and Train the XGBoost Model
# ------------------------
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

model.save_model('xgboost_model.json')

# ------------------------
# 5. Predictions and Evaluation
# ------------------------
y_pred = model.predict(X_test)
# print(y_test[5:])

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

# ------------------------
# 6. Plotting Actual vs Predicted Values
# ------------------------
n_outputs = len(y_test.columns)
columns = [f"Price_d{i+1}" for i in range(n_outputs)]

y_test_df = pd.DataFrame(y_test, index=y_test.index, columns=columns).T
y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=columns).T


mse_list, mae_list, r2_list = [], [], []

for i in range(1,len(y_pred_df.columns),Forecast):
    mse = mean_squared_error(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])
    mae = mean_absolute_error(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])
    r2  = r2_score(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])

    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)

    # plt.figure(figsize=(10, 5))
    # plt.plot(y_test_df.index, y_test_df.iloc[:, i], label="Actual")
    # plt.plot(y_pred_df.index, y_pred_df.iloc[:, i], label="Predicted")
    # plt.title(y_pred_df.columns[i])
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.legend()
    # plt.grid()
    # plt.show()

print("Average Mean Squared Error (MSE):", np.mean(mse_list))
print("Average Mean Absolute Error (MAE):", np.mean(mae_list))
print("Average R-squared (R2):", np.mean(r2_list))
    



joblib.dump(model,'xg2boost_lag_future.pkl')