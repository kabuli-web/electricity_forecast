import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import holidays

df = pd.read_csv("v2_data.csv")

# Convert Date/Time to datetime and set as index
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
 # Lag in hours
# df = df[~((df.index >= '2021-08-01') & (df.index <= '2024-03-01'))]

df["Price"] = abs(df["Price"])

df = df.asfreq('D')

df.ffill(inplace=True)


df = df.filter(["Price","temperature_2m (°C)","cloud_cover (%)","wind_speed_100m (km/h)","gas pr","oil pr","Production Total [MWh]","Nuclear [MWh]","Thermal [MWh]","Wind Onshore [MWh]","Consumption Total [MWh]"])
df["Consumption Total [MWh]"] = abs(df["Consumption Total [MWh]"])



for lag in range(1, 60):
    df[f"Price_lag_{lag}d"] = df["Price"].shift(lag) 

# lag_mean = 10

# # Calculate mean of the previous 10 days' prices
# df[f"{lag_mean}_days_avg"]  = df["Price"].shift(1).rolling(window=lag_mean).mean()


day_of_year = df.index.dayofyear
df['Yearly_Clock_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['Yearly_Clock_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)


se_holidays = holidays.CountryHoliday('SE')

df['Is_Holiday'] = pd.Series(df.index.date).isin(se_holidays).astype(int).values


df['weekDay'] = df.index.day_name()

df['weekDay'] = df['weekDay'].map({
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
})



def day_weight(day):
    if day in [5, 6]:  # Saturday, Sunday
        return 0.8  # lower weight for weekends (adjust as needed)
    else:
        return 1.0  # higher weight for weekdays

df['Day_Weight'] = df.index.weekday.map(day_weight)



Forecast = 5  # forecast 7 days ahead
weather_forecast = 5
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


columns_to_drop = ["Price"] + price_future_cols
X = df.drop(columns=columns_to_drop)
y = df.filter(price_future_cols)

df.dropna(inplace=True)



# Clean up column names to remove disallowed characters
X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '') for col in X.columns]




training_end = df.index.get_loc(pd.Timestamp("2024-03-01"))

X_train = X.iloc[:training_end, :]
y_train = y.iloc[:training_end]

X_test  = X.iloc[training_end:, :]
y_test  = y.iloc[training_end:]



model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

model.save_model('xgboost_model.json')


y_pred = model.predict(X_test)

n_outputs = len(y_test.columns)
columns = [f"Price_d{i+1}" for i in range(n_outputs)]



y_test_df = pd.DataFrame(y_test, index=y_test.index, columns=columns).T
y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=columns).T


mse_list, mae_list, r2_list,mapes_list = [], [], [], []
real = []
pred = []
dates = []

for i in range(1,len(y_pred_df.columns),Forecast):
    mse = mean_squared_error(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])
    mae = mean_absolute_error(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])
    r2  = r2_score(y_test_df.iloc[:, i], y_pred_df.iloc[:, i])
    mapes = np.mean(np.abs((y_test_df.iloc[:, i] - y_pred_df.iloc[:, i]) / y_test_df.iloc[:, i])) * 100
    real.append(y_test_df.iloc[:, i])
    pred.append(y_pred_df.iloc[:, i])
  
    base_date = pd.to_datetime(y_pred_df.columns[i+1])
    forecast_dates = pd.date_range(start=base_date, periods=Forecast, freq='D')    
    dates.extend(forecast_dates)
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)
    mapes_list.append(mapes)
    


real = np.array(real)
pred = np.array(pred)

min_len = min(real.shape[0], pred.shape[0])

mse = mean_squared_error(real, pred)
mae = mean_absolute_error(real, pred)
r2  = r2_score(real, pred)
mape = np.mean(np.abs((real - pred) / real)) * 100

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)
print("MAPE:", mape)



plt.figure(figsize=(20, 10))
plt.plot(dates,real.flatten(), label="Actual")
plt.plot(dates,pred.flatten(), label="Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs Predicted Price (XGBoost)")
plt.legend()
plt.grid()
plt.show()


joblib.dump(model,'xg2boost_lag_future.pkl')