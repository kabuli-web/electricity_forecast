import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
import holidays
import seaborn as sns

# ------------------------
# 1. Data Loading & Preprocessing
# ------------------------
df = pd.read_csv("v2_data.csv")

# Convert to datetime, set index, sort, and drop missing values
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
df.dropna(inplace=True)
df = df[df["Price"] > 0]
df["Price"] = abs(df["Price"])
df = df[~((df.index >= '2021-08-06') & (df.index <= '2024-02-17'))]
# df = df[(df.index >= '2023-01-01') & (df.index <= '2024-10-02')]


# Add additional time features
df["day"] = df.index.day
df["month"] = df.index.month

weather_forecast = 7

weather_columns = []
for d in range(1, weather_forecast+1):
    # Weather
    col_temp  = f"temp_d{d}"
    col_cloud = f"cloud_d{d}"
    col_wind  = f"wind_d{d}"
    weather_columns.extend([col_temp, col_cloud, col_wind])
    df[col_temp]  = df["temperature_2m (°C)"].shift(-d)
    df[col_cloud] = df["cloud_cover (%)"].shift(-d)
    df[col_wind]  = df["wind_speed_100m (km/h)"].shift(-d)

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

corr_matrix = df.corr()

# Extract correlations with 'Price' (dropping Price vs. Price)
corr_with_price = corr_matrix['Price'].drop(columns=['Price',"Metered [MWh]","Profiled [MWh]","Other [MWh]"])

# Optionally sort the correlations for a cleaner look
corr_with_price = corr_with_price.sort_values()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_price.index, y=corr_with_price.values, palette="viridis")
plt.xticks(rotation=45)
plt.ylabel('Correlation with Price')
plt.title('Feature Correlations with Price')
plt.tight_layout()
plt.show()


# ------------------------
# 2. Scaling the Data
# ------------------------
# We want to scale features and target separately.
# First, fit a scaler for the features (all columns except Price)
df["Consumption Total [MWh]"] = abs(df["Consumption Total [MWh]"])

feature_columns = ["Price","temperature_2m (°C)","cloud_cover (%)","wind_speed_100m (km/h)","gas pr","oil pr","Production Total [MWh]","Nuclear [MWh]","Thermal [MWh]","Wind Onshore [MWh]","Consumption Total [MWh]","day","month"]
feature_columns.extend(weather_columns)
feature_columns.extend(["Yearly_Clock_sin", "Yearly_Clock_cos", "Is_Holiday", "weekDay"])
scaler_features = MinMaxScaler()
df_features_scaled = pd.DataFrame(scaler_features.fit_transform(df[feature_columns]),
                                  index=df.index,
                                  columns=feature_columns)

# # Next, fit a scaler for the target Price
target_scaler = MinMaxScaler()
df_target_scaled = pd.DataFrame(target_scaler.fit_transform(df[["Price"]]),
                                index=df.index,
                                columns=["Price"])

# Combine scaled features and target back into one DataFrame

df_features_scaled["Price"] = df_target_scaled["Price"]
df = df_features_scaled
# For sequence creation we will use the scaled DataFrame.
# You can choose to include all features or only a subset.
# For example, if you want to use Price, day, month, and other features, ensure they are in df_scaled.
price = df.filter(["Price"])  # using the entire scaled df

print("Scaled DataFrame columns:", price.columns)

def train_lstm(price,lookback=60, future=7, hidden_layers=13, epochs=20, batch_size=15, learning_rate=0.07256196):

    
    # ------------------------
    # 3. Create Sequences for Time Series Forecasting
    # ------------------------

    def create_sequences(df, lookback, horizon):
        df_as_numpy = df.to_numpy()
        X, y = [], []
        for i in range(len(df_as_numpy) - lookback - horizon + 1):
            row =  df_as_numpy[i:(i + lookback)]
            X.append(row)
            # For the target, we take the next 'horizon' Price values.
            y.append(df_as_numpy[i + lookback : i + lookback + horizon, df.columns.get_loc("Price")])
        return np.array(X), np.array(y)

    
    X_seq, y_seq = create_sequences(price, lookback, future)

    # Use the datetime index (offset by lookback) for plotting later
    time_index = price.index[lookback:]

    # ------------------------
    # 4. Split Data into Training and Testing Sets (Time-Ordered Split)
    # ------------------------

    
    training_end = df.index.get_loc(pd.Timestamp("2024-03-01"))
    X_train = X_seq[:training_end]
    y_train = y_seq[:training_end]
    X_test  = X_seq[training_end:]
    y_test  = y_seq[training_end:]
    

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Also split the corresponding datetime indices for plotting
    train_index = time_index[:training_end]
    test_index  = time_index[training_end:]
    print("Length of test index:", len(test_index))

    # ------------------------
    # 5. Build and Train the LSTM Model
    # ------------------------
    model = Sequential()
    # LSTM expects input shape = (time_steps, features)
    model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(hidden_layers))
    model.add(Dense(future, activation='linear'))  # output: 10-step forecast for Price
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')



    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)


    def run_test():
        i=0
        y_pred_scaled = []
        y_test_scaled = []
        while i < len(X_test):
            prdict = model.predict(X_test[i:i+1])
            prdict = target_scaler.inverse_transform(prdict)[0]
            y_test_scaled.append(target_scaler.inverse_transform(y_test[i:i+1])[0])
            y_pred_scaled.append(prdict)
            i+=future


        
        # Inverse transform predictions and actual values to original scale using target_scaler.
        y_pred = np.array(y_pred_scaled).flatten()
        y_test_inv = np.array(y_test_scaled).flatten()

        
        # Calculate evaluation metrics on the original scale
        mse_val = mean_squared_error(y_test_inv, y_pred)
        mae_val = mean_absolute_error(y_test_inv, y_pred)
        r2_val  = r2_score(y_test_inv, y_pred)
        mape = np.mean(np.abs((y_test_inv - y_pred) / y_test_inv)) * 100

        return mse_val,mae_val,r2_val,mape ,y_test_scaled, y_pred_scaled


    mse,mae,r2,mape,real,pred =  run_test()
    
    x = future

    print("Length of real:", len(real))
    print("Length of pred:", len(pred))

    real_ = np.array(real).flatten()
    pred_ = np.array(pred).flatten()
    
    dates = []

    for i in range(0,len(X_test),future):
        # Convert the column header to a datetime object
        base_date = pd.to_datetime(test_index[i+1])
        # Create a date range starting at base_date with 'forecast' periods (one per day)
        forecast_dates = pd.date_range(start=base_date, periods=future, freq='D')

        dates.extend(forecast_dates)

    dates = np.array(dates).flatten()
    print("Length of dates:", len(dates))
    plt.figure(figsize=(20, 10))
    plt.plot(dates,real_, label='Actual', marker='o')
    plt.plot(dates,pred_, label='Predicted', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Price (LSTM)")
    plt.legend()
    plt.grid()
    plt.show()

    print("Average Mean Squared Error (MSE):", np.mean(mse)) 
    print("Average Mean Absolute Error (MAE):", np.mean(mae))
    print("Average R-squared (R2):", np.mean(r2))
    print("MAPE:", mape)

    return mse,mae, r2

train_lstm(price=price)