import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.optimizers import Adam

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
df = df[~((df.index >= '2021-08-06') & (df.index <= '2024-02-17'))]

# Add additional time features
df["day"] = df.index.day
df["month"] = df.index.month


# ------------------------
# 2. Scaling the Data
# ------------------------
# We want to scale features and target separately.
# First, fit a scaler for the features (all columns except Price)
# df["Consumption Total [MWh]"] = abs(df["Consumption Total [MWh]"])

# feature_columns = ["day","month","gas pr","oil pr"]
# scaler_features = MinMaxScaler()
# df_features_scaled = pd.DataFrame(scaler_features.fit_transform(df[feature_columns]),
#                                   index=df.index,
#                                   columns=feature_columns)

# # Next, fit a scaler for the target Price
target_scaler = MinMaxScaler()
df_target_scaled = pd.DataFrame(target_scaler.fit_transform(df[["Price"]]),
                                index=df.index,
                                columns=["Price"])

# Combine scaled features and target back into one DataFrame

df["Price"] = df_target_scaled["Price"]

# For sequence creation we will use the scaled DataFrame.
# You can choose to include all features or only a subset.
# For example, if you want to use Price, day, month, and other features, ensure they are in df_scaled.
price = df.filter(["Price"])  # using the entire scaled df

print("Scaled DataFrame columns:", price.columns)

def train_lstm(price,lookback=60, future=7, hidden_layers=10, epochs=10, batch_size=8):
    
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
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Also split the corresponding datetime indices for plotting
    train_index = time_index[:split_idx]
    test_index  = time_index[split_idx:]
    print("Length of test index:", len(test_index))

    # ------------------------
    # 5. Build and Train the LSTM Model
    # ------------------------
    model = Sequential()
    # LSTM expects input shape = (time_steps, features)
    model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(hidden_layers))
    model.add(Dense(7, activation='linear'))  # output: 10-step forecast for Price
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    # ------------------------
    # 6. Predictions and Evaluation
    # ------------------------
    # For example, we pick one sequence from the test set.
    test_sequences = [80,120,150,190,250,270]
    # This will be a single sample with shape (1, lookback, num_features)

    def run_test(test_sequence):

        y_pred_scaled = model.predict(X_test[test_sequence:test_sequence+1])

        # Inverse transform predictions and actual values to original scale using target_scaler.
        y_pred = target_scaler.inverse_transform(y_pred_scaled)[0]
        y_test_inv = target_scaler.inverse_transform(y_test[test_sequence:test_sequence+1])[0]
    
        # Calculate evaluation metrics on the original scale
        mse_val = mean_squared_error(y_test_inv, y_pred)
        mae_val = mean_absolute_error(y_test_inv, y_pred)
        r2_val  = r2_score(y_test_inv, y_pred)


        # print("predicted",y_pred)
        # print("actual",y_test_inv)

        # print("Mean Squared Error (MSE):", mse_val)
        # print("Mean Absolute Error (MAE):", mae_val)
        # print("R-squared (R2):", r2_val)

        
        # # ------------------------
        # # 7. Plotting Actual vs Predicted
        # # ------------------------
        # # Here we assume that for this sample, our forecast horizon is 10 steps.
        # # Let's create a forecast index starting from a given test timestamp.
        # start_date = test_index[test_sequence:test_sequence+1][0]
        # forecast_index = pd.date_range(start=start_date, periods=future, freq='D')

        # results = pd.DataFrame({"Actual": y_test_inv.flatten(), "Predicted": y_pred.flatten()}, index=forecast_index)
        # results = results.sort_index()

        # plt.figure(figsize=(10, 6))
        # plt.plot(results.index, results['Actual'], label='Actual', marker='o')
        # plt.plot(results.index, results['Predicted'], label='Predicted', marker='x')
        # plt.xlabel("Date")
        # plt.ylabel("Price")
        # plt.title("Actual vs Predicted Price (LSTM)")
        # plt.legend()
        # plt.xticks(rotation=45)
        # plt.show()

        return mse_val,mae_val,r2_val

    mses = []
    maes = []
    r2s = []

    for i in test_sequences:
        mse,mae,r2 =  run_test(i)
        mses.append(mse)
        maes.append(mae)
        r2s.append(r2)

    # print("Average Mean Squared Error (MSE):", np.mean(mses)) 
    # print("Average Mean Absolute Error (MAE):", np.mean(maes))
    # print("Average R-squared (R2):", np.mean(r2s))

    return np.mean(mses),np.mean(maes),np.mean(r2s)