import numpy as np
import pandas as pd
import time
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================================================
# CONFIG
# =========================================================
DATA = "supervised_contig_filtered_H6_L24.npz"
P_RATED_KW = 111.0
LAMBDA = 0.01

z = np.load(DATA)
X_uni = z["X_uni"].astype(np.float32)
Y = z["Y"].astype(np.float32)

# Convert W → kW
X_uni[:, :, 0] /= 1000.0
Y /= 1000.0

split = int(0.8 * len(Y))

# =========================================================
# METRICS
# =========================================================
def eval_model(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    nrmse = (rmse / P_RATED_KW) * 100

    return rmse, mae, nrmse

# =========================================================
# BASELINES
# =========================================================
def persistence(X_test):
    last = X_test[:, -1, 0]
    return np.repeat(last[:, None], Y.shape[1], axis=1)

def xgboost_model(X_train, Y_train, X_test):
    Xtr = X_train.reshape(X_train.shape[0], -1)
    Xte = X_test.reshape(X_test.shape[0], -1)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(Xtr, Y_train)
    return model.predict(Xte)

# =========================================================
# DL MODELS
# =========================================================
def build_lstm(input_shape, H):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(H)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(input_shape, H):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dense(H)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# =========================================================
# PHYSICS LOSS
# =========================================================
def physical_loss(y_pred):
    neg = tf.reduce_mean(tf.square(tf.minimum(y_pred, 0.0)))
    over = tf.reduce_mean(tf.square(tf.maximum(y_pred - P_RATED_KW, 0.0)))
    return neg + over

def build_pi_lstm(input_shape, H):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(H)
    ])

    def loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        phys = physical_loss(y_pred)
        return mse + LAMBDA * phys

    model.compile(optimizer="adam", loss=loss)
    return model

# =========================================================
# TRAIN FUNCTION
# =========================================================
def run_model(model_fn, name):
    X_train, X_test = X_uni[:split], X_uni[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    model = model_fn((X_train.shape[1], X_train.shape[2]), Y.shape[1])

    es = EarlyStopping(patience=8, restore_best_weights=True)
    rl = ReduceLROnPlateau(patience=4, factor=0.5)

    start = time.time()

    model.fit(
        X_train, Y_train,
        validation_split=0.1,
        epochs=60,
        batch_size=64,
        callbacks=[es, rl],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)

    t_inf = (time.time() - start) / len(X_test)

    rmse, mae, nrmse = eval_model(Y_test, y_pred)

    return [name, rmse, mae, nrmse, t_inf]

# =========================================================
# RUN EXPERIMENTS
# =========================================================
results = []

X_train, X_test = X_uni[:split], X_uni[split:]
Y_train, Y_test = Y[:split], Y[split:]

# DL models
results.append(run_model(build_lstm, "LSTM"))
results.append(run_model(build_gru, "GRU"))
results.append(run_model(build_pi_lstm, "PI-LSTM"))

# Baselines
results.append([
    "Persistence",
    *eval_model(Y_test, persistence(X_test)),
    0.0001
])

start = time.time()
y_xgb = xgboost_model(X_train, Y_train, X_test)
t_xgb = (time.time() - start) / len(X_test)
results.append(["XGBoost", *eval_model(Y_test, y_xgb), t_xgb])

# =========================================================
# FINAL TABLE
# =========================================================
df = pd.DataFrame(results, columns=[
    "Model", "RMSE", "MAE", "nRMSE%", "Inference_Time_s"
])

print(df)
df.to_csv("FINAL_MDPI_RESULTS.csv", index=False)