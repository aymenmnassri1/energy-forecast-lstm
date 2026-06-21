import pandas as pd
import numpy as np

DATA = r"C:\Users\aymen\OneDrive\Bureau\PV_Project\resultats\PV_2025_clean_5min.csv"
OUT  = r"C:\Users\aymen\OneDrive\Bureau\PV_Project\resultats\supervised_contig_H6_L24.npz"

L = 24   # 2h look-back
H = 6    # 30min horizon

df = pd.read_csv(DATA, parse_dates=["Datetime"]).set_index("Datetime")

# Keep only daytime available points
df = df.dropna(subset=["PV_W"])

# Add simple time features
df["hour"] = df.index.hour
df["minute"] = df.index.minute
df["doy"] = df.index.dayofyear

# Choose columns
cols_uni  = ["PV_W", "hour", "minute", "doy"]
cols_mul  = ["PV_W", "Irradiance_Wm2", "hour", "minute", "doy"]

df_uni = df[cols_uni].dropna().copy()
df_mul = df[cols_mul].dropna().copy()

# ---- Identify contiguous segments (Δt = 5 min) ----
dt = df_uni.index.to_series().diff()
breaks = (dt != pd.Timedelta(minutes=5)).fillna(True)
seg_id = breaks.cumsum()
df_uni["seg"] = seg_id.values
df_mul["seg"] = seg_id.values

def make_xy_from_segments(df_seg: pd.DataFrame, L: int, H: int):
    X_list, Y_list = [], []
    # df_seg has numeric columns only (no seg)
    mat = df_seg.values
    for i in range(L, len(mat) - H):
        X_list.append(mat[i-L:i, :])
        Y_list.append(mat[i:i+H, 0])  # PV is first column
    return X_list, Y_list

X_uni_all, X_mul_all, Y_all = [], [], []

min_len = L + H + 10   

for sid in sorted(df_uni["seg"].unique()):
    part_uni = df_uni[df_uni["seg"] == sid].drop(columns=["seg"])
    part_mul = df_mul[df_mul["seg"] == sid].drop(columns=["seg"])

    if len(part_uni) < min_len:
        continue

    X_u, Y_u = make_xy_from_segments(part_uni, L, H)
    X_m, Y_m = make_xy_from_segments(part_mul, L, H)

    # Append
    X_uni_all += X_u
    X_mul_all += X_m
    Y_all += Y_u  # same Y

X_uni = np.array(X_uni_all)
X_mul = np.array(X_mul_all)
Y = np.array(Y_all)

np.savez_compressed(OUT, X_uni=X_uni, X_mul=X_mul, Y=Y)

print("✅ Saved:", OUT)
print("X_uni:", X_uni.shape, " | X_mul:", X_mul.shape, " | Y:", Y.shape)
