import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
DATA = r"C:\Users\aymen\OneDrive\Bureau\PV_Project\resultats\PV_2025_clean_5min_filtered.csv"
OUT  = r"C:\Users\aymen\OneDrive\Bureau\PV_Project\resultats\supervised_contig_filtered_H6_L24.npz"

L = 24   # look-back = 24 steps = 2 hours (5-min resolution)
H = 6    # horizon  = 6 steps  = 30 minutes

# =========================================================
# LOAD & PREPARE
# =========================================================
df = pd.read_csv(DATA, parse_dates=["Datetime"]).set_index("Datetime")

# Daytime only (keep available PV values)
df = df.dropna(subset=["PV_W"])

# Time features
df["hour"] = df.index.hour
df["minute"] = df.index.minute
df["doy"] = df.index.dayofyear

# Feature sets
cols_uni = ["PV_W", "hour", "minute", "doy"]
cols_mul = ["PV_W", "Irradiance_Wm2", "hour", "minute", "doy"]

df_uni = df[cols_uni].dropna().copy()
df_mul = df[cols_mul].dropna().copy()

# =========================================================
# CONTIGUOUS SEGMENTS (Δt = 5 min)
# =========================================================
dt = df_uni.index.to_series().diff()
breaks = (dt != pd.Timedelta(minutes=5)).fillna(True)
seg_id = breaks.cumsum()

df_uni["seg"] = seg_id.values
df_mul["seg"] = seg_id.values

def make_xy(seg_df: pd.DataFrame, L: int, H: int):
    """
    Build X,Y windows inside one contiguous segment only.
    PV_W is assumed to be first column in seg_df.
    """
    mat = seg_df.values
    X, Y = [], []
    for i in range(L, len(mat) - H):
        X.append(mat[i - L:i, :])
        Y.append(mat[i:i + H, 0])  # PV target multi-step
    return X, Y

# =========================================================
# BUILD DATASET
# =========================================================
X_uni_all, X_mul_all, Y_all = [], [], []

# Segment must be long enough
min_len = L + H + 10

for sid in sorted(df_uni["seg"].unique()):
    part_uni = df_uni[df_uni["seg"] == sid].drop(columns=["seg"])
    part_mul = df_mul[df_mul["seg"] == sid].drop(columns=["seg"])

    if len(part_uni) < min_len:
        continue

    Xu, Yu = make_xy(part_uni, L, H)
    Xm, _  = make_xy(part_mul, L, H)

    X_uni_all += Xu
    X_mul_all += Xm
    Y_all += Yu

X_uni = np.array(X_uni_all)
X_mul = np.array(X_mul_all)
Y = np.array(Y_all)

# =========================================================
# SAVE
# =========================================================
np.savez_compressed(OUT, X_uni=X_uni, X_mul=X_mul, Y=Y)

print("✅ Saved:", OUT)
print("X_uni:", X_uni.shape, "| X_mul:", X_mul.shape, "| Y:", Y.shape)
print("Y max (W):", Y.max())
