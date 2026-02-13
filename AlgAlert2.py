import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ===============================
# Config
# ===============================
DATA_PATH = 'sitedata_dirty_rev.csv'
NROWS = None 
EPS = 1e-6
SEED = 42 
N_SPLITS = 5
TRAIN_DIRECT_RATIO_MODEL = False  # set True to also train a regressor for ratio

# Column names (as in your CSV)
BGA_PC_COL_RAW = 'BGA-PC (ug/L)'
CHLA_COL_RAW = 'Chl_a (ug/L)'
RATIO_COL_RAW = 'BGA-PC/ Chl-a'

LBL_CHLA = 'bloomstatus Chl-a'
LBL_BGA = 'bloomstatus BGA-PC'
LBL_RATIO = 'bloomstatus BGA-PC/Chl-a'

# ===============================
# Load
# ===============================
df = pd.read_csv(DATA_PATH, na_values=["", " "], nrows=NROWS)
df = df.loc[:, ~df.columns.duplicated()]  # in case there are duplicate names

# Basic checks
required_cols = [BGA_PC_COL_RAW, CHLA_COL_RAW]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# If ratio is missing in the dataset, compute it from the true values (optional)
if RATIO_COL_RAW not in df.columns:
    df[RATIO_COL_RAW] = df[BGA_PC_COL_RAW] / np.maximum(df[CHLA_COL_RAW], EPS)

# Check which labels exist
label_cols = [c for c in [LBL_CHLA, LBL_BGA, LBL_RATIO] if c in df.columns]
if not label_cols:
    raise ValueError("No bloomstatus columns found. Expected any of: "
                     f"{[LBL_CHLA, LBL_BGA, LBL_RATIO]}")

print("Missing values per column:\n", df.isna().sum())

# ===============================
# Feature/Target Split
# ===============================
# Targets for regression
y_chla = df[CHLA_COL_RAW].astype(float)
y_bga  = df[BGA_PC_COL_RAW].astype(float)

# Features for regression = all columns except explicit targets and labels
drop_cols = [BGA_PC_COL_RAW, CHLA_COL_RAW, RATIO_COL_RAW] + label_cols
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# ColumnTransformer: numeric median-impute+scale; categorical impute+onehot
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# Persist the fitted preprocessor (fit on all X to capture categories)
preprocessor.fit(X)
joblib.dump(preprocessor, "preprocessor_X.pkl")

# ===============================
# Regression Models
# ===============================
regression_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=SEED),
    "SVR": SVR(kernel='rbf', C=10.0, gamma='scale'),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED),
    "Ridge": Ridge(alpha=1.0, random_state=SEED),
    "Lasso": Lasso(alpha=0.01, random_state=SEED)
}

# ===============================
# Train regressors + save outputs
# ===============================
all_outputs = []  # collect per-model outputs for summary later

for model_name, reg_est in regression_models.items():
    print(f"\n=== {model_name} ===")

    # Pipelines wrapping the shared preprocessor
    chla_pipe = Pipeline([("pre", preprocessor), ("reg", reg_est)])
    bga_pipe  = Pipeline([("pre", preprocessor), ("reg", type(reg_est)(**getattr(reg_est, 'get_params', lambda: {})()))])

    # --- Fit on full data (you can add proper CV scoring separately) ---
    chla_pipe.fit(X, y_chla)
    bga_pipe.fit(X, y_bga)

    # Save the fitted pipes
    joblib.dump(chla_pipe, f"{model_name}_Chl_a_pipe.pkl")
    joblib.dump(bga_pipe,  f"{model_name}_BGA_PC_pipe.pkl")

    # --- Predictions (in-sample; add CV if you need honest metrics) ---
    chla_pred = chla_pipe.predict(X)
    bga_pred  = bga_pipe.predict(X)
    ratio_pred = bga_pred / np.maximum(chla_pred, EPS)

    # Optional: direct ratio model (usually less stable—computed ratio preferred)
    if TRAIN_DIRECT_RATIO_MODEL:
        ratio_true = df[RATIO_COL_RAW].astype(float).values
        ratio_pipe = Pipeline([
            ("pre", preprocessor),
            ("reg", type(reg_est)(**getattr(reg_est, 'get_params', lambda: {})()))
        ])
        ratio_pipe.fit(X, ratio_true)
        joblib.dump(ratio_pipe, f"{model_name}_Ratio_pipe.pkl")
        ratio_pred_direct = ratio_pipe.predict(X)
    else:
        ratio_pred_direct = None

    # Store predictions to df copy
    df_out = df.copy()
    df_out[f'Chl_a_predicted_{model_name}'] = chla_pred
    df_out[f'BGA_PC_predicted_{model_name}'] = bga_pred
    df_out[f'Ratio_predicted_{model_name}'] = ratio_pred
    if ratio_pred_direct is not None:
        df_out[f'Ratio_direct_predicted_{model_name}'] = ratio_pred_direct

    # Quick regression metrics (in-sample)
    r2_chla = r2_score(y_chla, chla_pred)
    mae_chla = mean_absolute_error(y_chla, chla_pred)
    r2_bga = r2_score(y_bga, bga_pred)
    mae_bga = mean_absolute_error(y_bga, bga_pred)
    print(f"Chl-a: R2={r2_chla:.3f}, MAE={mae_chla:.3f}")
    print(f"BGA-PC: R2={r2_bga:.3f}, MAE={mae_bga:.3f}")

    # ===============================
    # Classification (based on predicted data)
    # ===============================
    # Build classification feature matrix from predictions
    X_cls_pred = np.column_stack([bga_pred, chla_pred, ratio_pred])
    cls_feature_names = [f"{model_name}_bga_pred", f"{model_name}_chla_pred", f"{model_name}_ratio_pred"]

    # Classifiers to train (swap/add as needed)
    classifiers = {
        "LogReg": LogisticRegression(max_iter=200, class_weight="balanced", random_state=SEED),
        # You can add others (e.g., RandomForestClassifier) if desired:
        # "RF": RandomForestClassifier(n_estimators=400, random_state=SEED, class_weight="balanced")
    }

    # For each classifier type, fit one model per available label column
    for cls_name, cls_est in classifiers.items():
        print(f"\n-- Classifier: {cls_name} (features: predicted BGA, Chl-a, Ratio from {model_name})")

        # K-fold evaluation for quick sanity check
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        for lbl in label_cols:
            y_lbl = df[lbl].astype(int).values

            # Cross-validated metrics (on the predicted features)
            acc_scores, f1_scores = [], []
            for tr, va in kf.split(X_cls_pred):
                X_tr, X_va = X_cls_pred[tr], X_cls_pred[va]
                y_tr, y_va = y_lbl[tr], y_lbl[va]

                # Fresh instance per fold to avoid leakage of state
                est_fold = type(cls_est)(**getattr(cls_est, 'get_params', lambda: {})())
                est_fold.fit(X_tr, y_tr)
                y_hat = est_fold.predict(X_va)
                acc_scores.append(accuracy_score(y_va, y_hat))
                f1_scores.append(f1_score(y_va, y_hat, zero_division=0))

            print(f"Label: {lbl} | CV-Acc={np.mean(acc_scores):.3f} | CV-F1={np.mean(f1_scores):.3f}")

            # Train final classifier on all rows
            est_final = type(cls_est)(**getattr(cls_est, 'get_params', lambda: {})())
            est_final.fit(X_cls_pred, y_lbl)
            joblib.dump(est_final, f"{model_name}_{cls_name}_{lbl.replace(' ', '_').replace('/', '_')}.pkl")

            # Add predictions back to df_out
            y_hat_full = est_final.predict(X_cls_pred)
            df_out[f'pred_{lbl}_{model_name}_{cls_name}'] = y_hat_full

            # Print one global classification report (in-sample)
            print("\nClassification Report (full data, for reference):")
            print(classification_report(y_lbl, y_hat_full, digits=3))

    # Save per-model CSV
    csv_path = f'df_with_{model_name}_predictions_and_labels.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"Saved predictions and label outputs to: {csv_path}")

    # Collect a small summary
    all_outputs.append({
        "model": model_name,
        "r2_chla": r2_chla, "mae_chla": mae_chla,
        "r2_bga": r2_bga, "mae_bga": mae_bga,
        "csv": csv_path
    })

# ===============================
# Summary
# ===============================
print("\n=== Summary (in-sample regression metrics) ===")
for o in all_outputs:
    print(f"{o['model']}: Chl-a R2={o['r2_chla']:.3f} MAE={o['mae_chla']:.3f} | "
          f"BGA-PC R2={o['r2_bga']:.3f} MAE={o['mae_bga']:.3f} | CSV={o['csv']}")