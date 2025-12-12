# from meeting with dadson/corina:
# add a layer that will predict bgaPC
# then quantify that then as cyanobacteria
# if bgaPC > 45, then classify cyanobacteria

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load the data
df = pd.read_csv('SiteDataCleanedFormatted.csv', nrows=2557)

# Features for regression (excluding targets and bloomstatus)
X_reg = df.drop(['Chl_a', 'bloomstatus', 'BGA-PC (microg/L)'], axis=1).values

# Scale features
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)
joblib.dump(scaler_reg, 'scaler_reg.pkl')

# Targets for regression
y_Chl_a = df['Chl_a'].values.reshape(-1, 1)
y_bgaPC = df['BGA-PC (microg/L)'].values.reshape(-1, 1)

# Scale targets
scaler_y_Chl_a = StandardScaler()
scaler_y_bgaPC = StandardScaler()
y_Chl_a_scaled = scaler_y_Chl_a.fit_transform(y_Chl_a)
y_bgaPC_scaled = scaler_y_bgaPC.fit_transform(y_bgaPC)

joblib.dump(scaler_y_Chl_a, 'scaler_y_Chl_a.pkl')
joblib.dump(scaler_y_bgaPC, 'scaler_y_bgaPC.pkl')

# Define regression models
regression_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf'),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}

print("Missing values per column:\n", df.isna().sum())

# Loop through models
for model_name, reg_model in regression_models.items():
    # Train and predict Chl_a
    reg_model.fit(X_reg_scaled, y_Chl_a_scaled.ravel())
    joblib.dump(reg_model, f'{model_name}_Chl_a.pkl')
    Chl_a_predictions = reg_model.predict(X_reg_scaled)
    Chl_a_predictions = scaler_y_Chl_a.inverse_transform(Chl_a_predictions.reshape(-1, 1)).ravel()
    df[f'Chl_a_predicted_{model_name}'] = Chl_a_predictions

    # Train and predict bgaPC
    reg_model.fit(X_reg_scaled, y_bgaPC_scaled.ravel())
    joblib.dump(reg_model, f'{model_name}_bgaPC.pkl')
    bgaPC_predictions = reg_model.predict(X_reg_scaled)
    bgaPC_predictions = scaler_y_bgaPC.inverse_transform(bgaPC_predictions.reshape(-1, 1)).ravel()
    df[f'bgaPC_predicted_{model_name}'] = bgaPC_predictions

    # Classify cyanobacteria based on bgaPC threshold
    df[f'cyanobacteria_predicted_{model_name}'] = (bgaPC_predictions > 45).astype(int)

    # Save intermediate DataFrame
    df.to_csv(f'df_with_{model_name}_predictions.csv', index=False)

    # Classification using bloomstatus
    if 'bloomstatus' in df.columns:
        X_cls = df.drop(['Chl_a', 'bloomstatus', 'BGA-PC (microg/L)'], axis=1).iloc[:, 1:].values
        y_cls = df['bloomstatus'].values

        scaler_cls = StandardScaler()
        X_cls_scaled = scaler_cls.fit_transform(X_cls)

        # Neural network for bloom classification
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_cls_scaled.shape[1],)),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True)

        # K-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_scores = []

        for train_index, val_index in kf.split(X_cls_scaled):
            X_train_fold, X_val_fold = X_cls_scaled[train_index], X_cls_scaled[val_index]
            y_train_fold, y_val_fold = y_cls[train_index], y_cls[val_index]

            history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                                epochs=100, callbacks=[early_stopping], verbose=0)

            val_accuracy = model.evaluate(X_val_fold, y_val_fold)[1]
            accuracy_scores.append(val_accuracy)

        print(f"\n{model_name} Regression Model - Classification Cross-Validation Accuracy: {np.mean(accuracy_scores)}")
        model.save(f'{model_name}_bloom_classifier.h5')

        # Predictions for bloomstatus
        y_pred_probs = model.predict(X_cls_scaled)
        y_pred = (y_pred_probs > 0.5).astype(int)
        df['bloomstatus_predicted'] = y_pred
        df.to_csv(f'df_with_{model_name}_bloom_predictions.csv', index=False)

        print("\nClassification Report:")
        print(classification_report(y_cls, y_pred))
    else:
        print("'bloomstatus' column not found in the DataFrame.")
