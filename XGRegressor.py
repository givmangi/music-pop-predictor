import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd # Used if y_raw are pandas Series and for feature names
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# --- Assume your data loading and preprocessing function is available ---
# You'll need to adapt this part to call your actual function that returns:
# X_train_processed, y_train_raw, X_val_processed, y_val_raw,
# X_test_processed, y_test_raw, and your fitted preprocessor_X object.

def load_and_preprocess_data(csv_path, target_column_name):
    try:
        df = pd.read_csv(csv_path, sep=';') 
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found.")
        return None, None, None, None, None, None, 0, None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None, None, None, None, None, 0, None

    if target_column_name not in df.columns:
        print(f"Error: Target column '{target_column_name}' not found in DataFrame.")
        return None, None, None, None, None, None, 0, None

    initial_cols = df.shape[1]
    df.dropna(axis=1, how='all', inplace=True)
    cols_after_drop = df.shape[1]
    if initial_cols > cols_after_drop:
        print(f"Dropped {initial_cols - cols_after_drop} columns that were entirely NaN.")

    features_to_drop = [
        'track_id', 'track_name', 'track_artist',
        'track_album_id', 'track_album_name',
        'release_year', 'release_month',
    ]
    # Handle Date Feature
    if 'track_album_release_date' in df.columns:
        try:
            df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')
            df['track_album_release_date'] = df['track_album_release_date'].ffill() # Forward fill missing dates

            current_year = pd.to_datetime('today').year
            df['release_year'] = df['track_album_release_date'].dt.year
            df['release_month'] = df['track_album_release_date'].dt.month
            df['album_age_years'] = current_year - df['release_year']
            features_to_drop.append('track_album_release_date')
        except Exception as e:
            print(f"Warning: Could not process 'track_album_release_date': {e}. It will be dropped.")
            if 'track_album_release_date' not in features_to_drop:
                 features_to_drop.append('track_album_release_date')


    y_raw = df[target_column_name].copy()
    X_raw = df.drop(columns=[target_column_name] + features_to_drop, errors='ignore').copy()

    numerical_features = X_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_raw.select_dtypes(include='object').columns.tolist()

    potential_cats_numeric = ['key', 'mode']
    for p_cat in potential_cats_numeric:
        if p_cat in X_raw.columns: # Check ix exist
            if p_cat in numerical_features:
                numerical_features.remove(p_cat)
                categorical_features.append(p_cat)
            elif p_cat not in categorical_features:
                 categorical_features.append(p_cat)

    current_X_cols = X_raw.columns.tolist()
    numerical_features = [f for f in numerical_features if f in current_X_cols]
    categorical_features = [f for f in categorical_features if f in current_X_cols]

    print(f"Final Numerical features for X: {numerical_features}")
    print(f"Final Categorical features for X: {categorical_features}")
    print(f"Number of X features: {len(numerical_features) + len(categorical_features)}")


    X_train_val_raw, X_test_raw, y_train_val_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=42
    )

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_val_raw, y_train_val_raw, test_size=0.1764, random_state=42 # 0.1764 is approx 15% of 85%
    )
    # appr. 68% Train, 17% Val, 15% Test


    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        #too sensible to outliers ('scaler', StandardScaler())])
        ('scaler', RobustScaler())])
    categorical_transformer_X = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    print(f"Unique playlist_genre values: {X_raw['playlist_genre'].nunique()}")
    #print(f"Unique playlist_subgenre values: {X_raw['playlist_subgenre'].nunique()}")
    print(f"numerical features: {numerical_features}, \n categorical features: {categorical_features}")
    preprocessor_X = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer_X, categorical_features)],
        remainder='drop')

    try:
        X_train_processed = preprocessor_X.fit_transform(X_train_raw)
        X_val_processed = preprocessor_X.transform(X_val_raw)
        X_test_processed = preprocessor_X.transform(X_test_raw)
    except Exception as e:
        print(f"Error during X preprocessing: {e}")
        return None, None, None, None, None, None, 0, None

    y_scaler = StandardScaler()
    try:
        y_train_scaled = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val_raw.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()

        print(f"Target column '{target_column_name}' scaled.")
    except Exception as e:
        print(f"Error during y scaling: {e}")
        return None, None, None, None, None, None, 0, None

    #conversione in pytensors
    X_train_tensor = torch.tensor(X_train_processed.astype(np.float32))
    y_train_tensor = torch.tensor(y_train_scaled.astype(np.float32))
    X_val_tensor = torch.tensor(X_val_processed.astype(np.float32))
    y_val_tensor = torch.tensor(y_val_scaled.astype(np.float32))
    X_test_tensor = torch.tensor(X_test_processed.astype(np.float32))
    y_test_tensor = torch.tensor(y_test_scaled.astype(np.float32))

    input_size = X_train_processed.shape[1]

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, input_size, y_scaler


# Example placeholder for what your data loading should provide:
def your_data_pipeline_function(csv_path, target_column_name):
    # This is where your full 'load_and_preprocess_data' logic would go.
    # CRITICAL: Ensure y_train_raw, y_val_raw, y_test_raw are on the ORIGINAL 0-100 scale.
    # It should also return 'preprocessor_X' (the fitted ColumnTransformer).
    
    # For now, this is a placeholder showing the expected outputs.
    # Replace with your actual data loading and preprocessing.
    print("Placeholder: Your data loading and preprocessing function needs to be called here.")
    print("It should return: X_train_p, y_train_r, X_val_p, y_val_r, X_test_p, y_test_r, preprocessor_X_fitted")
    
    # Dummy data for example structure (replace with your actual data)
    n_train, n_val, n_test = 3000, 1000, 1000
    n_features_processed = 28 # Example: number of features after OHE
    X_train_p = np.random.rand(n_train, n_features_processed)
    y_train_r = pd.Series(np.random.randint(0, 101, n_train))
    X_val_p = np.random.rand(n_val, n_features_processed)
    y_val_r = pd.Series(np.random.randint(0, 101, n_val))
    X_test_p = np.random.rand(n_test, n_features_processed)
    y_test_r = pd.Series(np.random.randint(0, 101, n_test))
    
    # Dummy preprocessor for feature names (replace with your actual one)
    class DummyPreprocessor:
        def get_feature_names_out(self):
            return [f"feature_{i}" for i in range(n_features_processed)]
    preprocessor_X_fitted = DummyPreprocessor()
    
    return X_train_p, y_train_r, X_val_p, y_val_r, X_test_p, y_test_r, preprocessor_X_fitted

# --- XGBoost Training and Evaluation Function ---
def train_and_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor_for_names):
    print("\n--- Training XGBoost Regressor ---")
    
    # Initialize XGBoost Regressor
    # These are some common starting hyperparameters; you can tune them later.
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',  # For regression task
        n_estimators=1000,             # Max number of trees (boosting rounds)
        learning_rate=0.05,            # Lower learning rate often requires more n_estimators
        max_depth=5,                   # Max depth of individual trees
        subsample=0.8,                 # Fraction of samples used for fitting the individual trees
        colsample_bytree=0.8,          # Fraction of features used for fitting the individual trees
        random_state=42,
        early_stopping_rounds=50,
        n_jobs=-12                      # Use all available CPU cores for training
    )

    print("Fitting XGBoost model...")
    # Train the model with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],      # Use validation set for early stopping
      # Stop if performance on eval_set doesn't improve for 50 rounds
        verbose=False                  # Set to True or a number (e.g., 100) to see training progress
    )
    print(f"XGBoost training completed. Best number of trees: {xgb_model.best_iteration}")

    # Make predictions
    y_pred_val = xgb_model.predict(X_val)
    y_pred_test = xgb_model.predict(X_test)

    # Evaluate the model
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    print(f"\nXGBoost Validation RMSE: {val_rmse:.4f}")
    print(f"XGBoost Validation MAE:  {val_mae:.4f}")

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    print(f"XGBoost Test RMSE: {test_rmse:.4f}")
    print(f"XGBoost Test MAE:  {test_mae:.4f}")

    # Display Feature Importance
    try:
        feature_names = preprocessor_for_names.get_feature_names_out()
        importances = xgb_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        print("\nTop 15 Feature Importances (XGBoost):")
        for i in range(min(15, len(feature_names))):
            print(f"  {feature_names[sorted_indices[i]]}: {importances[sorted_indices[i]]:.4f}")
    except Exception as e:
        print(f"\nCould not display feature importances with names: {e}")
        print("Showing importances by index instead:")
        importances = xgb_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        for i in range(min(15, len(importances))): # Print top 15 or fewer if not enough features
            print(f"  Feature index {sorted_indices[i]}: {importances[sorted_indices[i]]:.4f}")
            
    return xgb_model, test_rmse, test_mae

# --- Main execution ---
if __name__ == "__main__":
    TARGET_COLUMN_NAME = 'track_popularity' # As in your script
    CSV_FILE_PATH = 'kaggle_data/spotify_songs.csv' # Your CSV path

    # 1. Load and preprocess data using your existing pipeline
    #    Ensure this function returns the processed X sets, raw Y sets, and the fitted preprocessor
    print("Loading and preprocessing data...")
    # You need to call YOUR actual data loading/preprocessing function here.
    # The line below is using the placeholder.
    X_train_p, y_train_r, X_val_p, y_val_r, X_test_p, y_test_r, preprocessor_X_fitted = \
        your_data_pipeline_function(CSV_FILE_PATH, TARGET_COLUMN_NAME)

    # 2. Train and evaluate XGBoost
    if X_train_p is not None: # Check if data loading was successful
        print("Data loaded. Starting XGBoost training and evaluation.")
        _, xgb_test_rmse, xgb_test_mae = train_and_evaluate_xgboost(
            X_train_p, y_train_r,
            X_val_p, y_val_r,
            X_test_p, y_test_r,
            preprocessor_X_fitted
        )
        print(f"\n--- XGBoost Benchmark Complete ---")
        print(f"Final XGBoost Test RMSE: {xgb_test_rmse:.2f}")
        print(f"Final XGBoost Test MAE:  {xgb_test_mae:.2f}")
        print("Compare this with your Neural Network's performance.")
    else:
        print("Data loading failed. Cannot run XGBoost model.")