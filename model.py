import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset # Added
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 256 # prima batchnorm, sperimentabile
TARGET_COLUMN_NAME = 'track_popularity'
CSV_FILE_PATH = 'kaggle_data/spotify_songs.csv'

def define_model(input_size, device):
    hidden1_size = 128 #provo non overfittare
    hidden2_size = 64 #sperimento con valori più alti

    model = nn.Sequential(
        nn.Linear(input_size, hidden1_size),
        nn.BatchNorm1d(hidden1_size),   #aggiungo batch normalization per convergenza
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden1_size, hidden2_size),
        nn.BatchNorm1d(hidden2_size),
        nn.ReLU(),
        nn.Dropout(0.2),  #cercando soluzione overfitting
        nn.Linear(hidden2_size, 1) 
    )
    #provo a caricare il modello migliore, cancellare se voglio riprovare da capo
    try:
        #nobatch model.load_state_dict(torch.load("music_regressor_model_best_val.pth", weights_only=True))
        model.load_state_dict(torch.load("music_regressor_model_best_val_batch.pth", map_location="cpu"))
        # model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Initializing a new model.")
    except RuntimeError as e:
         print(f"Error loading state_dict (may be due to architecture mismatch): {e}. \nInitializing new model.")
    model.to(device) # provo usare gpu per velocità
    return model

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'


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
        'release_year'
    ]
    # gestione data
    if 'track_album_release_date' in df.columns:
        try:
            df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')
            df['track_album_release_date'] = df['track_album_release_date'].ffill() # Forward fill missing dates

            current_year = pd.to_datetime('today').year
            df['release_year'] = df['track_album_release_date'].dt.year
            df['release_month'] = df['track_album_release_date'].dt.month
            df['release_season'] = df['release_month'].apply(get_season)    #provo aggiungere stagione di uscita
            features_to_drop.append('release_month')
            df['album_age_years'] = current_year - df['release_year']
            features_to_drop.append('track_album_release_date')
        except Exception as e:
            print(f"Warning: Could not process 'track_album_release_date': {e}. It will be dropped.")
            if 'track_album_release_date' not in features_to_drop:
                 features_to_drop.append('track_album_release_date')

    y_raw = df[target_column_name].copy()
    X_raw = df.drop(columns=[target_column_name] + features_to_drop, errors='ignore').copy()
    # Sperimento con manipolazione dati
    X_raw['acousticness_minus_instrumentalness'] = X_raw['acousticness'] - X_raw['instrumentalness']


    numerical_features = X_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_raw.select_dtypes(include='object').columns.tolist()

    potential_cats_numeric = ['key', 'mode']
    for p_cat in potential_cats_numeric:
        if p_cat in X_raw.columns: # Check if exist
            if p_cat in numerical_features:
                numerical_features.remove(p_cat)
                categorical_features.append(p_cat)
            elif p_cat not in categorical_features:
                 categorical_features.append(p_cat)

    current_X_cols = X_raw.columns.tolist()
    numerical_features = [f for f in numerical_features if f in current_X_cols]
    categorical_features = [f for f in categorical_features if f in current_X_cols]
    print(f"Number of X features: {len(numerical_features) + len(categorical_features)}")
    X_train_val_raw, X_test_raw, y_train_val_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=42
    )

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_val_raw, y_train_val_raw, test_size=0.1764, random_state=42 # 0.1764 = 15% of 85%
    )
    # appr. 68% Train, 17% Val, 15% Test


    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        #('scaler', StandardScaler())])
        #('scaler', RobustScaler())])
        ('scaler', MinMaxScaler())]) #provo a vedere se migliora
    categorical_transformer_X = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # print(f"Unique playlist_genre values: {X_raw['playlist_genre'].nunique()}")
    # print(f"Unique playlist_subgenre values: {X_raw['playlist_subgenre'].nunique()}")
    print(f"numerical features: {numerical_features}, \n categorical features: {categorical_features}")

    #corr matrix
    corr_matrix = df.corr(numeric_only=True)
    print("\nCorrelation matrix:")
    print(corr_matrix[target_column_name].sort_values(ascending=False))
    for cat in categorical_features:
        print(f"\n '{target_column_name}' median per Cat. sFeature '{cat}':")
        print(df.groupby(cat)[target_column_name].mean().sort_values(ascending=False))
    #heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Purples", square=True)
    plt.title("Numerical Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
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


# --- NO BATCH TRAIN ---
def train_step_func(model, input_batch, target_batch, criterion, optimizer):
    model.train() #IMPORTANTE cambiare modalitè
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch.unsqueeze(1))
    loss.backward()
    optimizer.step()
    return loss.item()

#--- BATCH TRAIN ---
def train_step_func(model, data_loader, criterion, optimizer, device):
    model.train() #IMPORTANTE cambiare modalitè
    epoch_loss = 0
    for input_batch, target_batch in data_loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * input_batch.size(0) # loss accumulata
    return epoch_loss / len(data_loader.dataset) #media

# --- NO BATCH EVAL ---
def evaluate_model(model, input_data, target_data, criterion, y_scaler):
    model.eval() # ANCHE QUI
    with torch.no_grad():
        output = model(input_data)
        loss = criterion(output, target_data.unsqueeze(1))
        predictions_scaled = output.squeeze(1).numpy()
        target_scaled = target_data.numpy()
        predictions_original = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        target_original = y_scaler.inverse_transform(target_scaled.reshape(-1, 1)).flatten()
        mae_original = mean_absolute_error(target_original, predictions_original)
        rmse_original = np.sqrt(mean_squared_error(target_original, predictions_original))
    return loss.item(), mae_original, rmse_original

# --- BATCH EVAL ---
def evaluate_model(model, data_loader, criterion, y_scaler, device):
    model.eval() # ANCHE QUI
    total_loss = 0
    predictions_scaled = []
    target_scaled = []
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            output = model(input_batch)
            loss = criterion(output, target_batch.unsqueeze(1))
            total_loss += loss.item() * input_batch.size(0) # loss accumulata

            predictions_scaled.extend(output.squeeze(1).cpu().numpy())
            target_scaled.extend(target_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    
    predictions_scaled_np = np.array(predictions_scaled)
    targets_scaled_np = np.array(target_scaled)

    predictions_original = y_scaler.inverse_transform(predictions_scaled_np.reshape(-1, 1)).flatten()
    target_original = y_scaler.inverse_transform(targets_scaled_np.reshape(-1, 1)).flatten()

    mae_original = mean_absolute_error(target_original, predictions_original)
    rmse_original = np.sqrt(mean_squared_error(target_original, predictions_original))
    
    return avg_loss, mae_original, rmse_original


def main():
    #setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device=torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # IPERPARAMETRI GIGA IMPORTANTI
    learning_rate = 0.0005
    epochs = 250 
    check = 10  #numero di cicli per controllo/validazione
    patience = 10 #abbasso visto il plateau drammatico 
    min_delta = 0.0006 
    weight_decay = 0.0001 #regolarizzazione L2 

    print("1. Loading and Preprocessing Data...")
    train_input, train_target, val_input, val_target, test_input, test_target, input_size, y_scaler = \
        load_and_preprocess_data(CSV_FILE_PATH, TARGET_COLUMN_NAME)

    if train_input is None or input_size == 0:
        print("Exiting due to data loading or preprocessing errors.")
        return
    
    #num_workers = 4 if device.type == 'cuda' else 0 # da vedere
    num_workers=0
    pin_memory = True if device.type == 'cuda' else False
    print(f"Using DataLoader with num_workers={num_workers}, pin_memory={pin_memory}")

    #proviamo dataLoading
    train_dataset = TensorDataset(train_input, train_target)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = TensorDataset(val_input, val_target)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dataset = TensorDataset(test_input, test_target)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"   Input feature size: {input_size}")
    print(f"   Loaded {train_input.shape[0]} training samples.")
    print(f"   Loaded {val_input.shape[0]} validation samples.")
    print(f"   Loaded {test_input.shape[0]} testing samples.")
    print(f"   Target variable '{TARGET_COLUMN_NAME}' is numerical.")

    print("\n2. Defining Model...")
    model = define_model(input_size, device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #aggiunta regolarizzazione per overfitting  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, min_lr=1e-6) #lrScheduling per il meme

    print("\n3. Starting Training with Early Stopping...")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        #commento per no batch train_loss = train_step_func(model, train_input, train_target, criterion, optimizer)
        #batchversion
        train_loss = train_step_func(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % check == 0:
            # validazione
            #commento no batch val_loss, val_mae, val_rmse = evaluate_model(model, val_input, val_target, criterion, y_scaler)
            val_loss, val_mae, val_rmse = evaluate_model(model, val_loader, criterion, y_scaler, device)
            scheduler.step(val_loss)

            print(f'Epoch [{epoch+1}/{epochs}],Learning Rate: {scheduler.get_last_lr()}, Train Loss (MSE): {train_loss:.4f}, Val Loss (MSE): {val_loss:.4f}, Val MAE (Orig): {val_mae:.2f}, Val RMSE (Orig): {val_rmse:.2f}')
            # stop prematuro su min_delta
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
                # salvo best model
                best_model_state = model.state_dict().copy()
                print(f"   Validation loss improved to {best_val_loss:.4f}. Saving model state.") 
            else:
                epochs_no_improve += 1
                # print(f"   Validation loss didn't improve. Patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                    print(f"\tLoaded model state from epoch {best_epoch} (best validation loss).")
                break 
    if epochs_no_improve < patience and best_model_state:
         model.load_state_dict(best_model_state)
         print(f"\nTraining finished (reached max epochs). Loaded model state from epoch {best_epoch} (best validation loss).")
    elif not best_model_state:
         print("\nTraining finished, but no model state was saved (validation loss never decreased). Using final epoch model state.")


    print("\n4.\tTraining Finished!")

    print("\n5.\tEvaluating on Full Test Set (Metrics on Original Scale)...")
    
    #commento per nobatch final_test_loss, final_test_mae_orig, final_test_rmse_orig = evaluate_model(model, test_input, test_target, criterion, y_scaler)
    #batchversion
    final_test_loss, final_test_mae_orig, final_test_rmse_orig = evaluate_model(model, test_loader, criterion, y_scaler, device)
    print(f'\tFinal Test Loss (MSE SCALED): {final_test_loss:.4f}')
    print(f'\tFinal Test MAE (Original): {final_test_mae_orig:.2f}')
    print(f'\tFinal Test RMSE (Original): {final_test_rmse_orig:.2f}')


    print("\n6.\tSaving Model...")
    try:
        #nobatch torch.save(model.state_dict(), 'music_regressor_model_best_val.pth') # Changed filename to indicate best val
        #batchversion
        torch.save(model.state_dict(), 'music_regressor_model_best_val_batch.pth') # Changed filename to indicate best val

        print("\tBest model state saved to music_regressor_model_best_val_batch.pth")
    except Exception as e:
        print(f"\tError saving model: {e}")

if __name__ == "__main__":
    main()