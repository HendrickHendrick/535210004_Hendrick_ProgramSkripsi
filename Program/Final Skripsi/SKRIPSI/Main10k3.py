import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-GUI
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import os

def train_models():

    # Path untuk menyimpan model
    rf_model_path = 'models/random_forest_model10k3.joblib'
    xgb_model_path = 'models/xgboost_model10k3.joblib'

    # Load dataset
    file_path = 'C:\\Users\\User\\Desktop\\Final Skripsi\\SKRIPSI\\Mental_Health_10k(3).csv'
    df = pd.read_csv(file_path)

    # Dataset sebelum isi missing value
    df_before = df.copy()

    # Data preprocessing
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Mencari kolom dan baris yang nilai awalnya NaN tetapi sekarang sudah diisi
    filled_values = {}
    for col in df.columns:
        filled_rows = df_before[col].isna() & ~df[col].isna()  # Baris yang awalnya NaN dan sekarang sudah diisi
        if filled_rows.any():  # Jika ada yang diisi
            filled_values[col] = df.loc[filled_rows, col]

    # Menampilkan hasil imputasi
    for col, values in filled_values.items():
        print(f"Kolom '{col}' - Nilai yang diisi:")
        print(values)
        print("\n")

    # Feature selection
    X = df.drop('Diagnosed_by_professional', axis=1)
    y = df['Diagnosed_by_professional']

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Konversi kolom boolean menjadi numerik
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # One hot encoding
    X = pd.get_dummies(X)

    df.info();

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ### Random Forest ###
    if os.path.exists(rf_model_path):
        # Load the pre-trained model
        print("Loading pre-trained Random Forest model...")
        rf_model = joblib.load(rf_model_path)
    else:
        # Train Random Forest if not already trained
        print("Training Random Forest model...")
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt']
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(
            estimator=rf, param_grid=param_grid_rf, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        # Save the trained model
        joblib.dump(grid_search_rf.best_estimator_, rf_model_path)
        rf_model = grid_search_rf.best_estimator_

    ### Confusion Matrix untuk Data Training - Random Forest ###
    y_train_pred_rf = rf_model.predict(X_train)
    cm_train_rf = confusion_matrix(y_train, y_train_pred_rf)
    TN_train_rf, FP_train_rf, FN_train_rf, TP_train_rf = cm_train_rf.ravel()
    TN_train_rf, FP_train_rf, FN_train_rf, TP_train_rf = map(int, [TN_train_rf, FP_train_rf, FN_train_rf, TP_train_rf])

    print(f"Confusion Matrix Data Training - Random Forest")
    print(f"TP: {TP_train_rf}, TN: {TN_train_rf}, FP: {FP_train_rf}, FN: {FN_train_rf}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train_rf, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Training - Random Forest')
    plt.savefig('static/confusion_matrix_rf_training10k3.png')
    plt.close()

    # Make predictions with Random Forest
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf) 
    precision_rf = precision_score(y_test, y_pred_rf, average='binary')
    recall_rf = recall_score(y_test, y_pred_rf, average='binary')
    f1_rf = f1_score(y_test, y_pred_rf, average='binary')
    print(f'Random Forest => Accuracy Random Forest: {accuracy_rf:.2f}, Precision: {precision_rf:.2f}, Recall: {recall_rf:.2f}, F1-Score: {f1_rf:.2f}')
    
    # Confusion Matrix Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    TN_rf, FP_rf, FN_rf, TP_rf = cm_rf.ravel()  # Ekstrak nilai TP, TN, FP, FN
    TN_rf = int(TN_rf)
    FP_rf = int(FP_rf)
    FN_rf = int(FN_rf)
    TP_rf = int(TP_rf)
    print(f"Random Forest Confusion Matrix - TP: {TP_rf}, TN: {TN_rf}, FP: {FP_rf}, FN: {FN_rf}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Random Forest')
    plt.savefig('static/confusion_matrix_rf10k3.png')
    plt.close()

    ### XGBoost ###
    if os.path.exists(xgb_model_path):
        # Load the pre-trained model
        print("Loading pre-trained XGBoost model...")
        xgb_model = joblib.load(xgb_model_path)
    else:
        # Train XGBoost if not already trained
        print("Training XGBoost model...")
        param_grid_xgb = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.3],
            'reg_lambda': [1, 1.5, 2],
            'reg_alpha': [0, 0.1, 0.5]
        }

        xgb_model = xgb.XGBClassifier(eval_metric='logloss')
        grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
        grid_search_xgb.fit(X_train, y_train)
        # Save the trained model
        joblib.dump(grid_search_xgb.best_estimator_, xgb_model_path)
        xgb_model = grid_search_xgb.best_estimator_

    ### Confusion Matrix untuk Data Training - XGBoost ###
    y_train_pred_xgb = xgb_model.predict(X_train)
    cm_train_xgb = confusion_matrix(y_train, y_train_pred_xgb)
    TN_train_xgb, FP_train_xgb, FN_train_xgb, TP_train_xgb = cm_train_xgb.ravel()
    TN_train_xgb, FP_train_xgb, FN_train_xgb, TP_train_xgb = map(int, [TN_train_xgb, FP_train_xgb, FN_train_xgb, TP_train_xgb])

    print(f"Confusion Matrix Data Training - XGBoost")
    print(f"TP: {TP_train_xgb}, TN: {TN_train_xgb}, FP: {FP_train_xgb}, FN: {FN_train_xgb}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train_xgb, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Training - XGBoost')
    plt.savefig('static/confusion_matrix_xgb_training10k3.png')
    plt.close()

    # Make predictions with XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    # Menambahkan perhitungan Precision, Recall, dan F1-score untuk XGBoost
    precision_xgb = precision_score(y_test, y_pred_xgb, average='binary')
    recall_xgb = recall_score(y_test, y_pred_xgb, average='binary')
    f1_xgb = f1_score(y_test, y_pred_xgb, average='binary')
    print(f'XGBoost => Accuracy XGBoost: {accuracy_xgb:.2f}, Precision: {precision_xgb:.2f}, Recall: {recall_xgb:.2f}, F1-Score: {f1_xgb:.2f}')

    # Confusion Matrix XGBoost
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    TN_xgb, FP_xgb, FN_xgb, TP_xgb = cm_xgb.ravel()  # Ekstrak nilai TP, TN, FP, FN
    # Ubah menjadi int64
    TN_xgb = int(TN_xgb)
    FP_xgb = int(FP_xgb)
    FN_xgb = int(FN_xgb)
    TP_xgb = int(TP_xgb)
    print(f"XGBoost Confusion Matrix - TP: {TP_xgb}, TN: {TN_xgb}, FP: {FP_xgb}, FN: {FN_xgb}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix XGBoost')
    plt.savefig('static/confusion_matrix_xgb10k3.png')
    plt.close()

    ### Comparison Chart ###
    accuracies = [accuracy_rf, accuracy_xgb]
    methods = ['Random Forest', 'XGBoost']
    plt.figure(figsize=(8, 5))
    plt.bar(methods, accuracies, color=['black', 'grey'])
    plt.ylabel('Akurasi')
    plt.title('Perbandingan Akurasi Metode Random Forest dan XGBoost')
    plt.ylim([0, 1])
    plt.savefig('static/accuracy_comparison10k3.png')
    plt.close()

    # Return results for Flask
    return accuracy_rf, accuracy_xgb, precision_rf, precision_xgb, recall_rf, recall_xgb, f1_rf, f1_xgb, TN_xgb, FP_xgb, FN_xgb, TP_xgb, TN_rf, FP_rf, FN_rf, TP_rf, TN_train_rf, FP_train_rf, FN_train_rf, TP_train_rf, TN_train_xgb, FP_train_xgb, FN_train_xgb, TP_train_xgb
