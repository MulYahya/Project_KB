# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'data_season.csv'  # Ganti dengan lokasi file Anda
data = pd.read_csv(file_path)

# Pilih kolom yang relevan
selected_data = data[['Rainfall', 'Temperature', 'Humidity', 'Crops']]

# Handle outliers using IQR
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

selected_data = remove_outliers(selected_data, ['Rainfall', 'Temperature', 'Humidity'])

# Tambahkan fitur baru
selected_data['Rainfall_Humidity_Ratio'] = selected_data['Rainfall'] / (selected_data['Humidity'] + 1e-5)
selected_data['Temp_Humidity_Product'] = selected_data['Temperature'] * selected_data['Humidity']

# Tambahkan fitur polinomial
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(selected_data[['Rainfall', 'Temperature', 'Humidity']])
poly_feature_names = poly.get_feature_names_out(['Rainfall', 'Temperature', 'Humidity'])
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# Gabungkan fitur polinomial dengan dataset asli
selected_data = pd.concat([selected_data, poly_df], axis=1)

# Pisahkan data menjadi fitur (X) dan target (y)
X = selected_data.drop(columns=['Crops'])
y = selected_data['Crops']

# Pastikan semua kolom di X sudah numerik
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # Isi nilai NaN jika ada

# Encode label 'Crops' menjadi angka menggunakan LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Konversi label menjadi angka

# Cek tipe data
print("Tipe data fitur (X):")
print(X.dtypes)
print("Label unik pada y_encoded:", np.unique(y_encoded))

# Normalisasi fitur menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Menyeimbangkan data dengan SMOTE (jika diperlukan)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_normalized, y_encoded)

# Bagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning menggunakan Grid Search
param_grid = {
    'n_neighbors': list(range(1, 31)),  # Uji lebih banyak nilai k
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # Untuk metrik Minkowski
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Model terbaik dari Grid Search
best_knn = grid_search.best_estimator_

# Prediksi pada data pengujian
y_pred = best_knn.predict(X_test)

# Evaluasi model
report = classification_report(y_test, y_pred)
print("Best Parameters:", grid_search.best_params_)
print("Classification Report:\n", report)

# Jika ingin mengembalikan prediksi ke label asli
decoded_labels = label_encoder.inverse_transform(y_pred)
print("Predicted Crops:\n", decoded_labels)