import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(raw_data):
    # 1. Isi missing value kolom total_bedrooms dengan rata-rata
    raw_data['total_bedrooms'] = raw_data['total_bedrooms'].fillna(raw_data['total_bedrooms'].mean())
    
    # 2. Label encoding untuk kolom ocean_proximity
    label_encoder = LabelEncoder()
    raw_data['ocean_proximity'] = label_encoder.fit_transform(raw_data['ocean_proximity'])
    
    # 3. Standarisasi fitur numerik
    features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    scaler = StandardScaler()
    raw_data[features] = scaler.fit_transform(raw_data[features])
    
    return raw_data

def main():
    # Load data
    input_path = 'namadataset_raw/california_housing.csv'
    output_path = 'namadataset_preprocessing/processed_california_housing.csv'
    
    print("Loading data...")
    data = pd.read_csv(input_path)
    
    print("Starting preprocessing...")
    processed = preprocess_data(data)
    
    # Save hasil preprocessing
    processed.to_csv(output_path, index=False)
    
    print(f"Preprocessing selesai. Data tersimpan di {output_path}")

if __name__ == '__main__':
    main()
