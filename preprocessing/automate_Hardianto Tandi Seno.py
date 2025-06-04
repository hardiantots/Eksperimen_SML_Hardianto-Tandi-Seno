import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def preprocess_and_manual_split(input_path, output_folder, split_ratio=0.9):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Memastikan folder output '{output_folder}' ada.")

    try:
        df = pd.read_csv(input_path)
        print("File input berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File input tidak ditemukan di '{input_path}'. Harap periksa path file.")
        return

    if 'Experience' in df.columns:
        df['Experience'] = abs(df['Experience'])

    drop_columns = ['ID', 'ZIP Code', 'Experience']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    print("Pra-pemrosesan awal selesai.")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Data telah diacak.")

    split_index = int(len(df_shuffled) * split_ratio)
    train_df = df_shuffled.iloc[:split_index]
    test_df = df_shuffled.iloc[split_index:]
    print(f"Data dibagi menjadi {len(train_df)} baris untuk pelatihan dan {len(test_df)} baris untuk pengujian.")

    selected_features = ['Age', 'Income', 'Family', 'CCAvg', 'Education',
                         'Mortgage', 'Securities Account', 'CD Account',
                         'Online', 'CreditCard']
    
    available_features = [f for f in selected_features if f in train_df.columns]
    scaler = StandardScaler()

    train_df[available_features] = scaler.fit_transform(train_df[available_features])
    test_df[available_features] = scaler.transform(test_df[available_features])
    print("Scaling fitur telah diterapkan.")

    if 'Personal Loan' in test_df.columns:
        test_df = test_df.drop(columns=['Personal Loan'])
        print("Kolom 'Personal Loan' telah dihapus dari data uji.")
    
    # Tentukan path output
    output_train_path = os.path.join(output_folder, "Bank_Personal_Loan_preprocessing.csv")
    output_test_path = os.path.join(output_folder, "Bank_Personal_Loan_test.csv")

    # Simpan ke CSV
    train_df.to_csv(output_train_path, index=False)
    print(f"✅ File data latih disimpan di: {output_train_path}")

    test_df.to_csv(output_test_path, index=False)
    print(f"✅ File data uji disimpan di: {output_test_path}")

# Blok utama untuk menjalankan skrip
if __name__ == "__main__":
    INPUT_CSV_FILE = "Bank_Personal_Loan.csv"
    OUTPUT_FOLDER_NAME = "preprocessing/Bank_Personal_Loan"
    
    preprocess_and_manual_split(INPUT_CSV_FILE, OUTPUT_FOLDER_NAME)
