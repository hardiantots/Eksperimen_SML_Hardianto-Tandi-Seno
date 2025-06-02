import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    if 'Experience' in df.columns:
        df['Experience'] = abs(df['Experience'])

    drop_columns = ['ID', 'ZIP Code', 'Experience']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    selected_features = ['Age', 'Income', 'Family', 'CCAvg', 'Education', 
                         'Mortgage', 'Securities Account', 'CD Account', 
                         'Online', 'CreditCard']
    available = [f for f in selected_features if f in df.columns]

    df[available] = StandardScaler().fit_transform(df[available])
    df.to_csv(output_path, index=False)
    print(f"Preprocessed file saved to: {output_path}")

if __name__ == "__main__":
    input_file = "Bank_Personal_Loan.csv"
    output_file = "preprocessing/Bank_Personal_Loan_preprocessing.csv"
    preprocess_csv(input_file, output_file)