import argparse
import pandas as pd 
import glob 
import os 
import csv



def load_data(file_path):

    files = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    df = pd.concat((pd.read_csv(f) for f in files if f.endswith('csv')), ignore_index=True).reset_index()
    
    df =  df[ (df.PsrType !=  'B02')|
              (df.PsrType !=  'B03')|
              (df.PsrType !=  'B04')|
              (df.PsrType !=  'B05')|
              (df.PsrType !=  'B06')|
              (df.PsrType !=  'B07')|
              (df.PsrType !=  'B08')|
              (df.PsrType !=  'B14')|
              (df.PsrType !=  'B15')|
              (df.PsrType !=  'B17')|
              (df.PsrType !=  'B20')
            ]
    print(df.head())
    df.to_csv('../data/TEST.csv', index=False)
    
    return df

# Functions for clean_data
def missing_values(df, column_to_ignore):
    df_cleaned = df.dropna(subset=df.columns.difference([column_to_ignore]))
    return df_cleaned

def duplicates(df):
    df_cleaned = df.drop_duplicates()
    return df_cleaned

def remove_outliers_iqr(df, column_name, threshold=1.5):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    cleaned_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return cleaned_df

def clean_data(df):
    df_clean = missing_values(df, "Load")
    df_clean = duplicates(df_clean)
    df_clean = remove_outliers_iqr(df_clean, "quantity")

    df_clean.to_csv('../data/df2.csv', index=False)
    return df_clean

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.

    return df_processed

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    ## main(args.input_file, args.output_file)
    x = load_data()
    clean_data(x)