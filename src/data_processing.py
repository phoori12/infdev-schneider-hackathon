import argparse
import pandas as pd 
import os 
import csv

def delete_row(csv_file_path, column_name, value_to_delete):
    # Read the CSV file and filter out rows with the specified value in the specified column
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row[column_name] != value_to_delete]

    # Write the remaining rows back to the CSV file
    with open(csv_file_path, 'w', newline='') as file:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def load_data(file_path):
    files = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    df = pd.concat((pd.read_csv(f) for f in files if f.endswith('csv')), ignore_index=True).reset_index()

    df =  df[ (df.PsrType == 'B01')|
              (df.PsrType == 'B09')|
              (df.PsrType == 'B10')|
              (df.PsrType == 'B11')|
              (df.PsrType == 'B12')|
              (df.PsrType == 'B13')|
              (df.PsrType == 'B16')|
              (df.PsrType == 'B18')|
              (df.PsrType == 'B19')
            ]
    
    df.to_csv('../data/TEST.csv', index=False)
    
    return df

def clean_data(df):
    
    
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
    clean_data(load_data(os.path.join(os.path.split(os.getcwd())[0], 'data')))
    