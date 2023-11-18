import argparse
import pandas as pd 
import os 
import math
import csv
from dateutil import parser

COUNTRY_ID_MAP = {
    '10YES-REE------0': 0,  # SP
    '10Y1001A1001A92E': 1,  # UK
    '10Y1001A1001A83F': 2,  # DE
    '10Y1001A1001A65H': 3,  # DK
    '10YHU-MAVIR----U': 4,  # HU
    '10YSE-1--------K': 5,  # SE
    '10YIT-GRTN-----B': 6,  # IT
    '10YPL-AREA-----S': 7,  # PO
    '10YNL----------L': 8   # NL
}

    
def load_data(file_path):
    files = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    df = pd.DataFrame()
    for f in files:
        if not ((f.endswith('csv') and (not 'TEST' in f) and (not 'test' in f))):
            continue
        df_placeholder = pd.read_csv(f)
        df_placeholder = df_placeholder.drop('EndTime', axis=1)
        df_placeholder['StartTime'] = df_placeholder['StartTime'].str.replace(r'\+00:00Z', '', regex=True)


        # Convert columns to datetime using pd.to_datetime
        df_placeholder['StartTime'] = pd.to_datetime(df_placeholder['StartTime'], format='%Y-%m-%dT%H:%M')
        df_placeholder.set_index('StartTime', inplace=True)
        agg_dict = {}
        for col in df_placeholder.columns:
            if  col == 'quantity' or col == 'Load':
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'first'
        df_placeholder = df_placeholder.resample('H').agg(agg_dict)
        df_placeholder.reset_index(inplace=True)
        df = pd.concat([df, df_placeholder], ignore_index=True)

    df = df[(df.PsrType != 'B02') |
            (df.PsrType != 'B03') |
            (df.PsrType != 'B04') |
            (df.PsrType != 'B05') |
            (df.PsrType != 'B06') |
            (df.PsrType != 'B07') |
            (df.PsrType != 'B08') |
            (df.PsrType != 'B14') |
            (df.PsrType != 'B15') |
            (df.PsrType != 'B17') |
            (df.PsrType != 'B20')
            ]
    df.to_csv('../data/TEST.csv', index=False)
    print(df)

    df2 = pd.DataFrame(
        columns=['Country IDs', 'StartTime', 'UnitName', 'Biomass', 'Geothermal', 'Hydro Pumped Storage',
                 'Hydro Run-of-river and poundage', 'Hydro Water Reservoir', 'Maring', 'Solar', 'Wind Offshore',
                 'Wind Onshore', 'Load'])
    
    result_data = []

    for _, group in df.groupby(['AreaID', 'StartTime']):
        country_id = COUNTRY_ID_MAP.get(group['AreaID'].iloc[0], 0)
        start_time = group['StartTime'].iloc[0]
        unit_name = group['UnitName'].iloc[0]

        quantities = dict(zip(group['PsrType'], group['quantity']))

        biomass = quantities.get('B01', 0)
        geothermal = quantities.get('B09', 0)
        hydro_pump = quantities.get('B10', 0)
        hydro_run = quantities.get('B11', 0)
        hydro_water = quantities.get('B12', 0)
        marine = quantities.get('B13', 0)
        solar = quantities.get('B16', 0)
        wind_off = quantities.get('B18', 0)
        wind_on = quantities.get('B19', 0)

        load = group['Load'].dropna().iloc[0] if ('Load' in group.columns) and (not group['Load'].dropna().empty) else 0

        result_data.append([country_id, start_time, unit_name, biomass, geothermal, hydro_pump, hydro_run, hydro_water,
                        marine, solar, wind_off, wind_on, load])
        
        print([country_id, start_time, unit_name,
               biomass, geothermal, hydro_pump, hydro_run, hydro_water,
               marine, solar, wind_off, wind_on, load])
        
    df2 = pd.concat([pd.DataFrame(result_data, columns=['Country IDs', 'StartTime', 'UnitName', 'Biomass', 'Geothermal',
                                                     'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
                                                     'Hydro Water Reservoir', 'Maring', 'Solar', 'Wind Offshore',
                                                     'Wind Onshore', 'Load'])], ignore_index=True)

    df2.to_csv('../data/test_formatted.csv', index=False)

    return df2
 

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
    load_data(os.path.join(os.path.split(os.getcwd())[0], 'data'))
    
