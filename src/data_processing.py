import argparse
import pandas as pd 
import os
import numpy as np
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
        if not ((f.endswith('csv') and (not 'test' in f))):
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
            (df.PsrType != 'B17') |
            (df.PsrType != 'B20')
            ]
    df.to_csv('../data/test.csv', index=False)

    df2 = pd.DataFrame(
        columns=['Country IDs', 'StartTime', 'UnitName', 'Biomass', 'Geothermal', 'Hydro Pumped Storage',
                 'Hydro Run-of-river and poundage', 'Hydro Water Reservoir', 'Marine', 'Other Renewable', 'Solar', 'Wind Offshore',
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
        other_renew = quantities.get('B15',0)
        solar = quantities.get('B16', 0)
        wind_off = quantities.get('B18', 0)
        wind_on = quantities.get('B19', 0)

        load = group['Load'].dropna().iloc[0] if ('Load' in group.columns) and (not group['Load'].dropna().empty) else 0

        result_data.append([country_id, start_time, unit_name, biomass, geothermal, hydro_pump, hydro_run, hydro_water,
                        marine,other_renew,solar, wind_off, wind_on, load])
        
        
        # print([country_id, start_time, unit_name,
        #        biomass, geothermal, hydro_pump, hydro_run, hydro_water,
        #        marine, other_renew, solar, wind_off, wind_on, load])
        
    df2 = pd.concat([pd.DataFrame(result_data, columns=['Country IDs', 'StartTime', 'UnitName', 'Biomass', 'Geothermal',
                                                     'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
                                                     'Hydro Water Reservoir', 'Marine','Other Renewable', 'Solar', 'Wind Offshore',
                                                     'Wind Onshore', 'Load'])], ignore_index=True)

    df2.to_csv('../data/test_formatted.csv', index=False)

    return df2
 

# Functions for clean_data
def missing_values(df):
    df_cleaned = df.loc[df['Load'] != 0]
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

# def remove_rows_with_zero(df):
#     df_cleaned = df[(df != 0).all(axis=1)]
#
#     return df_cleaned

def clean_data(df):
    df_cleaned = missing_values(df)
    df_cleaned = duplicates(df_cleaned)

    # List of unique Country IDs
    unique_country_ids = df_cleaned['Country IDs'].unique()

    cleaned_dfs = []

    for country_id in unique_country_ids:
        country_df = df_cleaned[df_cleaned['Country IDs'] == country_id]

        # Apply remove_outliers_iqr for each relevant column (excluding 'Country IDs', 'StartTime', and 'UnitName')
        for col in df_cleaned.columns:
            if col == 'Country IDs' or col == 'StartTime' or col == 'UnitName':
                continue
            country_df = remove_outliers_iqr(country_df, col)

        cleaned_dfs.append(country_df)

    df_cleaned_final = pd.concat(cleaned_dfs, ignore_index=True)
    # df_cleaned_final = remove_rows_with_zero(df_cleaned_final)
    df.to_csv('../data/test_clean.csv', index=False)
    return df_cleaned_final


def preprocess_data(df): #
    # TODO: Generate new features, transform existing features, resampling, etc.
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%dT%H:%M')

    aggregated_values = []

    column_mapping = {
        0: 'Surplus_SP',
        1: 'Surplus_UK',
        2: 'Surplus_DE',
        3: 'Surplus_DK',
        4: 'Surplus_HU',
        5: 'Surplus_SE',
        6: 'Surplus_IT',
        7: 'Surplus_PO',
        8: 'Surplus_NL',
        # Add more mappings as needed
    }

    unique_timestamps = sorted(df['StartTime'].unique())
    
    for timestamp in unique_timestamps:
        # Filter rows for the current timestamp
        timestamp_data = df[df['StartTime'] == timestamp]

        # Calculate the result for each country
        result = timestamp_data.iloc[:, 3:13].sum(axis=1) - timestamp_data['Load']

        # Sum up the result for each country
        total_sum = result.groupby(timestamp_data['Country IDs']).sum().to_dict()

        # Map column numbers to names
        total_sum_mapped = {column_mapping.get(key, key): value for key, value in total_sum.items()}

        # Print for debugging
        #print("Timestamp:", timestamp)
        #print("Total Sum Mapped:", total_sum_mapped)

        # Append the results to the list
        aggregated_values.append({'StartTime': timestamp, **total_sum_mapped})

    # Convert the list of dictionaries to a DataFrame
    df_processed = pd.DataFrame(aggregated_values).set_index('StartTime').fillna(0)
    #df_processed = df_processed[(df_processed != 0.0).all(axis=1)] # Remove rows with "0.0"
    df_processed = df_processed.reindex(sorted(df_processed.columns), axis=1)

    df_processed = fill_data(df_processed)
    df_processed = find_max(df_processed)
    df_processed = df_processed.replace({0: -99999.0})

    return df_processed
       
def find_max(df):
    # UK Problem with values between 0 and -1 
    df_copy = df.copy()
 
    df_copy[(df_copy >= -1) & (df_copy <= 0)] = np.nan
    df['Predicted_Surplus_Max'] = df_copy.idxmax(axis=1)
    for column_name in df.columns[:-1]:
        df.loc[(df[column_name] >= -1) & (df[column_name] <= 0), column_name] = 0


    column_mapping = {
         'Surplus_SP':0,
         'Surplus_UK':1,
         'Surplus_DE':2,
         'Surplus_DK':3,
         'Surplus_HU':4,
         'Surplus_SE':5,
         'Surplus_IT':6,
         'Surplus_PO':7,
         'Surplus_NL':8,
        # Add more mappings as needed
    }
    df['Predicted_Surplus_Max'] = df['Predicted_Surplus_Max'].map(column_mapping)
    df['Predicted_Surplus_Max'] = df['Predicted_Surplus_Max'].shift(-1)
    df = df.iloc[1:-1]
    return df

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    df.to_csv('../data/test_final.csv', index=True)
    pass

def fill_data(df):
    # df = df.interpolate(method='linear',limit_direction='both', axis=0, fill_value=-99999)
    for column_name in df.columns:
    # Iterate through each row
        for i in range(1, len(df)):
            if df.iloc[i][column_name] == 0:
                # Find the previous non-zero value
                prev_non_zero = df.iloc[i - 1][column_name] if i - 1 >= 0 else 0

                # Find the next non-zero value
                remaining_non_zero = df[column_name].iloc[i:].replace(0, pd.NA).dropna()

                # Check if there are any non-zero values remaining
                if not remaining_non_zero.empty:
                    next_non_zero = remaining_non_zero.iloc[0]
                else:
                    next_non_zero = 0
                
                # Calculate the average of the previous and next non-zero values
                average_value = (prev_non_zero + next_non_zero) / 2
                
                # Set the zero value to the calculated average
                df.iloc[i, df.columns.get_loc(column_name)] = average_value    
    
    
    return df    

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
    df = load_data(os.path.join(os.path.split(os.getcwd())[0], 'data'))
    df = clean_data(df)
    df = preprocess_data(df)
    save_data(df, "idk")

    
