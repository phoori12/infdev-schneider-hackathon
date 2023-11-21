import argparse
import pandas as pd 
import os
import logging
import sys

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout) 

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
    logging.info("Loading Data then doing Interpolating and Resampling . . .")
    idx = 0
    for f in files:
        if not ((f.endswith('csv') and (('load' in f) or ('gen' in f)))):
            continue
        df_placeholder = pd.read_csv(f)
        df_placeholder = df_placeholder.drop('EndTime', axis=1)

        df_placeholder.iloc[:, -1] = df_placeholder.iloc[:, -1].interpolate(method='linear', limit_direction='both')

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
        idx = idx+1

    logging.info(f"{idx} files have been loaded, interpolated and resampled in to a dataframe")

    df_count = df.size
    logging.info(f"{df_count} data points has been assigned in a Dataframe")
    logging.info("Removing non-green energy sources")

    df = df[(df.PsrType != 'B02') &
            (df.PsrType != 'B03') &
            (df.PsrType != 'B04') &
            (df.PsrType != 'B05') &
            (df.PsrType != 'B06') &
            (df.PsrType != 'B07') &
            (df.PsrType != 'B08') &
            (df.PsrType != 'B14') &
            (df.PsrType != 'B17') &
            (df.PsrType != 'B20')
            ]
    
    # df.to_csv('../data/test.csv', index=False)

    logging.info(f"{df_count - df.size} data points has been removed")

    logging.info("Reformatting the dataframe")
    df2 = pd.DataFrame(
        columns=['Country IDs', 'StartTime', 'UnitName', 'Biomass', 'Geothermal', 'Hydro Pumped Storage',
                 'Hydro Run-of-river and poundage', 'Hydro Water Reservoir', 'Marine', 'Other Renewable', 'Solar', 'Wind Offshore',
                 'Wind Onshore', 'Load'])
    
    logging.info("Dataframe formatted columns formated from: ")
    logging.info("-------------------------")
    logging.info(list(df.columns))
    
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
        
        
        # logging.info([country_id, start_time, unit_name,
        #        biomass, geothermal, hydro_pump, hydro_run, hydro_water,
        #        marine, other_renew, solar, wind_off, wind_on, load])
        
    df2 = pd.concat([pd.DataFrame(result_data, columns=['Country IDs', 'StartTime', 'UnitName', 'Biomass', 'Geothermal',
                                                     'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
                                                     'Hydro Water Reservoir', 'Marine','Other Renewable', 'Solar', 'Wind Offshore',
                                                     'Wind Onshore', 'Load'])], ignore_index=True)

    logging.info("To")
    logging.info(list(df2.columns))
    logging.info("-------------------------")
    logging.info(f"Created new dataframe with {df2.size} data points")

    # df2.to_csv('../data/test_formatted.csv', index=False)

    return df2
 

# Functions for cleaning data
def duplicates(df):
    df_cleaned = df.drop_duplicates()
    return df_cleaned

def clean_data(df):
    # df_cleaned = missing_values(df)
    logging.info("Removing possible duplicates from dataframe")
    df_cleaned = duplicates(df)
    logging.info(f"{df.size - df_cleaned.size} duplicate data points removed")

    # df_cleaned.to_csv('../data/test_clean.csv', index=False)
    return df_cleaned


def preprocess_data(df): #
    logging.info("Starting to preprocess the dataframe")

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
    }

    unique_timestamps = sorted(df['StartTime'].unique())
    logging.info("Sorted the dataframe by timestamp and adding the green energy generation then subtracting by the load")
     # Convert 'Load' to numeric
    for timestamp in unique_timestamps:
        # Filter rows for the current timestamp
        timestamp_data = df[df['StartTime'] == timestamp]

        # Calculate the result for each country
        result = timestamp_data.iloc[:, 3:13].sum(axis=1) - timestamp_data['Load']
        # if timestamp_data['Load'].eq(0).any().any():
        #     print(timestamp)

        # Sum up the result for each country
        total_sum = result.groupby(timestamp_data['Country IDs']).sum().to_dict()
        #print(timestamp_data)
        # Map column numbers to names
        total_sum_mapped = {column_mapping.get(key, key): value for key, value in total_sum.items()}

        # Print for debugging
        # print("Timestamp:", timestamp)
        # print("Total Sum Mapped:", total_sum_mapped)

        # Append the results to the list
        aggregated_values.append({'StartTime': timestamp, **total_sum_mapped})

    # Convert the list of dictionaries to a DataFrame
    df_processed = pd.DataFrame(aggregated_values).set_index('StartTime').fillna(0)
    df_processed = df_processed.reindex(sorted(df_processed.columns), axis=1)
    logging.info(f"{df.size-df_processed.size} data points has been processed in to the pre-processed dataframe")
    
    df_processed = df_processed.drop("Surplus_UK",axis=1)
    df_processed = find_max(df_processed)
    
    return df_processed
       
def find_max(df):
    logging.info("Finding the country with the most energy surplus")
    df_copy = df.copy()
    df_size = df.size
    df['Predicted_Surplus_Max'] = df_copy.idxmax(axis=1)
    logging.info("Created Predicted_Surplus_Max column")
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
    }
    df['Predicted_Surplus_Max'] = df['Predicted_Surplus_Max'].map(column_mapping)
    df['Predicted_Surplus_Max'] = df['Predicted_Surplus_Max'].shift(-1)
    df = df.iloc[1:-1]

    logging.info(f"{df.size-df_size} data points has been added to the dataframe")

    return df

def save_data(df, output_file):
    # delete all files in data folder
    logging.info("Cleaning data directory")
    files = os.listdir('data/')
    # Iterate through the files and delete each one
    for file in files:
        file_path = os.path.join('data/', file)
        try:
            if ('test' in file):
                continue
            os.unlink(file_path)
        except Exception as e:
            logging.info(e)
    logging.info(f"Saving dataframe with {df.size} data points")
    df.to_csv(output_file, index=True)
    pass

def fill_data(df):
    df = df.interpolate(method='linear',limit_direction='both', axis=0)
    return df    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data',
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
    main(args.input_file, args.output_file)
    
