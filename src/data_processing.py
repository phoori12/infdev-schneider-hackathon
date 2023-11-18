import argparse
import pandas as pd 
import os 
import csv

def getCoutryId(AreaID):
    if AreaID == '10YES-REE------0':
        return 0 # SP
    elif AreaID == '10Y1001A1001A92E':
        return 1 # UK
    elif AreaID == '10Y1001A1001A83F':
        return 2 # DE
    elif AreaID == '10Y1001A1001A65H':
        return 3 # DK
    elif AreaID == '10YHU-MAVIR----U':
        return 4 # HU
    elif AreaID == '10YSE-1--------K':
        return 5 # SE
    elif AreaID == '10YIT-GRTN-----B':
        return 6 # IT
    elif AreaID == '10YPL-AREA-----S':
        return 7 # PO
    elif AreaID == '10YNL----------L':
        return 8 # NL

def load_data(file_path):
    files = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    df = pd.concat((pd.read_csv(f) for f in files if (f.endswith('csv') and "gen" in f)), ignore_index=True).reset_index()

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
    
    df2 = pd.DataFrame(columns=['Country IDs', 'StartTime', 'EndTime', 'UnitName', 'Biomass', 'Geothermal', 'Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Hydro Water Reservoir', 'Maring', 'Solar', 'Wind Offshore', 'Wind Onshore'])
    for row in df.itertuples():
        Country_id = 0
        Start_time = 0
        End_time = 0
        Unit_name = ""
        Biomass = 0
        Geothermal = 0
        Hydro_pump = 0
        Hydro_run = 0
        Hydro_water = 0
        Marine = 0
        Solar = 0
        Wind_off = 0
        Wind_on = 0
        Country_id = getCoutryId(row.AreaID)
        Start_time = row.StartTime
        End_time = row.EndTime
        Unit_name = row.UnitName

        df_lookup = df.loc[(df['AreaID'] == row.AreaID) & (df['StartTime'] == Start_time) & (df['EndTime'] == End_time)]
        
        for j_row in df_lookup.itertuples():
            if j_row.PsrType == 'B01':
                Biomass = j_row.quantity
            elif j_row.PsrType == 'B09':
                Geothermal = j_row.quantity
            elif j_row.PsrType == 'B10':
                Hydro_pump = j_row.quantity
            elif j_row.PsrType == 'B11':
                Hydro_run = j_row.quantity
            elif j_row.PsrType == 'B12':
                Hydro_water = j_row.quantity
            elif j_row.PsrType == 'B13':
                Marine = j_row.quantity
            elif j_row.PsrType == 'B16':
                Solar = j_row.quantity
            elif j_row.PsrType == 'B18':
                Wind_off = j_row.quantity
            elif j_row.PsrType == 'B19':
                Wind_on = j_row.quantity

        
        df2 = pd.concat([df2, pd.DataFrame([Country_id, 
                    Start_time,
                    End_time,
                    Unit_name,
                    Biomass,
                    Geothermal,
                    Hydro_pump,
                    Hydro_run,
                    Hydro_water,
                    Marine,
                    Solar,
                    Wind_off,
                    Wind_on])], ignore_index=True)
        print(([Country_id, 
                        Start_time,
                        End_time,
                        Unit_name,
                        Biomass,
                        Geothermal,
                        Hydro_pump,
                        Hydro_run,
                        Hydro_water,
                        Marine,
                        Solar,
                        Wind_off,
                        Wind_on]))


    df2.to_csv('../data/test.csv', index=False)
    
    return df2

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
    load_data(os.path.join(os.path.split(os.getcwd())[0], 'data'))
    