import pandas as pd
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import json

def save_predictions(predictions, predictions_file):
   
     
    json_data = {"target": {}}
    for i, num in enumerate((predictions)):
        json_data["target"][str(i+1)] = num

    with open(predictions_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
    pass


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    
    index_split = int(0.8 * len(df))
    train_data = df.iloc[:index_split, :]
    test_data = df.iloc[index_split:, :]
    test_data.to_csv('../data/test_dataset.csv', index=False)
    # Return the training set
    return train_data,test_data


def train_model(df):
        all_forecasts_df = df['StartTime']
        country_columns = ['Surplus_DE', 'Surplus_DK', 'Surplus_HU', 'Surplus_IT', 'Surplus_NL', 'Surplus_PO', 'Surplus_SE', 'Surplus_SP', 'Surplus_UK']
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        for country_column in country_columns:
            country_data = df[['StartTime', country_column]].rename(columns={'StartTime': 'ds', country_column: 'y'})
            
            # Create and fit the Prophet model
            model = Prophet(interval_width=0.95,  daily_seasonality=True)
            model.fit(country_data)
            
            # Make future DataFrame for prediction
            future = model.make_future_dataframe(periods=1, freq='H')  # Adjust the number of periods as needed

            # Predict
            forecast = model.predict(future)

            # Extract the predicted values for the last timestamp
            predicted_value = forecast['yhat'].iloc[-1]
            
            # Store the predicted value for each country (you may want to store it in a dictionary)
           
            country_forecast_df = forecast[['ds', 'yhat']].rename(columns={'yhat': country_column})
            all_forecasts_df  = pd.concat([all_forecasts_df, country_forecast_df[country_column]], axis=1)
            # For simplicity, I'm printing the country and its corresponding predicted value
            print(f'Country: {country_column}, Predicted Surplus: {predicted_value}')
           
        print(all_forecasts_df.head())
        
        df2 = all_forecasts_df.iloc[:,1:]
        all_forecasts_df['Country_with_Surplus'] = df2.idxmax(axis=1)
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
        all_forecasts_df['Country_with_Surplus'] = all_forecasts_df['Country_with_Surplus'].map(column_mapping)
        all_forecasts_df.to_csv('../data/test_train_predicted.csv', index=False)
        return all_forecasts_df

def save_model(model, model_file):
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='../data/test_final.csv',
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='../models/model.pkl',
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    train_data,test_data = split_data(df)
    predictions = train_model(test_data)
    # save_predictions(predictions,'../data/test_val_predicted.csv')
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)

