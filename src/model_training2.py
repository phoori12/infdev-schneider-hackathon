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
import pickle


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    # Convert 'StartTime' column to datetime type
    df['StartTime'] = pd.to_datetime(df['StartTime'])

    # Define the date range for splitting
    start_training_date = '2022-01-01'
    end_testing_date = '2023-01-01'

    # Filter data based on the specified date range
    selected_data = df.loc[(df['StartTime'] >= start_training_date) & (df['StartTime'] < end_testing_date)]

    # Calculate the total number of days in the selected date range
    total_days_in_range = (selected_data['StartTime'].max() - selected_data['StartTime'].min()).days

    # Determine the split date for separating training and testing sets
    split_date = selected_data['StartTime'].min() + pd.to_timedelta(0.8 * total_days_in_range, unit='D')

    # Create training and testing sets
    training_set = selected_data[selected_data['StartTime'] <= split_date]
    testing_set = selected_data[selected_data['StartTime'] > split_date]

    # Save the testing set to a CSV file
    test_data_path = '../data/test_data.csv'
    testing_set.to_csv(test_data_path, index=False)

    # Display the sizes of the training and testing sets
    print(f"Training Data Size: {len(training_set)}, Test Data Size: {len(testing_set)}")

    # Return the training set
    return training_set

def calculate_accuracy(y_true, y_pred):
    # Implement your custom accuracy calculation based on your specific use case
    # This is a placeholder; replace it with the appropriate calculation for your problem
    accuracy = np.mean(np.abs((y_true - y_pred) / y_true))
    return accuracy

def train_model(training_set):
    trained_models = {}
    countries = ['SDE', 'DK', 'SP', 'UK', 'HU', 'SE', 'IT', 'PO', 'NL']

    for country in countries:
        try:
            # Construct the column name for the specific country
            country_column = f'Surplus_{country}'
    
            # Filter the training set for the specific country
            country_data = training_set[['StartTime', country_column]]

            # # Debugging: Print intermediate results
            # print(f"After filtering for {country} - Rows: {len(country_data)}")
            # print(country_data.head())

            # Rename the columns
            country_data = country_data.rename(columns={'StartTime': 'ds', country_column: 'y'})

            country_data = country_data.sort_values(by=['ds'])
            prophet_data = country_data[['ds', 'y']]

            model = Prophet()
            model.fit(prophet_data)
            trained_models[country] = model

            # Assuming you have a validation set to make predictions on
            forecast = model.predict(prophet_data)
            y_pred = forecast['yhat'].values

            # Calculate and print accuracy
            train_accuracy = calculate_accuracy(prophet_data['y'], y_pred)
            print(f"Train accuracy for {country}: {train_accuracy}")

            # Access training information
            training_info = model.params['history']
            train_loss = training_info['train_loss']  # Train loss (RMSE)
            print(f"Train loss for {country}: {train_loss}")

            print(f"Successfully trained model for {country}")

        except Exception as exception:
            print(f"Error processing {country}: {str(exception)}")

    return trained_models

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
    training_set = split_data(df)
    model = train_model(training_set)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)