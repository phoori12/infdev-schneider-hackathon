# Schneider Electric Europe Data Challenge - Team InfDev

## The Repository
This repository `infdev-schneider-hackathon` is where we develop the code for the Schneider Electric's `EcoForecast: Revolutionizing Green Energy Surplus Prediction in Europe` Chanllenge.

## Installation and Usage
We use Anaconda 3 and Python 3.9.18 on Windows 11.

After You have configured the Anaconda 3 Environment, you can install the dependencies via,

`conda create -n infdev-se --file requirements.txt`

After You've installed the dependencies, you can open the environment with,

`conda activate infdev-se`

Lastly to run our script you can it on the terminal with this command in the scripts folder,

`./run_pipeline.sh <start_date> <end_date> <raw_data_file> <processed_data_file> <model_file> <test_data_file> <predictions_file>`

For Example: 

`./run_pipeline.sh 2020-01-01 2020-01-31 data/raw_data.csv data/processed_data.csv models/model.pkl data/test_data.csv predictions/predictions.json`