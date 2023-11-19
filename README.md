# Schneider Electric Europe Data Challenge - Team InfDev

## The Repository
This repository `infdev-schneider-hackathon` is where we develop the code for the Schneider Electric's `EcoForecast: Revolutionizing Green Energy Surplus Prediction in Europe` Challenge.

## Installation and Usage
We use Anaconda 3 and Python 3.9.18 on Windows 11.

After You have configured the Anaconda 3 Environment, you can install the dependencies via,

`conda create -n infdev-se --file requirements.txt`

After You've installed the dependencies, you can activate the environment with,

`conda activate infdev-se`

Lastly to run our script you can it on the terminal with this command in the scripts folder,

`./run_pipeline.sh <start_date> <end_date> <raw_data_file> <processed_data_file> <model_file> <test_data_file> <predictions_file>`

For Example: 

`./run_pipeline.sh 2020-01-01 2020-01-31 data/raw_data.csv data/processed_data.csv models/model.pkl data/test_data.csv predictions/predictions.json`

## The Challenge

Our goal is to create a model capable of predicting the country (from a list of nine) that will have the most surplus of green energy in the next hour. The Dataset is already provided from the use of the API of the ENTSO-E Transparency portal. Before using the dataset, we have to filter and clean up the dataset before using it.

## Our Approach

We have splitted our workflows into 3 sections:

* [Data Ingestion and Data Processing] (#data-ingestion-and-data-processing)
* [Model Training] (#model-training)
* [Model Prediction and Re-Evaluation] (#model-prediction-and-re-evaluation)

### Data Ingestion and Data Processing

* Data Ingestion

    As for the Data Ingestion part, we decided to use the code provided from the template to fetch the data through the API. After fetching, we are provided with multiple .csv files which contains the information of the load each countries and amount of energy generation from multiple sources. Which have to be processed later on in `src/data_processing.py`.

* Data Processing

1. As the time series of each .csv may varies. We have decided to resample the data as we load it in a panda dataframe. We resampled it into 1 Hour series, by summing up the values in a smaller time series to reach 1 Hour.


2. After resampling all of the time series in the dataframe, we filter out the unnessesary energy sources because we only want the values from the green energy sources. After filtering We are left with 10 types of green energy sources.

    | Code        | Meaning           |
    | ------------- |:-------------:|
    | B01      | Biomass |
    | B09      | Geothermal      | 
    | B10 | Hydro Pumped Storage     | 
    | B11     | Hydro Run-of-river and poundage |
    | B12     | Hydro Water Reservoir |
    | B13      | Marine |
    | B15      | Other Renewable |
    | B16      | Solar |
    | B18      | Wind Offshore |
    | B19      | Wind Onshore |

3. After acquring all the data and packing them all in a single dataframe. We will have to for format the dataframe columns from:


    | StartTime        | AreaID           | UnitName        | PsrType           | quantity        | Load           |
    | ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|

    To:

    | Country IDs        | StartTime           | UnitName        | Biomass           | Geothermal        | Hydro Pumped Storage           | Hydro Run-of-river and poundage        | Hydro Water Reservoir           | Marine        | Other Renewable           | Solar        | Wind Offshore           | Wind Onshore        | Load           |
    | ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|


4. After acquiring the formatted dataframe, we will clean it with the `data_cleaning`` function.
The Data Cleaning Consists of 3 Process:
    - Removing missing values. All the missing values for each type of green energy were removed. Excluding the load.
    - Removing duplicate rows
    - Removing outliers
        - By using the IQR-Method (Interquartile Range). The reason behind this is, the mean is not reliable and stable and could be highly affected by these outliers or so called anamolies. By calculation first quartile (25% of the data sorted) and third quartile, the IQR is the difference between these two. The accepted datas are the datas that are between the Q1 and Q3. These outliers are identified and remove depending on the 
        countries and done for each column


5. After the Cleaning process, we will reformat the dataframe columns again to:

    | StartTime        | Surplus_DE           | Surplus_DK        | Surplus_HU           | Surplus_IT        | Surplus_NL           | Surplus_PO        | Surplus_SE           | Surplus_SP        | Surplus_UK        | Surplus_Max           |
    | ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|

    - When doing the operation of summing the total amount of energy produced and deduct it by the amount of load. A problem occurs, which is that some of the timestamp don't match. 
    - We decided to approach this by setting the timestamp as indices and going through it one by one. If a data is not present in the timestamp we may use the mean of the preceding and following values to assign the missing value, which is demonstrated in the `fill_Data` method.
    - On worst case, like the data we got from the uk, in which many values are missing and cannot be interpolated at all. We decided to set the values to 0.
    - In the `find_max` method, where we implement the function to find the maximum country with the energy surplus, we have to also add an extra condition that if a country has 0 surplus, we will ignore that country.

### Model Training

### Model Prediction and Re-Evaluation


