# Schneider Electric Europe Data Challenge - Team InfDev

## The Repository
This repository, infdev-schneider-hackathon, is where we develop code for Schneider Electric's challenge, "EcoForecast: Revolutionizing Green Energy Surplus Prediction in Europe."

## Installation and Usage
We use Anaconda 3 and Python 3.9.18 on Windows 11.

After configuring the Anaconda 3 environment, install dependencies with:

`conda create -n infdev-se --file requirements.txt`

After You've installed the dependencies, activate the environment with,

`conda activate infdev-se`

Lastly to run our script, navigate to the scripts folder and use the following command on the terminal:

`./run_pipeline.sh <start_date> <end_date> <raw_data_file> <processed_data_file> <model_file> <test_data_file> <predictions_file>`

For Example: 

`./run_pipeline.sh 2020-01-01 2020-01-31 data/raw_data.csv data/processed_data.csv models/model.pkl data/test_data.csv predictions/predictions.json`

## The Challenge

Our goal is to create a model capable of predicting the country (from a list of nine) that will have the most surplus of green energy in the next hour. The dataset is already provided through the use of the API of the ENTSO-E Transparency portal. Before utilizing the dataset, we need to filter and clean it up.

## Our Approach

We have divided our workflows into three sections:

* [Data Ingestion and Data Processing](#data-ingestion-and-data-processing)
* [Model Training](#model-training)
* [Model Prediction and Re-Evaluation](#model-prediction-and-re-evaluation)

### Data Ingestion and Data Processing <a name="data-ingestion-and-data-processing"></a>

* Data Ingestion

    For the Data Ingestion part, we decided to use the code provided from the template to fetch the data through the API. After fetching, we are provided with multiple .csv files that contain information about the load in each country and the amount of energy generation from multiple sources. This data will be processed later in src/data_processing.py.

* Data Processing

1. Regarding the time series of each .csv, which may vary, we decided to resample the data as we load it into a pandas dataframe. We resampled it into 1-hour series by summing up the values in smaller time series to reach 1 hour.


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

3. After acquiring all the data and packing them into a single dataframe, we need to format the dataframe columns from:


    | StartTime        | AreaID           | UnitName        | PsrType           | quantity        | Load           |
    | ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|

    To:

    | Country IDs        | StartTime           | UnitName        | Biomass           | Geothermal        | Hydro Pumped Storage           | Hydro Run-of-river and poundage        | Hydro Water Reservoir           | Marine        | Other Renewable           | Solar        | Wind Offshore           | Wind Onshore        | Load           |
    | ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|


4. After acquiring the formatted dataframe, we will clean it with the data_cleaning function. The Data Cleaning consists of three processes:
    - Removing missing values. All the missing values for each type of green energy, excluding the load, were removed.
    - Removing duplicate rows
    - Removing outliers using the IQR-Method (Interquartile Range). The mean is not reliable and stable and could be highly affected by these outliers. By calculating the first quartile (25% of the data sorted) and third quartile (75% of the data sorted), we can calculate the IQR, which is the difference between the first and third quartile. The accepted data are those between Q1 and Q3. Outliers are identified and removed for each column, depending on the countries.


5. After the Cleaning process, we will reformat the dataframe columns again to:

    | StartTime        | Surplus_DE           | Surplus_DK        | Surplus_HU           | Surplus_IT        | Surplus_NL           | Surplus_PO        | Surplus_SE           | Surplus_SP        | Surplus_UK        | Surplus_Max           |
    | ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|

    - When performing the operation of summing the total amount of energy produced and deducting it from the load, a problem occurs. Some timestamps don't match.
    - We decided to approach this by setting the timestamp as indices and going through it one by one. If a data point is not present in the timestamp, we may use the mean of the preceding and following values to assign the missing value, as demonstrated in the fill_Data method.
    - In the worst case, like the data we got from the UK, in which many values are missing and cannot be interpolated at all, we decided to set the values to 0.
    - In the find_max method, where we implement the function to find the maximum country with energy surplus, we also have to add an extra condition that if a country has 0 surplus, we will ignore that country.

### Model Training <a name="model-training"></a>

### Model Prediction and Re-Evaluation <a name="model-prediction-and-re-evaluation"></a>


