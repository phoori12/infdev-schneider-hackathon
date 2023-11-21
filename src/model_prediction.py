import pandas as pd
import argparse
import torch
from model_training import MulticlassClassification as mcc_model
import json
import logging
import sys

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout) 

# load validation dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# redefine model and give the weigh after training to the model
def load_model(model_path):
    model_state_dict = torch.load(model_path) # load best_state_dict
    model = mcc_model(num_feature = 8, num_class = 8)
    model.load_state_dict(model_state_dict)
    return model

def make_predictions(df, model):
    logging.info("Proceeding to make predictions")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = df.iloc[:,:-1]
    model.to(device)

    #Evaluation mode
    model.eval()
    
    input_data = torch.tensor(df.values, dtype=torch.float32)  # Adjust the dtype accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)

    # Perform forward pass
    with torch.no_grad():
        output = model(input_data)
    
    # print(output)
    # Apply softmax to convert raw scores to probabilities
    probabilities = torch.log_softmax(output, dim = 1)

    # For each sample, get the predicted class
    _, predicted_classes = torch.max(probabilities, 1)

    # Convert the predicted_classes tensor to a Python list
    predicted_classes = predicted_classes.cpu().numpy().tolist()
    
    return predicted_classes

def save_predictions(predictions, predictions_file):
    logging.info("Saving predictions in predictions.json")

    json_data = {"target": {}}
    for i, num in enumerate(predictions):
        if i == 442: # stop at the required size  
            break
        json_data["target"][str(i)] = num
    # to the required structured JSON file
    with open(predictions_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test.csv',
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pt',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions.json',
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
