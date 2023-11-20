import pandas as pd
import argparse
import torch
from model_training import MulticlassClassification as mcc_model

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_model(model_path):
    model_state_dict = torch.load(model_path)
    model = mcc_model(num_class = 9, num_feature = 9)
    model.load_state_dict(model_state_dict)
    return model

def make_predictions(df, model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # Disable gradient computation during inference
        predictions = model(torch.tensor(df.values))
        
    return predictions

def save_predictions(predictions, predictions_file):
    predictions_list = predictions.tolist()
    json_path = predictions_file
    with open(json_path, 'w') as json_file:
        json.dump(predictions_list, json_file)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data("/home/main/Hackathon/infdev-schneider-hackathon/data/test_final.csv")
    model = load_model("/home/main/Hackathon/infdev-schneider-hackathon/models/model.pt")
    print(df.values.dtype)
    model.eval()
    print(model)
    predictions = make_predictions(df, model)
    save_predictions(predictions, "/home/main/Hackathon/infdev-schneider-hackathon/predictions/predictions.json")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
