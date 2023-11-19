import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from model_training import MulticlassClassification as mcc_model
from model_training import ClassifierDataset
import numpy as np

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
    df = df.iloc[: , 1:11]
    df = df.astype(float)
    print(df.values)
    # x_val = np.array(df.iloc[: , 1:10])
    # y_val = np.array(df.iloc[: , -1])

    #dataset = ClassifierDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())

    model.to(device)
    # model.eval()
    y_pred_list = []

    with torch.no_grad():  # Disable gradient computation during inference
        predictions = model(torch.tensor(df))
    print(predictions.shape)

    # for X_batch, _ in test_loader:
    #     X_batch = X_batch.to(device)
    #     y_test_pred = model(X_batch)
    #     _, y_pred_tags = torch.max(y_test_pred, dim = 1)
    #     y_pred_list.append(y_pred_tags.cpu().numpy())

    # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    # # y_pred_list = np.array(y_pred_list).flatten().tolist()
    # print(y_pred_list)
    
    return y_pred_list


def save_predictions(predictions, predictions_file):
    predictions_list = predictions.tolist()
    
    # print(predictions_list)
    # json_path = predictions_file
    # with open(json_path, 'w') as json_file:
    #     json.dump(predictions_list, json_file)
    # pass

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
    df = load_data("../data/test_final.csv")
    model = load_model("../models/model.pt")
    model.eval()
    print(model)
    predictions = make_predictions(df, model)
    save_predictions(predictions, "../predictions/predictions.json")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
