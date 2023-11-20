import pandas as pd
import numpy as np
import argparse
import torch
from model_training import MulticlassClassification as mcc_model
from model_training import ClassifierDataset,split_data
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_model(model_path):
    model_state_dict = torch.load(model_path)
    model = mcc_model(num_feature = 8 , num_class = 8)
    model.load_state_dict(model_state_dict)
    return model

def make_predictions(df, model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # df = df.iloc[: , 1:10]
    # df = df.iloc[:-10]
    df = df.iloc[:,:-1]
    model.to(device)
    model.eval()
    
    input_data = torch.tensor(df.values, dtype=torch.float32)  # Adjust the dtype accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)

    # Perform forward pass
    with torch.no_grad():
        output = model(input_data)
    
    # Apply softmax to convert raw scores to probabilities
    probabilities = torch.log_softmax(output, dim = 1)

    # For each sample, get the predicted class
    _, predicted_classes = torch.max(probabilities, 1)

    # Convert the predicted_classes tensor to a Python list
    predicted_classes = predicted_classes.cpu().numpy().tolist()

    
    return predicted_classes




    # df = df.astype(float)
    # print(df)

    # # with torch.no_grad():  # Disable gradient computation during inference
    # #     predictions = model(torch.tensor(df.values, dtype=torch.float32))

    # # probabilities = F.softmax(predictions, dim=1)

    # # # For each sample, get the predicted class
    # # _, predicted_classes = torch.max(probabilities, 1)

    # # # Convert the predicted_classes tensor to a Python list
    # # predicted_classes = predicted_classes.cpu().numpy()

    # # print(predicted_classes)
    # # print(predicted_classes.shape)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # Assuming the special label is in the last column
    # df_features = df.iloc[:, 1:10].astype(float).values
    # df_labels = df.iloc[:, -1].astype(int).values

    # test_dataset = ClassifierDataset(torch.from_numpy(df_features).float(), torch.from_numpy(df_labels).long())
    # test_loader = DataLoader(test_dataset, batch_size=32)
    
    # model.to(device)
    # model.eval()
    # y_pred_list = []

    # for X_batch, _ in test_loader:
    #     X_batch = X_batch.to(device)
    #     y_test_pred = model(X_batch)
    #     _, y_pred_tags = torch.max(y_test_pred, dim=1)
    #     y_pred_list.append(y_pred_tags.cpu().numpy())

    # # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    # # y_pred_list = np.array(y_pred_list).tolist()
    # predictions_df = pd.DataFrame(y_pred_list, columns=[f'Prediction_{i}' for i in range(len(y_pred_list[0]))])

    # print(predictions_df)
    # return y_pred_list
    # print(predictions)
    # print(predictions.shape)

    # max_indices = torch.argmax(predictions, dim=1)
    # print(max_indices[:5])

    # print(max_indices)
    # return predictions


def save_predictions(predictions, predictions_file):
   
     
    json_data = {"target": {}}
    for i, num in enumerate(predictions):
        json_data["target"][str(i)] = num

    with open(predictions_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='../data/val_dataset.csv',
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='../models/model.pt',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='../predictions/predictions.json',
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
