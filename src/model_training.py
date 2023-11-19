import pandas as pd
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

BATCH_SIZE = 32
input_size =  9  # Number of features (excluding timestamp)
hidden_size = 50
num_classes = 9  # Number of countries
num_epochs = 300
class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            output, _ = self.lstm(x)
            output = self.fc(output[:, -1, :])  # Take the output from the last time step
            return output

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    scaler = MinMaxScaler()
    train = df.iloc[:, 1:-1]
    train_scaled = scaler.fit_transform(train)
    target = df.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
        # df['StartTime'] = pd.to_datetime(df['StartTime'])
        train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        model = LSTMClassifier(input_size ,  hidden_size, num_classes)
        train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Test Accuracy: {accuracy:.2%}')


            
        return model

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data("/home/main/Hackathon/infdev-schneider-hackathon/data/test_final.csv")
    X_train, X_val, y_train, y_val = split_data(df)    
    model = train_model(X_train, y_train)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)