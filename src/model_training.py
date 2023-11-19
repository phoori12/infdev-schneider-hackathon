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

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)




class MulticlassClassification(nn.Module):
            def __init__(self, num_feature, num_class):

                super(MulticlassClassification, self).__init__()

                self.layer_1 = nn.Linear(num_feature, 512)
                self.layer_2 = nn.Linear(512, 128)
                self.layer_3 = nn.Linear(128, 64)
                self.layer_out = nn.Linear(64, num_class)

                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(p=0.2)
                self.batchnorm1 = nn.BatchNorm1d(512)
                self.batchnorm2 = nn.BatchNorm1d(128)
                self.batchnorm3 = nn.BatchNorm1d(64)

            def forward(self, x):
                x = self.layer_1(x)
                x = self.batchnorm1(x)
                x = self.relu(x)

                x = self.layer_2(x)
                x = self.batchnorm2(x)
                x = self.relu(x)
                x = self.dropout(x)

                x = self.layer_3(x)
                x = self.batchnorm3(x)
                x = self.relu(x)
                x = self.dropout(x)


                x = self.layer_out(x)

                return x

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    scaler = MinMaxScaler()
    train = df.iloc[:, 1:-1]
    train_scaled = scaler.fit_transform(train)
    target = df.iloc[:, -1]
    # print(target)
    # print(train)
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    return X_train, X_val, y_train, y_val


def train_model(X_train,y_train,X_val,y_val):

    BATCH_SIZE = 32
    input_size =  9  # Number of features (excluding timestamp)
    num_classes = 9  # Number of countries
    num_epochs = 150

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }         

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
                
    model = MulticlassClassification(input_size ,num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Begin training.")
    for e in (range(1, num_epochs+1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()


        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in test_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(test_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(test_loader))


        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(test_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(test_loader):.3f}')
    
    print("Training finished. Calculating final accuracy...")

    # Evaluate the final accuracy on the training set
    final_train_acc = sum(accuracy_stats['train']) / len(accuracy_stats['train'])
    print(f"Final Training Accuracy: {final_train_acc:.3f}")

    # Evaluate the final accuracy on the validation set
    final_val_acc = sum(accuracy_stats['val']) / len(accuracy_stats['val'])
    print(f"Final Validation Accuracy: {final_val_acc:.3f}")
    model.eval()
    return model

def save_model(model, model_path):
    best_model_state = model.state_dict()
    # model_scripted = torch.jit.script(best_model_state)
    # torch.jit.save(best_model_state, model_path)
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
    df = load_data("../data/test_final.csv")
    X_train, X_val, y_train, y_val = split_data(df) 
    model = train_model(X_train,y_train,X_val,y_val)
    #save_model(model, '../models/model.pkl')

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)