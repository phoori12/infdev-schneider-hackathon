import pandas as pd
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import logging
import sys

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=sys.stdout) 


# calculate accuracy
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc*100)

    return acc

# create custom dataset sheet
class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)



# define our neueral network model
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

# load data from preprocessed dataframe
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# split and scale data
def split_data(df,file_path):
    train = df.iloc[:, 1:-1]
    target = df.iloc[:,-1]
    # debugging
    #print(df.shape)
    #print(train)
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, shuffle=False)
    # save validation Dataset
    val_dataset = pd.concat([X_val, y_val], axis=1)
    val_dataset.to_csv(file_path, index=False)

    # convert dataset to numpyarray
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    
    return X_train, X_val, y_train, y_val


def train_model(X_train,y_train,X_val,y_val):
    # hyperparameters
    BATCH_SIZE = 32 
    input_size =  8 # Number of features (excluding timestamp)
    num_classes = 8  # Number of countries
    num_epochs = 70
    LEARNING_RATE = 0.001
    # store accuracy_stats and loss_stats
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }         

    # store custom dataset(our train and validation datasets)
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    # load Data into Dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # define model instance             
    model = MulticlassClassification(input_size ,num_classes)
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # define device  to train our model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # start training 
    logging.info("Begin training.")
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

            test_epoch_loss = 0
            test_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in test_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                test_loss = criterion(y_val_pred, y_val_batch)
                test_acc = multi_acc(y_val_pred, y_val_batch)

                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()

        # check performance of our model
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(test_epoch_loss/len(test_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(test_epoch_acc/len(test_loader))


        logging.info(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {test_epoch_loss/len(test_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {test_epoch_acc/len(test_loader):.3f}')
    
    # Assuming loss_stats and accuracy_stats are available after training
    # plot_learning_curve(loss_stats['train'], loss_stats['val'], accuracy_stats['train'], accuracy_stats['val'])
    logging.info("Training finished. Calculating final accuracy...")

    # Evaluate the final accuracy on the training set
    final_train_acc = sum(accuracy_stats['train']) / len(accuracy_stats['train'])
    logging.info(f"Final Training Accuracy: {final_train_acc:.3f}")

    # Evaluate the final accuracy on the validation set
    final_test_acc = sum(accuracy_stats['val']) / len(accuracy_stats['val'])
    logging.info(f"Final Validation Accuracy: {final_test_acc:.3f}")


    return model

def save_model(model, model_path):

    # load weight of the model
    best_model_state = model.state_dict()
    #print our model's weights
    print("Model's weights: \n",best_model_state)
   
    # save model
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
        default='models/model.pt', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df,"data/test.csv") # split the 20% end of the data for prediction
    model = train_model(X_train,y_train,X_val,y_val)
    save_model(model,model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)