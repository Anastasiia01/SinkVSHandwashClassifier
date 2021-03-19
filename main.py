import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
#from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from Utils import Utils
from models.CNN import CNNClassifier

def main():
    dataroot = "C:/Users/anast/Documents/_Spring_2021/Senior_Project/dataset/"
    #--------------prepare data------------------------
    accuracy_stats = {
    'train': [],
    "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    utils = Utils()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workers = 1
    batch_size = 16
    image_size = 224
    train_loader, test_loader = utils.prepare_data(dataroot, image_size, batch_size)
    #utils.plot_images(train_loader, device, 'training_images')
    #utils.plot_images(test_loader, device, 'testing_images')
    model = CNNClassifier()
    model.to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.008)

    print("Begin training.")
    for e in tqdm(range(1, 21)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:            
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch).squeeze() # returns a tensor with all the dimensions of input of size 1 removed.
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = binary_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            print("one batch down...")
        # VALIDATION
        """with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch).squeeze()
                y_val_pred = torch.unsqueeze(y_val_pred, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = binary_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()"""
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        #loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        #accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}')
    # TESTING
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        model.eval()
        test_epoch_loss = 0
        test_epoch_acc = 0

        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)

            test_acc = binary_acc(y_test_pred, y_batch)
            test_loss = criterion(y_test_pred, y_batch)
            test_epoch_loss += test_loss.item()
            test_epoch_acc += test_acc.item()

            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())

    print(f'Test Loss: {test_epoch_loss/len(test_loader):.5f} | Test Acc: {test_epoch_acc/len(test_loader):.3f}')

    y_pred_list = [i[0][0][0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]

    print(classification_report(y_true_list, y_pred_list))
    print(confusion_matrix(y_true_list, y_pred_list))



def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

if __name__ == "__main__":
    sys.exit(int(main() or 0))
