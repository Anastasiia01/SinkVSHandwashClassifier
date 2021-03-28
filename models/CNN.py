import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.transforms as transforms
#from tqdm import tqdm
from tqdm.notebook import tqdm
import os
import time
from PIL import Image

class CNNClassifier(nn.Module):

    def __init__(self, device):
        super(CNNClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=56, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.008)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block

    def trainCNN(self, train_loader):
        print("Begin training...")
        self.t_begin = time.time()
        for e in tqdm(range(1, 15)):
            train_epoch_loss = 0
            train_epoch_acc = 0
            self.train()
            for X_train_batch, y_train_batch in train_loader:            
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                self.optimizer.zero_grad()
                y_train_pred = self(X_train_batch).squeeze() # returns a tensor with all the dimensions of input of size 1 removed.
                #print("real: ", y_train_batch)
                #print("prediction: ", y_train_pred )
                train_loss = self.criterion(y_train_pred, y_train_batch)
                train_acc = self.binary_acc(y_train_pred, y_train_batch)
                train_loss.backward()
                self.optimizer.step()
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
            print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}')
        self.t_end = time.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        #self.save_model()

    def evaluate(self, test_loader, best_acc=0): 
        print("Begin testing...")
        with torch.no_grad():
            self.eval()
            test_epoch_loss = 0
            test_epoch_acc = 0

            for x_batch, y_batch in tqdm(test_loader):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_test_pred = self(x_batch)
                _, y_pred_tag = torch.max(y_test_pred, dim = 1)
                y_test_pred = y_test_pred.squeeze()
                #y_test_pred = torch.unsqueeze(y_test_pred, 0)

                test_acc = self.binary_acc(y_test_pred, y_batch)
                test_loss = self.criterion(y_test_pred, y_batch)
                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()
        test_epoch_acc/=len(test_loader)
        print(f'Test Loss: {test_epoch_loss/len(test_loader):.5f} | Test Acc: {test_epoch_acc:.3f}')
        if test_epoch_acc > best_acc:
            print('Saving model..')
            state = {
                'model': self.state_dict(),
                'accuracy': test_epoch_acc,
            }
            print("with accuracy:", state['accuracy'])
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/model.pth')


    def predict(self, filename, image_size):
        image = Image.open(filename, mode = 'r') #reading an image.
        #image = np.array(image) #the 2-d array of integer pixel values    
        #image = image/255.0  #toTensor transform will bring from [0,255] tp [0, 1]
        preproc=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        input_image = preproc(image)
        input_image = input_image.view(1, input_image.size(0), input_image.size(1), input_image.size(2))
        input_image = input_image.to(self.device)
        print(input_image.size())
        with torch.no_grad():
            self.eval()
            y_pred = self(input_image)          
            print("pure prediction: ", y_pred)
            y_pred_tag = torch.log_softmax(y_pred, dim = 1)
            print("after softmax: ", y_pred_tag)
            _, y_pred_tag = torch.max(y_pred_tag, dim = 1)
            print("final output: ", y_pred_tag)
        return y_pred_tag








    
    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
        correct_results_sum = (y_pred_tags == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
        return acc


    """def load_model(self, D_model_filename = './discriminator.pkl', G_model_filename = './generator.pkl'):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))"""


class CNN(nn.Module): 
    def __init__(self, image_size=128, channels=3): 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # we use the maxpool multiple times, but define it once
        self.pool = nn.MaxPool2d(2,2)
        # in_channels = 6 because self.conv1 output 6 channel
        self.conv2 = nn.Conv2d(6,16,5) 
        # 5*5 comes from the dimension of the last convnet layer
        self.fc1 = nn.Linear(16*5*5, 120) #input is 400 as it is flatten after previous layer of 16x5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.main = nn.Sequential(
            # input is (channels) x image_size x image_size; (image_size = 128)
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False), #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.LeakyReLU(0.2, inplace=True), 
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        """model = models.Sequential()
        model.add(Conv2D(16, (15, 15), activation='relu', input_shape=(64, 64,1))) #filters, kernel_size, strides
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (7, 7), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dense(2))  # for sparse_categorial_crossentropy - then choose 2 neurons in next layer
        model.add(layers.Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(lr=0.0005, decay=1e-6)
        model.compile(optimizer= opt , loss= tf.keras.losses.binary_crossentropy, metrics=['accuracy'])"""



    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation on final layer 
        return x


"""
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
  
    def forward(self, input):
        return self.main(input)

"""

"""
def create_model(self):
        # highest accuracy liveness on non-diffused iages- 16 (13,13) => 32 (7,7) => 64 (5,5) => 64 => 1 - acc = 94.2
        model = models.Sequential()
        model.add(Conv2D(16, (15, 15), activation='relu', input_shape=(64, 64,1)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (7, 7), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dense(2))  # for sparse_categorial_crossentropy - then choose 2 neurons in next layer
        model.add(layers.Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(lr=0.0005, decay=1e-6)
        model.compile(optimizer= opt , loss= tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        return model

    def load_model(self,model_file_name):
        model = tf.keras.models.load_model(model_file_name)
        return model

    def train_model(self, model, train_images,train_labels,test_images, test_labels, epochs):
        cbk = CustomModelCheckpoint()  # so that we can save the best model
        history = model.fit(train_images, train_labels, epochs=epochs, callbacks=[cbk], 
                        validation_data=(test_images, test_labels))
        #plt.plot(history.history['accuracy'], label='accuracy')
        #plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        #plt.xlabel('Epoch')
        #plt.ylabel('Accuracy')
        #plt.ylim([0.5, 1])
        #plt.legend(loc='lower right')
        #plt.show()
        return model

    def evaluate(self, model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        return test_loss, test_acc
"""
