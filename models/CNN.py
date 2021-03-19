import torch
import torch.nn as nn 
import torch.nn.functional as F 

class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=56, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

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
