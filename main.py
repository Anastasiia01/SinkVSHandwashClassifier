import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Utils import Utils
from models.CNN import CNNClassifier

def main(args):
    dataroot = "/content/drive/MyDrive/dataset/"
    #dataroot = "C:/Users/anast/Documents/_Spring_2021/Senior_Project/dataset/"
    
    #--------------prepare data------------------------
    utils = Utils()
    batch_size = 16
    image_size = 224
    trainloader, testloader = utils.prepare_data(dataroot, image_size, batch_size)
    #utils.plot_images(trainloader, device, 'training_images')
    #utils.plot_images(testloader, device, 'testing_images')

    #--------------define classifier------------------------
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNClassifier(device)
    model.to(device)
    #print(model)
    best_acc = 0 

    if args.loadmodel:
        # Load model.
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        best_acc = checkpoint['accuracy']
        print('==> Loading Classifier..')
        model.load_state_dict(checkpoint['model'])

    model.trainCNN(trainloader)
    model.test(testloader, best_acc)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Handwashing Classifier Training')
    parser.add_argument('--loadmodel', '-l', action='store_true',
                        help='load trained model')
    args = parser.parse_args()
    sys.exit(int(main(args) or 0))
