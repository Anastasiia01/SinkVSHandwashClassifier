from Utils import Utils
import sys
import torch


def main():
    dataroot = "C:/Users/anast/Documents/_Spring_2021/Senior_Project/dataset"
    #--------------prepare data------------------------
    utils = Utils()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workers = 1
    batch_size = 16
    image_size = 128
    trainloader, testloader = utils.prepare_data(dataroot, image_size, batch_size)
    utils.plot_images(trainloader, device, 'training_images')
    utils.plot_images(testloader, device, 'testing_images')


if __name__ == "__main__":
    sys.exit(int(main() or 0))
