from Utils import Utils
import sys


def main():
    dataroot = "C:/Users/anast/Documents/_Spring_2021/Senior_Project/dataset"
    #--------------prepare data------------------------
    utils = Utils()
    dataloader = utils.prepare_data(dataroot, image_size, batch_size, workers)
    utils.plot_training_images(dataloader, device)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
