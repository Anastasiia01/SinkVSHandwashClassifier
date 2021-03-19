import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

class Utils(object):
    def prepare_data(self, dataroot, image_size, batch_size):
        # Create the dataset
        print("Preparing dataset...")
        train_dataset = dset.ImageFolder(root=dataroot+'Train',
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        print(train_dataset.classes)
        print(train_dataset)
        print(train_dataset.class_to_idx)
        # Create the dataloader
        test_dataset = dset.ImageFolder(root=dataroot+'Test',
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        print(test_dataset.classes)
        

        print("Preparing dataloaders...")
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size//2,
                                                 shuffle=True)
        return trainloader, testloader

    def plot_images(self, dataloader, device, plot_title):
        # Plot some training images
        print(f"Plotting some {plot_title}")
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(plot_title)
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
