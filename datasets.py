from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader

train_data = mnist.MNIST('mnist', train=True, transform=transforms.ToTensor(), download=True)
test_data = mnist.MNIST('mnist', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

if __name__ == "__main__":
    img, label = train_data[0]
    print(img.dtype, img.shape, label)