import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing (resize, data augmentation normalize images)
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Load all images from the directory
full_dataset = ImageFolder(root='/Users/ameersohel/Downloads/FER-CNN-AE-main', transform=transform)

# Split the dataset into training and testing sets (70% train, 30% test)
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

#def create_random_subset(dataset, subset_size):
 #   indices = np.random.choice(len(dataset), size=subset_size, replace=False)
  #  return Subset(dataset, indices)

# Create random subsets for experimentation
#subset_size = 20000  # Adjust the subset size as needed
#train_subset = create_random_subset(train_dataset, subset_size)
#test_subset = create_random_subset(test_dataset, subset_size // 4)  # Smaller test subset

# Create DataLoaders for training and testing subsets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# FER CNN-Autoencoder
class FERCNN(nn.Module):
    def __init__(self):
        super(FERCNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 96 x 96
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Output: 64 x 48 x 48

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 48 x 48
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # Output: 128 x 24 x 24

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 256 x 24 x 24
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # Output: 256 x 12 x 12

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Output: 512 x 12 x 12
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),  # Output: 512 x 6 x 6

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # Output: 1024 x 6 x 6
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2)  # Output: 1024 x 3 x 3
        )
        self.fc1 = nn.Linear(1024 * 3 * 3, 2048)  # Fully connected layer
        self.fc2 = nn.Linear(2048, 1024 * 3 * 3)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),  # Output: 512 x 6 x 6
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Output: 256 x 12 x 12
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Output: 128 x 24 x 24
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Output: 64 x 48 x 48
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),  # Output: 3 x 96 x 96
            nn.Sigmoid()  # Use Sigmoid to match the normalized input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 1024, 3, 3)  # Reshape for the decoder
        x = self.decoder(x)
        return x

model = FERCNN().to(device)

# Hyperparameters
learning_rate = 0.0001
epochs = 15
noise_factor = 0.5

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        # Add random noise to input images
        noisy_imgs = images + noise_factor * torch.randn_like(images)
        # Clip the images to be between 0 and 1
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)  # Ensure values are within [0, 1]

        optimizer.zero_grad()
        outputs = model(noisy_imgs.to(device))
        loss = criterion(outputs, images.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# Plot training loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


def visualize_reconstruction(model, dataloader, num_images=10):
    model.eval()
    dataiter = iter(dataloader)  # Get an iterator over the DataLoader
    images, labels = next(dataiter)  # Fetch the first batch of data

    # Add noise to the test images
    noisy_imgs = images + noise_factor * torch.randn_like(images)
    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)

    # Move images and noisy_imgs to device
    images = images.to(device)
    noisy_imgs = noisy_imgs.to(device)

    # Get sample outputs
    outputs = model(noisy_imgs)

    # Convert tensors to numpy arrays for visualization
    images_np = images.cpu().numpy()
    noisy_imgs_np = noisy_imgs.cpu().numpy()
    outputs_np = outputs.cpu().detach().numpy()

    # Plot original and reconstructed images
    fig, axes = plt.subplots(3, num_images, figsize=(25, 4))

    # Plot original images
    for i in range(num_images):
        axes[0, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

    # Plot noisey images
    for i in range(num_images):
        axes[1, i].imshow(np.transpose(noisy_imgs_np[i], (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, i].set_title('Noisey Image')

    # Plot reconstructed images
    for i in range(num_images):
        axes[2, i].imshow(np.transpose(outputs_np[i], (1, 2, 0)))
        axes[2, i].axis('off')
        axes[2, i].set_title('Reconstructed')

    plt.show()

# Visualize original and reconstructed images from the test set
visualize_reconstruction(model, test_loader)
