import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset, random_split, Dataset
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from torchvision.transforms import ElasticTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
print(f"Using device: {device}")

# Data preprocessing (resize and normalize images)
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Define the custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_labels = [file for file in os.listdir(img_dir) if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Load all images from the directory
data_dir = r'C:\Users\Mehraz\Documents\Actual Documents\KSU CS Grad School\CS7357 (Neural Nets and Deep Learning)\Python\Project\Code\FER-CNN-AE\Data'
full_dataset = CustomImageDataset(img_dir=data_dir, transform=transform)

# Split the dataset into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Function to create a random subset from a dataset
def create_random_subset(dataset, subset_size):
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    return Subset(dataset,indices)

subset_size = 4000  # Adjust the subset size as needed
train_subset = create_random_subset(train_dataset, subset_size)
test_subset = create_random_subset(test_dataset, subset_size // 4)  # Smaller test subset

# # Create DataLoaders for training and testing subsets
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create DataLoaders for training and testing subsets
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

class FERCNN(nn.Module):
    def __init__(self):
        super(FERCNN, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(3, 32),  # Output: 32 x 96 x 96
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 64 x 48 x 48
            nn.ReLU(),
            ResidualBlock(64, 128),  # Output: 128 x 48 x 48
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # Output: 256 x 24 x 24
            nn.ReLU(),
            ResidualBlock(256, 512),  # Output: 512 x 24 x 24
            nn.Conv2d(512, 1024, 3, stride=2, padding=1), # Output: 1024 x 12 x 12
            nn.ReLU(),
            ResidualBlock(1024, 2048),  # Output: 2048 x 12 x 12
            nn.Conv2d(2048, 4096, 3, stride=2, padding=1) # Output: 4096 x 6 x 6
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, 3, stride=2, padding=1, output_padding=1),  # Output: 2048 x 12 x 12
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),  # Output: 1024 x 24 x 24
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # Output: 512 x 48 x 48
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # Output: 256 x 96 x 96
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 3, stride=1, padding=1),  # Output: 64 x 96 x 96
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),  # Output: 3 x 96 x 96
            nn.Sigmoid()  # Sigmoid to match the normalized input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
# Instantiate the model
model = FERCNN().to(device)

# Hyperparameters
learning_rate = 0.0001
epochs = 30
noise_factor = 0

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images.to(device))
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


def visualize_reconstruction(model, dataloader, num_images=10, noise_factor=0):
    model.eval()
    dataiter = iter(dataloader)
    mse_loss = 0.0
    images = next(dataiter)  # Fetch the first batch of data

    # Adjust num_images if it exceeds batch size
    num_images = min(num_images, images.size(0))

    print(f"Original images shape: {images.shape}")

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Add noise to the test images
    noisy_imgs = images + noise_factor * torch.randn_like(images)
    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)

    # Move images and noisy_imgs to device
    images = images.to(device)
    noisy_imgs = noisy_imgs.to(device)

    # Ensure batch dimension is included
    if len(noisy_imgs.shape) == 3:
        noisy_imgs = noisy_imgs.unsqueeze(0)

    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    # Get sample outputs
    outputs = model(noisy_imgs)
    mse_loss += criterion(outputs, images).item()
    # Ensure the images are in the correct range [0, 1] if they are normalized
    outputs = torch.clamp(outputs, 0, 1)

    # Convert tensors to numpy arrays for visualization
    images_np = images.cpu().numpy()
    noisy_imgs_np = noisy_imgs.cpu().numpy()
    outputs_np = outputs.cpu().detach().numpy()

    # Plot original, noisy, and reconstructed images
    fig, axes = plt.subplots(3, num_images, figsize=(25, 4))

    for i in range(num_images):
        # Plot original images
        axes[0, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        # Plot noisy images
        axes[1, i].imshow(np.transpose(noisy_imgs_np[i], (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, i].set_title('Noisy Image')

        # Plot reconstructed images
        axes[2, i].imshow(np.transpose(outputs_np[i], (1, 2, 0)))
        axes[2, i].axis('off')
        axes[2, i].set_title('Reconstructed')

        # Calculate and print SSIM for each image in the batch
        ssim_index = ssim_metric(outputs[i].unsqueeze(0), images[i].unsqueeze(0))
        print(f'SSIM for image {i+1}: {ssim_index.item()}')
        print(f'Mean Squared Error on test set: {mse_loss / len(test_loader):.4f}')
    plt.show()
    


visualize_reconstruction(model, test_loader)
