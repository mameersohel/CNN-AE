import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset, random_split, Dataset
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

subset_size = 1000  # Adjust the subset size as needed
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
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # Output: 32 x 96 x 96
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # Output: 64 x 48 x 48
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # Output: 128 x 48 x 48
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # Output: 256 x 24 x 24
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), # Output: 512 x 24 x 24
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1), # Output: 1024 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 3, stride=1, padding=1), # Output: 2048 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(2048, 4096, 3, stride=2, padding=1), # Output: 4096 x 6 x 6
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, 3, stride=2, padding=1, output_padding=1), # Output: 2048 x 12 x 12
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 3, stride=1, padding=1), # Output: 1024 x 12 x 12
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1), # Output: 512 x 24 x 24
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1), # Output: 256 x 24 x 24
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # Output: 128 x 48 x 48
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1), # Output: 64 x 48 x 48
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # Output: 32 x 96 x 96
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),  # Output: 3 x 96 x 96
            nn.Sigmoid()  # Sigmoid to match the normalized input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = FERCNN().to(device)

# Hyperparameters
learning_rate = 0.001
epochs = 100

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')


# Define ElasticTransform
elastic_transform = ElasticTransform(alpha=34.0, sigma=4.0)


# Denormalization function
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return tensor

# Testing and comparing results
model.eval()
mse_loss = 0.0
with torch.no_grad():
    for clean_images in test_loader:
        clean_images = clean_images.to(device)
        distorted_images = torch.stack([elastic_transform(img.cpu()) for img in clean_images]).to(device)
        outputs = model(distorted_images)

        mse_loss += criterion(outputs, clean_images).item()

        # Denormalize images
        outputs = denormalize(outputs, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # Visualize some examples
        n = min(clean_images.size(0), 4)
        fig, axes = plt.subplots(n, 3, figsize=(15, 15))
        for i in range(n):
            axes[i, 0].imshow(clean_images[i].cpu().permute(1, 2, 0))
            axes[i, 0].set_title("Clean Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(distorted_images[i].cpu().permute(1, 2, 0))
            axes[i, 1].set_title("Distorted Image")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(outputs[i].cpu().permute(1, 2, 0))
            axes[i, 2].set_title("Reconstructed Image")
            axes[i, 2].axis('off')
        plt.show()

print(f'Mean Squared Error on test set: {mse_loss / len(test_loader):.4f}')