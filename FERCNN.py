import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

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
    return Subset(dataset, indices)

# Create random subsets for experimentation
subset_size = 1000  # Adjust the subset size as needed
train_subset = create_random_subset(train_dataset, subset_size)
test_subset = create_random_subset(test_dataset, subset_size // 4)  # Smaller test subset

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

# Feature extraction from encoder layers
class Encoder(nn.Module):
    def __init__(self, autoencoder):
        super(Encoder, self).__init__()
        self.encoder = autoencoder.encoder

    def forward(self, x):
        x = self.encoder(x)
        return x

encoder = Encoder(model).to(device)
encoder.eval()

def extract_features(dataloader):
    features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            output = encoder(images)
            features.append(output.view(images.size(0), -1).cpu())  # Flatten features for clustering
    return torch.cat(features, dim=0)

# Extract features for the dataset
ferplus_features = extract_features(train_loader).cpu().numpy()

# Perform K-means clustering
kmeans = KMeans(n_clusters=8, random_state=0).fit(ferplus_features)
cluster_labels = kmeans.labels_

#print("K-means clustering labels:", cluster_labels)


# Visualize feature maps of a single image
def visualize_features(image, model):
    model.eval()
    with torch.no_grad():
        feature_maps = model.encoder(image.unsqueeze(0).to(device))

    num_feature_maps = feature_maps.shape[1]
    grid_size = int(np.ceil(np.sqrt(num_feature_maps)))

    # Plot the feature maps
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    for i in range(num_feature_maps):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(feature_maps[0, i].cpu(), cmap='gray')
        plt.axis('off')
    plt.show()


# Example usage
for images in train_loader:
    sample_image = images[0]
    break

visualize_features(sample_image, model)

# Example of how you might implement plotting clusters
def plot_clusters(cluster_labels, dataset, num_clusters=8, num_samples_per_cluster=5):
    # Create a directory to save the cluster images
    os.makedirs("cluster_images", exist_ok=True)

    # Collect a few images from each cluster
    images_per_cluster = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(cluster_labels):
        if len(images_per_cluster[label]) < num_samples_per_cluster:
            image = dataset[idx]  # No need to unpack
            images_per_cluster[label].append(image)

    # Plotting
    fig, axs = plt.subplots(num_clusters, num_samples_per_cluster, figsize=(12, 12))

    for cluster_idx in range(num_clusters):
        for sample_idx in range(num_samples_per_cluster):
            if sample_idx < len(images_per_cluster[cluster_idx]):
                image = images_per_cluster[cluster_idx][sample_idx]
                axs[cluster_idx, sample_idx].imshow(make_grid(image, normalize=True).permute(1, 2, 0))
            axs[cluster_idx, sample_idx].axis('off')
            axs[cluster_idx, sample_idx].set_title(f'Cluster {cluster_idx}')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_clusters(cluster_labels, train_subset)