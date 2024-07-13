#FER CNN model for unsupervised learning (implements encoder and decoder - autoencoder)
#https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/#

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing (resize and normalize images)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Load all images from the directory
full_dataset = ImageFolder(root='/Users/ameersohel/Downloads/FER-CNN-AE-main', transform=transform)

# Split the dataset into training and testing sets (70% train, 30% test)
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoaders for training and testing subsets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#FER CNN-Autoencoder
class FERCNN(nn.Module):
    def __init__(self):
        super(FERCNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 48 x 48
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 48 x 48
            nn.Tanh(),
            nn.MaxPool2d(2, 2),  # Output: 64 x 24 x 24
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),  # Output: 128 x 24 x 24
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # Output: 128 x 12 x 12
        )
        self.fc1 = nn.Linear(128 * 12 * 12, 2048)  # Fully connected layer
        self.fc2 = nn.Linear(2048, 128 * 12 * 12)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  # Output: 128 x 24 x 24
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Output: 128 x 48 x 48
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 48 x 48
            nn.Tanh(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 48 x 48
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output: 3 x 48 x 48
            nn.Sigmoid()  # Use Sigmoid to match the normalized input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 12, 12)  # Reshape for the decoder
        x = self.decoder(x)
        return x


model = FERCNN().to(device)

# Hyperparameters
learning_rate = 0.0001
epochs = 10

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop through training loader set
losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
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

# Encoder class to use only encoder part of CNN-AE
class Encoder(nn.Module):
    def __init__(self, autoencoder):
        super(Encoder, self).__init__()
        self.encoder = autoencoder.encoder

    def forward(self, x):
        x = self.encoder(x)
        return x


#define and set to evaluation mode
encoder = Encoder(model).to(device)
encoder.eval()

#extract features from encoder function and flatten
def extract_features(dataloader):
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = encoder(images)
            features.append(output.view(images.size(0), -1).cpu())  # Flatten features for clustering
    return torch.cat(features, dim=0)


# Extract from training dataset
train_features = extract_features(train_loader).cpu().numpy()

# K-means clustering on the training
kmeans_train = KMeans(n_clusters=8, random_state=0).fit(train_features)
train_cluster_labels = kmeans_train.labels_

# Extract from test dataset
test_features = extract_features(test_loader).cpu().numpy()

# Perform K-means clustering on the test features
kmeans_test = KMeans(n_clusters=8, random_state=0).fit(test_features)
test_cluster_labels = kmeans_test.labels_


#plot clusters for train/test and cluster labels
def plot_clusters(cluster_labels, dataset, num_clusters=8, num_samples_per_cluster=4):
    # Create a directory to save the cluster images
    os.makedirs("cluster_images", exist_ok=True)

    # Collect a few images from each cluster
    images_per_cluster = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(cluster_labels):
        if len(images_per_cluster[label]) < num_samples_per_cluster:
            image, _ = dataset[idx]
            images_per_cluster[label].append(image)

    # Plotting
    fig, axs = plt.subplots(num_clusters, num_samples_per_cluster, figsize=(12, 12))

    for cluster_idx in range(num_clusters):
        for sample_idx in range(num_samples_per_cluster):
            image = images_per_cluster[cluster_idx][sample_idx]
            axs[cluster_idx, sample_idx].imshow(make_grid(image, normalize=True).permute(1, 2, 0))
            axs[cluster_idx, sample_idx].axis('off')
            axs[cluster_idx, sample_idx].set_title(f'Cluster {cluster_idx}')

    plt.tight_layout()
    plt.show()

# Plot clusters for the training data
print("Training data clusters:")
plot_clusters(train_cluster_labels, train_dataset)

# Plot clusters for the test data
print("Test data clusters:")
plot_clusters(test_cluster_labels, test_dataset)
