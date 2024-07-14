#FER CNN model for unsupervised learning (implements encoder and decoder - autoencoder)
#https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/#

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing (resize, data augmentation normalize images)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
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
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # Output: 64 x 24 x 24
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 24 x 24
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # Output: 128 x 12 x 12
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 256 x 12 x 12
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # Output: 256 x 6 x 6
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Output: 512 x 6 x 6
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)   # Output: 512 x 3 x 3
        )
        self.fc1 = nn.Linear(512 * 3 * 3, 2048)  # Fully connected layer
        self.fc2 = nn.Linear(2048, 512 * 3 * 3)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Output: 256 x 6 x 6
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Output: 128 x 12 x 12
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Output: 64 x 24 x 24
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),  # Output: 3 x 48 x 48
            nn.Sigmoid()  # Use Sigmoid to match the normalized input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 512, 3, 3)  # Reshape for the decoder
        x = self.decoder(x)
        return x

model = FERCNN().to(device)

# Hyperparameters
learning_rate = 0.0001
epochs = 15

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

# PCA dimensionality reduction
pca = PCA(n_components=50)
train_features_reduced = pca.fit_transform(train_features)

# K-means clustering on the training
kmeans_train = KMeans(n_clusters=8, random_state=0).fit(train_features_reduced)
train_cluster_labels = kmeans_train.labels_

# Extract from test dataset
test_features = extract_features(test_loader).cpu().numpy()

# Dimensionality reduction
test_features_reduced = pca.transform(test_features)

# Perform K-means clustering on the test features
kmeans_test = KMeans(n_clusters=8, random_state=0).fit(test_features_reduced)
test_cluster_labels = kmeans_test.labels_

# Silhouette score to measure clusters and evaluate
train_silhouette_score = silhouette_score(train_features_reduced, train_cluster_labels)
test_silhouette_score = silhouette_score(test_features_reduced, test_cluster_labels)

print(f'Training Silhouette Score: {train_silhouette_score:.4f}')
print(f'Test Silhouette Score: {test_silhouette_score:.4f}')


#plot clusters for train/test and cluster labels
def plot_clusters(cluster_labels, dataset, num_clusters=8, num_samples_per_cluster=5):
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
