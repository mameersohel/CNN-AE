import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

#data preprocessing (resize and normalize images)
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Load datasets
affectnet_dataset = ImageFolder(root='affectnet path', transform=transform)
ferplus_dataset = ImageFolder(root='ferplus path', transform=transform)

#DataLoaders
# Load datasets
train_dataset = ImageFolder(root='path/to/fer/train', transform=transform)
test_dataset = ImageFolder(root='path/to/fer/test', transform=transform)
val_dataset = ImageFolder(root='path/to/fer/validation', transform=transform)

# Create DataLoaders
FER_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
FER_test = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#FER CNN model for unsupervised learning (implements encoder and decoder - autoencoder)
#https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/#

class FERCNN(nn.Module):
    def __init__(self):
        super(FERCNN, self).__init__()
        self.conv1 = nn.Conv2d(96, 96, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(96, 96, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 8, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten
        x = self.fc3(x)

        # Decoder
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.t_conv3(x))  # Sigmoid activation for the output
        return x

# Instantiate the model
model = FERCNN()

#hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 100

#loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training
for epoch in range(epochs):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for images, data in range(affectnet_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        # print running stats
        running_loss += loss.item()
        if images % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {images + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

#feature extraction from encoder layers
class Encoder(nn.Module):
    def __init__(self, autoencoder):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            autoencoder.conv1,
            nn.ReLU(),
            autoencoder.pool,
            autoencoder.conv2,
            nn.ReLU(),
            autoencoder.pool,
            autoencoder.conv3,
            nn.ReLU(),
            autoencoder.pool
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

# Instantiate the encoder
encoder = Encoder(model)
encoder.eval()

def extract_features(dataloader):
    features = []
    for images, data in dataloader:
        with torch.no_grad():
            output = encoder(images)
        features.append(output)
    return torch.cat(features, dim=0)

# Extract features for both datasets
affectnet_features = extract_features(affectnet_loader)
ferplus_features = extract_features(ferplus_loader)

from sklearn.cluster import KMeans

# Combine features from both datasets
features = torch.cat((affectnet_features, ferplus_features), dim=0).view(len(affectnet_features) + len(ferplus_features), -1).numpy()

# Perform K-means clustering
kmeans = KMeans(n_clusters=8, random_state=0).fit(features)
labels = kmeans.labels_

# Actual labels from datasets to test accuracy?
affectnet_labels = [label for _, label in affectnet_dataset]
ferplus_labels = [label for _, label in ferplus_dataset]
actual_labels = affectnet_labels + ferplus_labels

#calculate accuracy
predicted_labels = labels  # need to replace with prooper clustering labels
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f'Accuracy: {accuracy}')

#Calinski-Harabasz Index

