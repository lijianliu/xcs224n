import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


# Parameters for the green circle
circle_center = (0.5, -0.5)
circle_radius = 0.5

# Generate training data
np.random.seed(0)
x = np.random.uniform(-0.5, 2.5, 2000)
y = np.random.uniform(-1, 1, 2000)
data = np.vstack((x, y)).T

# Define the curve function
def curve(x):
    return 2*x - 3*x**2 + x**3

# Label the data
labels = np.zeros(len(x), dtype=int)
labels[y > curve(x)] = 1  # Red group
labels[np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2) < circle_radius] = 2  # Green group

# Label the blue group
labels[(labels == 0) & (y <= curve(x))] = 0  # Blue group

# Prepare the data for PyTorch
X = torch.tensor(data, dtype=torch.float32)
Y = torch.tensor(labels, dtype=torch.long)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 3)  # Output layer now has 3 neurons for 3 classes

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the figure for animation
fig, ax = plt.subplots(figsize=(10, 6))

# Create the test data for visualization
test_x = np.linspace(-0.5, 2.5, 100)
test_y = np.linspace(-1, 1, 100)
test_data = np.array([[tx, ty] for tx in test_x for ty in test_y])
test_X = torch.tensor(test_data, dtype=torch.float32)
light_red = '#FFAAAA'
light_green = '#AAFFAA'
light_blue = '#AAAAFF'
light_yellow = '#FFFFAA'
# Function to plot the decision boundary
def plot_decision_boundary(epoch, test_data, predicted):
    ax.clear()
    #cmap = plt.cm.get_cmap('viridis', 3)  # Define a colormap with 3 colors
    cmap = ListedColormap([light_red, light_green, light_blue])

    ax.scatter(test_data[:, 0], test_data[:, 1], c=predicted, cmap=cmap, alpha=0.5, label='Decision Boundary')

    # Plot a subset of the training data
    subset_indices = np.random.choice(len(data), size=int(0.1 * len(data)), replace=False)
    subset_data = data[subset_indices]
    subset_labels = labels[subset_indices]

    for label, color, label_name in zip([0, 1, 2], ['red', 'green', 'blue'], ['Red', 'Green', 'Blue']):
        idx = subset_labels == label
        ax.scatter(subset_data[idx, 0], subset_data[idx, 1], color=color, label=label_name, edgecolor='k', alpha=0.7)

    ax.plot(np.linspace(-0.5, 2.5, 100), curve(np.linspace(-0.5, 2.5, 100)), 'k-', linewidth=2)
    circle = plt.Circle(circle_center, circle_radius, fill=False, edgecolor='k')
    ax.add_artist(circle)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Decision Boundary after Epoch {epoch + 1}')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1, 1)
    ax.legend()

# Animation function
def update(epoch):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    # Test the model
    with torch.no_grad():
        test_outputs = model(test_X)
        _, predicted = torch.max(test_outputs, 1)
        predicted = predicted.numpy()

    # Plot the decision boundary after each epoch
    plot_decision_boundary(epoch, test_data, predicted)

    # Calculate accuracy for each class
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == Y).sum().item()
    accuracy = correct / len(Y)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Number of epochs
num_epochs = 500  # Reduced for demonstration purposes

# Create animation
ani = FuncAnimation(fig, update, frames=num_epochs, interval=300, repeat=False)

# Display the animation
plt.show()
