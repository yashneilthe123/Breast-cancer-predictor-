
# Breast Cancer Classification using PyTorch


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load Dataset


breast_cancer = fetch_ucirepo(id=17)

X = breast_cancer.data.features
y = breast_cancer.data.targets

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# Convert labels
# M = Malignant -> 1
# B = Benign -> 0

y = y.replace({'M': 1, 'B': 0})

# If y is a dataframe, convert to series
y = y.squeeze()


# 2. Feature Scaling


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 3. Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Convert to PyTorch Tensors


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)

# 5. Define Neural Network


class BreastCancerNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(30, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Create model
model = BreastCancerNN()

print(model)


# 6. Loss Function and Optimizer

criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# 7. Training Loop


epochs = 100

loss_history = []

for epoch in range(epochs):

    model.train()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# 8. Plot Loss Curve

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# 9. Model Evaluation

model.eval()

with torch.no_grad():

    y_pred_prob = model(X_test)

    y_pred = (
        y_pred_prob >= 0.5
    ).float()

# Convert tensors to numpy

y_true = y_test.numpy()
y_pred_np = y_pred.numpy()

accuracy = accuracy_score(
    y_true,
    y_pred_np
)

print("\nTest Accuracy:")
print(f"{accuracy*100:.2f}%")


# 10. Confusion Matrix


cm = confusion_matrix(
    y_true,
    y_pred_np
)

print("\nConfusion Matrix:")
print(cm)


# 11. Classification Report


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred_np,
        target_names=["Benign", "Malignant"]
    )
)

# 12. Sample Predictions


print("\nSample Predictions:\n")

for i in range(5):

    print(
        f"Actual: {int(y_true[i][0])} | "
        f"Predicted: {int(y_pred_np[i][0])}"
    )