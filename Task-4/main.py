import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import requests
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import time
from tqdm import tqdm

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage:")
    print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
    print(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

# Data loading and preprocessing
print("Loading and preprocessing data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)
print(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# Initialize model, loss, and optimizer
print("Initializing model...")
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epochs):
    # Clear previous training data
    try:
        requests.post('http://localhost:5000/clear_history')
    except:
        print("Couldn't connect to visualization server to clear history")
    
    train_losses = []
    train_accuracies = []
    last_update_time = 0
    update_interval = 0.5  # 500ms
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate running loss and accuracy
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            
            _, predicted = torch.max(output.data, 1)
            running_total += target.size(0)
            running_correct += (predicted == target).sum().item()
            accuracy = 100 * running_correct / running_total
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
            
            # Update visualization
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                update_data = {
                    'loss': float(avg_loss),  # Ensure it's a native Python float
                    'accuracy': float(accuracy),  # Ensure it's a native Python float
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1
                }
                
                try:
                    response = requests.post('http://localhost:5000/update', 
                                          json=update_data,
                                          timeout=1)
                    if response.status_code != 200:
                        print(f"Failed to update visualization: {response.text}")
                except Exception as e:
                    print(f"Error updating visualization: {str(e)}")
                
                last_update_time = current_time
        
        print(f'Epoch {epoch+1}/{epochs} completed. Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print("Training completed!")

def evaluate_random_samples():
    print("Evaluating random samples...")
    model.eval()
    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    
    # Select 10 random indices
    random_indices = np.random.choice(len(images), 10, replace=False)
    
    plt.figure(figsize=(20, 4))
    for idx, i in enumerate(random_indices):
        image = images[i].to(device)
        true_label = labels[i].item()
        
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            pred_label = output.argmax(dim=1).item()
        
        plt.subplot(2, 5, idx + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')
    
    plt.savefig('static/results.png')
    plt.close()
    print("Evaluation completed. Results saved to 'static/results.png'")

if __name__ == "__main__":
    train(epochs=4)
    time.sleep(1)  # Give time for final update
    evaluate_random_samples()
