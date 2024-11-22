import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Mnist_Model
from setup import train_loader, device


def test_gradient_flow(model):
    model.train()
    for param in model.parameters():
        param.grad = None  # Reset gradients
    inputs = torch.randn(1, 1, 28, 28).to(device)  # Dummy input
    outputs = model(inputs)
    loss = outputs.sum()  # Dummy loss
    loss.backward()
    
    assert any(param.grad is not None for param in model.parameters()), '❌ FAILED: Gradients are not flowing'
    print('✅ PASSED: Gradients are flowing')
    
    
def test_output_shape(model):
    model.eval()
    inputs = torch.randn(1, 1, 28, 28).to(device)  # Dummy input
    outputs = model(inputs)
    assert outputs.shape == (1, 10), '❌ FAILED: Output shape is incorrect'
    print('✅ PASSED: Output shape is correct (1, 10)')
    
    
def test_loss_monitoring(model, optimizer, train_loader):
    model.train()
    losses = []  # List to store the loss values for each epoch
    num_epochs = 2  # Number of epochs to monitor

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize the loss for the epoch
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss =  F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Accumulate the loss

        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)

    assert losses[0] > losses[num_epochs - 1], '❌ FAILED: Loss did not decrease as expected'
    print('✅ PASSED: Loss monitoring test passed.')


if __name__ == '__main__':
    model = Mnist_Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    test_gradient_flow(model)
    test_output_shape(model)
    test_loss_monitoring(model, optimizer, train_loader)