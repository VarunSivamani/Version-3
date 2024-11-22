import torch
import torch.optim as optim
from model import Mnist_Model
from setup import device

# Test for model parameters
def test_model_parameters(model):
    assert sum(p.numel() for p in model.parameters()) < 25000, '❌ FAILED: Model has more than 25000 parameters'
    print('✅ PASSED: Model has less than 25000 parameters')


def test_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 0.01, '❌ FAILED: Learning rate is not set correctly'
    print('✅ PASSED: Learning rate is set correctly')
    
    
if __name__ == '__main__':
    model = Mnist_Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    test_model_parameters(model)
    test_learning_rate(optimizer)