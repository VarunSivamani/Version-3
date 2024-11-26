from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from setup import device, test_loader, test_dataset
from model import Mnist_Model


def accuracy_per_class(model, device, test_loader, test_data):  
    model = model.to(device)
    model.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    final_output = {}
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Calculating class accuracies" ):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    classes = test_data.classes
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        final_output[classes[i].split("-")[1]] = accuracy
        
    original_class = list(final_output.keys())
    class_accuracy = list(final_output.values())
    plt.figure(figsize=(8, 6))
    plt.bar(original_class, class_accuracy)
    plt.xlabel('classes')
    plt.ylabel('accuracy')
    
    # Save the plot
    plt.savefig('accuracy_per_class.png')
    plt.close()
    
    return final_output


if __name__ == '__main__':
    model = Mnist_Model().to(device)
    
    accuracy_per_class(model, device, test_loader, test_dataset)