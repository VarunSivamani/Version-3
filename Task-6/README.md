# MNIST Model Project

This project is designed to train a machine learning model on the MNIST dataset. It includes automated workflows to ensure the model trains correctly and efficiently using GitHub Actions.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Usage](#usage)
- [Model Summary](#model-summary)
- [Model Tests](#model-tests)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, clone the repository and install the required dependencies.

```python
git clone https://github.com/VarunSivamani/Version-3
cd Task-6
pip install -r requirements.txt
```

## Project Structure

```python
Task-6
├── model.py           # Model creation
├── tests.py           # Tests 
├── setup.py           # Dataset creation and dataloaders
├── train.py           # Model training and testing code
├── utils.py           # Utility file
├── T6-Notebook.ipyb   # Sample notebook for reference
├── requirements.txt   # Packages and modules needed
└── .github/workflows  # CI/CD configuration
    └── mnist.yml      # GitHub actions for MNIST dataset and model trained
```

## Workflow

The `.github/workflows/mnist.yml` file defines a GitHub Actions workflow that automates testing and training the model on push or pull request events to the `main` or `master` branches, excluding markdown and documentation changes.

### Steps Included in the Workflow:

1. **Set up Python**: Configures the Python environment.
2. **Install Dependencies**: Installs required Python packages.
3. **Setup Dataset and Dataloaders**: Prepares the MNIST dataset for training.
4. **Model Summary**: Outputs a summary of the model.
5. **Model tests**: Runs tests for the model.
6. **Training Model**: Trains the model using the dataset.
7. **Utils File Run**: Executes additional utility scripts.
8. **Check Workflow Status**: Outputs the completion status of the workflow.

## Usage

To run the project manually, execute the following scripts:

1. Setup dataset and dataloader
```python
python setup.py
```

2. Summarize the model
```python
python model.py
```

3. Run the tests
```python
python tests.py
```

4. Train the model
```python
python train.py
```

5. Run utility functions
```python
python utils.py
```

## Model Summary

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
            Conv2d-3           [-1, 16, 24, 24]           1,152
              ReLU-4           [-1, 16, 24, 24]               0
       BatchNorm2d-5           [-1, 16, 24, 24]              32
           Dropout-6           [-1, 16, 24, 24]               0
            Conv2d-7            [-1, 8, 24, 24]             128
         MaxPool2d-8            [-1, 8, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           1,152
             ReLU-10           [-1, 16, 10, 10]               0
      BatchNorm2d-11           [-1, 16, 10, 10]              32
          Dropout-12           [-1, 16, 10, 10]               0
           Conv2d-13              [-1, 8, 8, 8]           1,152
             ReLU-14              [-1, 8, 8, 8]               0
      BatchNorm2d-15              [-1, 8, 8, 8]              16
          Dropout-16              [-1, 8, 8, 8]               0
           Conv2d-17             [-1, 16, 6, 6]           1,152
             ReLU-18             [-1, 16, 6, 6]               0
      BatchNorm2d-19             [-1, 16, 6, 6]              32
          Dropout-20             [-1, 16, 6, 6]               0
           Conv2d-21             [-1, 16, 4, 4]           2,304
             ReLU-22             [-1, 16, 4, 4]               0
...
Forward/backward pass size (MB): 0.50
Params size (MB): 0.03
Estimated Total Size (MB): 0.53
----------------------------------------------------------------
```

## Model Tests

1. Model Parameters less than 25000
2. Use of Batch Normalization layers
3. Use of Dropout layers
4. Use of Linear and GAP layers

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature.
3. Add your changes and commit them.
4. Push to your branch.
5. Create a pull request against the `main` or `master` branch.

## License

This project is released under the MIT License. See the LICENSE file for more details.