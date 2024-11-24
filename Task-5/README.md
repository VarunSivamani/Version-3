# MNIST Model Project

This project is designed to train a machine learning model on the MNIST dataset. It includes automated workflows to ensure the model trains correctly and efficiently using GitHub Actions.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, clone the repository and install the required dependencies.

```python
git clone https://github.com/VarunSivamani/Version-3
cd Task-5
pip install -r requirements.txt
```

## Project Structure

```python
Task-5
├── model.py           # Model creation
├── pre-test.py        # Tests before training
├── post-test.py       # Tests after training
├── setup.py           # Dataset creation and dataloaders
├── train.py           # Model training and testing code
├── utils.py           # Utility file
├── T5-Notebook.ipyb   # Sample notebook for reference
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
5. **Pre-training Model Check**: Runs initial tests before training the model.
6. **Training Model**: Trains the model using the dataset.
7. **Post-training Model Check**: Validates the model after training.
8. **Utils File Run**: Executes additional utility scripts.
9. **Check Workflow Status**: Outputs the completion status of the workflow.

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

3. Run the pre-tests
```python
python pre-test.py
```

4. Train the model
```python
python train.py
```

5. Run the post-tests
```python
python post-test.py
```

6. Run utility functions
```python
python utils.py
```

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature.
3. Add your changes and commit them.
4. Push to your branch.
5. Create a pull request against the `main` or `master` branch.

## License

This project is released under the MIT License. See the LICENSE file for more details.