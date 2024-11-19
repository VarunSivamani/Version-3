# CNN MNIST Training Visualization

This project implements a 4-layer CNN trained on MNIST with real-time training visualization.

## Requirements 
```
pip install torch torchvision flask numpy matplotlib
```

## Project Structure

├── main.py # Main training script
├── model.py # CNN model definition
├── visualizer.py # Flask server for visualization
├── templates/
│ └── index.html # Training visualization page
└── static/
└── style.css # Basic styling

## How to Run

1. Start the visualization server:

```
python visualizer.py
```

2. In a new terminal, start the training:

```
python main.py
```
3. Open your browser and go to:

http://localhost:5000


You will see the training progress and loss curves updating in real-time.
After training completes, the results on 10 random test images will be displayed.