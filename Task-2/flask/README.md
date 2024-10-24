# Animal Viewer and File Analyzer

This is a simple Flask web application that allows users to view animal images and analyze uploaded files. 

### Features:
1. Animal Viewer: Users can select an animal (cat, dog, or elephant) and view its image.
2. File Analyzer: Users can upload a file and receive information about its name, size, type, and potential uses.

## Project Structure

```
project/
├── flask/
│ ├── static/
│ │ ├── script.js
│ │ └── styles.css
│ ├── templates/
│ │ └── index.html
│ ├── app.py
│ └── README.md 
├── images/
│ │ ├── dog.jpg
│ │ ├── cat.jpg
│ │ └── elephant.jpg
└── README.md
```

## Prerequisites

- Python 3.7 or higher
- Flask

## Setup

1. Clone this repository or download the source code.

2. Navigate to the project directory:
   ```
   cd path/to/project
   ```

3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

5. Install the required packages:
   ```
   pip install flask
   ```

## Running the Application

1. Navigate to the `flask` directory:
   ```
   cd flask
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and go to `http://127.0.0.1:5000/` to view the application.

## Usage

### Animal Viewer

1. Select one of the radio buttons (Cat, Dog, or Elephant).
2. The corresponding animal image will be displayed automatically.

### File Analyzer

1. Click on the "Choose a file" button to select a file from your computer.
2. Click the "Analyze" button to upload and analyze the file.
3. The file information (name, size, type, and potential uses) will be displayed below.

## Customization

- To add more animals, update the HTML in `templates/index.html` and add corresponding images to the `images` folder.
- To modify the styling, edit the `static/styles.css` file.
- To change the file analysis logic or add more features, modify the `static/script.js` and `app.py` files.

## Disclaimer

This extension is for educational purposes only. 