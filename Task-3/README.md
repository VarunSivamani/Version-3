# FastAPI Data Processing and Augmentation Service

This FastAPI application provides endpoints for loading, preprocessing, and augmenting text and image data. It is designed to demonstrate basic data processing and augmentation techniques using Python libraries such as TorchText and TorchVision.

## Features

- **Load Data**: Load text or image data from predefined sample files.
- **Preprocess Data**: Tokenize and process text data or resize image data.
- **Augment Data**: Apply padding to text data or perform random augmentations on image data.
- **Display Image**: Serve a sample image file.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Torch
- TorchText
- TorchVision
- Pillow

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/VarunSivamani/Version-3.git
   cd Version-3/Task-3
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a `sample` directory with a `sample.txt` file and a `cat.jpg` image for testing.

## Usage

1. Run the application:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Access the API documentation at `http://localhost:8000/docs` to explore the available endpoints.

## API Endpoints

- **GET /load/{type}**: Load text or image data. `{type}` can be `text` or `image`.
- **GET /preprocess/{type}**: Preprocess text or image data. `{type}` can be `text` or `image`.
- **GET /augment/{type}**: Augment text or image data. `{type}` can be `text` or `image`.
- **GET /display_image**: Display the sample image.

## Example Requests

- Load text data:

  ```bash
  curl http://localhost:8000/load/text
  ```

- Preprocess image data:

  ```bash
  curl http://localhost:8000/preprocess/image
  ```

- Augment text data with a maximum length of 15:
  ```bash
  curl "http://localhost:8000/augment/text?max_length=15"
  ```
