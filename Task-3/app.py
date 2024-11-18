from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
import os
import torch
from torchtext.data.utils import get_tokenizer
from PIL import Image
import uvicorn
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import random
import tempfile

app = FastAPI()

tokenizer = get_tokenizer('basic_english')
PAD_TOKEN = '<pad>'
PAD_ID = 0  # Assuming 0 is used for padding

def load_data(type: str):
    if type == "text":
        file_path = os.path.join('sample', 'sample.txt')
        with open(file_path, 'r') as file:
            return file.read().splitlines()
    elif type == "image":
        image_path = os.path.join('sample', 'cat.jpg')
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        return [image_path]
    else:
        raise HTTPException(status_code=400, detail="Invalid data type")

def preprocess_text(lines):
    processed_data = []
    for line in lines:
        tokens = tokenizer(line.strip())
        token_ids = [ord(token[0]) for token in tokens]
        processed_data.append(token_ids)
    return processed_data

def preprocess_image(image_paths, target_size=(224, 224)):
    processed_data = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            img = img.resize(target_size)
            processed_data.append(img)
    return processed_data

def augment_text(processed_data, original_data, max_length):
    # Convert to tensors
    tensors = [torch.tensor(seq) for seq in processed_data]
    
    # Pad sequences
    padded_sequences = pad_sequence(tensors, batch_first=True, padding_value=PAD_ID)
    
    # Truncate or pad to max_length
    if padded_sequences.size(1) > max_length:
        padded_sequences = padded_sequences[:, :max_length]
    elif padded_sequences.size(1) < max_length:
        padding = torch.full((padded_sequences.size(0), max_length - padded_sequences.size(1)), PAD_ID)
        padded_sequences = torch.cat([padded_sequences, padding], dim=1)
    
    # Create data with words and padded token ids
    data_with_padding = []
    for i, (orig, padded) in enumerate(zip(original_data, padded_sequences)):
        tokens = tokenizer(orig.strip())
        padded_tokens = tokens[:max_length] + [PAD_TOKEN] * (max_length - len(tokens))
        data_with_padding.append([padded_tokens, padded.tolist()])
    
    return padded_sequences.tolist(), data_with_padding

def augment_image(image):
    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    ]
    
    # Choose one random augmentation
    aug_transform = random.choice(augmentation_transforms)
    
    # Apply the chosen augmentation
    augmented_image = aug_transform(image)
    
    return augmented_image

@app.get("/load/{type}")
async def get_data(type: str):
    try:
        data = load_data(type)
        return {"type": type, "data": data}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preprocess/{type}")
async def preprocess_data(type: str):
    try:
        data = load_data(type)
        if type == "text":
            processed_data = preprocess_text(data)
            return {"type": type, "processed_data": processed_data}
        elif type == "image":
            target_size = (224, 224)
            processed_data = preprocess_image(data, target_size)
            return FileResponse(data[0])  # Return the original resized image
        else:
            raise HTTPException(status_code=400, detail="Invalid data type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/augment/{type}")
async def augment_data(type: str, max_length: int = Query(default=10, ge=1)):
    try:
        data = load_data(type)
        if type == "text":
            processed_data = preprocess_text(data)
            padded_data, data_with_padding = augment_text(processed_data, data, max_length)
            return {
                "preprocessed_data": padded_data,
                "data": data_with_padding
            }
        elif type == "image":
            processed_data = preprocess_image(data)
            augmented_image = augment_image(processed_data[0])
            
            # Save the augmented image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                augmented_image.save(temp_file.name)
                return FileResponse(temp_file.name)
        else:
            raise HTTPException(status_code=400, detail="Invalid data type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/display_image")
async def display_image():
    image_path = os.path.join('sample', 'cat.jpg')
    return FileResponse(image_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
