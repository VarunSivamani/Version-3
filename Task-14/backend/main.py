from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import json
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer
from model import LlamaForCausalLM

app = FastAPI()

# First load the tokenizer and set pad token
tokenizer = GPT2Tokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token
model_vocab_size = len(tokenizer)

# Now create model with the correct vocab size from the start
vocab_size = model_vocab_size
hidden_size = 576
num_hidden_layers = 30
num_heads = 9
intermediate_size = 1536
max_seq_len = 2048

# Load the model
model = LlamaForCausalLM(vocab_size, hidden_size, num_hidden_layers, num_heads, intermediate_size, max_seq_len)
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu'))) 
model.eval()

# History storage
history = []

class GenerateRequest(BaseModel):
    prompt: str
    length: int = 25
    

@app.post("/generate")
async def generate(request: GenerateRequest):
    if not (10 <= request.length <= 200):
        raise HTTPException(status_code=400, detail="Length must be between 10 and 200.")

    # encode and generate text
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")  # Convert request.prompt to input_ids
    generated_ids = model.generate(input_ids, max_length=request.length)  # Generate text

    # convert ids to a list
    if isinstance(generated_ids, torch.Tensor):
        generated_ids = generated_ids.tolist()  # Convert tensor to list of lists

    # batch_decode for multiple sequences
    generated_texts = tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)

    # Store in history
    history.append({"prompt": request.prompt, "responses": generated_texts})

    return {"responses": generated_texts}

@app.get("/history")
async def get_history():
    return {"history": history}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Text Generation API"}


# if __name__ == '__main__':
#     # Load the model
#     model = LlamaForCausalLM(vocab_size, hidden_size, num_hidden_layers, num_heads, intermediate_size, max_seq_len)

#     # Print model size
#     model_size = sum(p.numel() for p in model.parameters())
#     print(f"Model Size: {model_size / 1e6:.2f}M parameters.")