import gradio as gr
import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForCausalLM,
    SiglipVisionModel,
    BitsAndBytesConfig
)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------- MODEL LOADING -----------------------------

def load_models():
    print("Loading models with 4-bit quantization...")

    # BitsAndBytes config for 4-bit
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load SigLIP (vision model)
    siglip_model = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        quantization_config=quant_config,
        device_map="auto"
    )
    siglip_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # Load Phi-2 in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        quantization_config=quant_config,
        device_map="auto"
    )

    # Load LoRA fine-tuned weights
    phi_model = PeftModel.from_pretrained(base_model, "TharunSivamani/phi3-vlm-peft", device_map="auto")

    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    if phi_tokenizer.pad_token is None:
        phi_tokenizer.pad_token = phi_tokenizer.eos_token

    # Load projection layers
    linear_proj = torch.load('linear_projection.pth', map_location=device)
    image_text_proj = torch.load('image_text_proj.pth', map_location=device)

    return siglip_model, siglip_processor, phi_model, phi_tokenizer, linear_proj, image_text_proj

# -------------------------- LOAD ON STARTUP --------------------------

print("Loading all models...")
siglip_model, siglip_processor, phi_model, phi_tokenizer, linear_proj, image_text_proj = load_models()
print("Models loaded successfully!")

# ---------------------------- DATA PREP ----------------------------

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
first_100_images = [(img, label) for img, label in list(testset)[:100]]

questions = [
    "Give a description of the image?",
    "How does the main object in the image look like?",
    "How can the main object in the image be useful to humans?",
    "What is the color of the main object in the image?",
    "Describe the setting of the image?"
]

# ------------------------- INFERENCE FUNCS -------------------------

def get_image_embedding(image, siglip_model, siglip_processor, linear_proj, device):
    with torch.no_grad():
        inputs = siglip_processor(image, return_tensors="pt").to(device)
        outputs = siglip_model(**inputs)
        pooled = outputs.pooler_output
        projected = linear_proj(pooled)
        return projected

def get_random_images():
    indices = random.sample(range(100), 10)
    imgs = [first_100_images[i][0] for i in indices]
    imgs_np = [img.permute(1, 2, 0).numpy() for img in imgs]
    return imgs_np, indices

def generate_answer(image_tensor, question_index):
    if image_tensor is None:
        return "Please select an image first!"
    
    try:
        image_embedding = get_image_embedding(image_tensor, siglip_model, siglip_processor, linear_proj, device)
        question = questions[question_index]

        tokens = phi_tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
        question_embeds = phi_model.get_input_embeddings()(tokens.input_ids)

        image_embeds = image_text_proj(image_embedding).unsqueeze(1)
        combined = torch.cat([image_embeds, question_embeds], dim=1)

        attention_mask = torch.ones((1, combined.size(1)), dtype=torch.long).to(device)

        outputs = phi_model.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            max_new_tokens=100,
            num_beams=3,
            temperature=0.7,
            do_sample=True,
            pad_token_id=phi_tokenizer.pad_token_id,
            eos_token_id=phi_tokenizer.eos_token_id
        )
        return phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------- GRADIO UI -----------------------------

with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ CIFAR-10 Visual Question Answering with Phi-2 + SigLIP")

    selected_image_tensor = gr.State(None)
    image_indices = gr.State([])

    with gr.Row():
        with gr.Column():
            random_btn = gr.Button("🎲 Get Random Images")
            gallery = gr.Gallery(
                label="Select an Image",
                columns=[5],
                rows=[2],
                height="auto"
            )
        with gr.Column():
            selected_img = gr.Image(label="Selected Image", height=200)
            answer_box = gr.Textbox(label="Generated Answer", lines=4)
            q_buttons = [gr.Button(f"Q{i+1}: {q}") for i, q in enumerate(questions)]

    def on_random_click():
        images, indices = get_random_images()
        return {
            gallery: images,
            image_indices: indices,
            selected_image_tensor: None,
            selected_img: None,
            answer_box: ""
        }

    random_btn.click(
        fn=on_random_click,
        outputs=[gallery, image_indices, selected_image_tensor, selected_img, answer_box]
    )

    def on_image_select(evt: gr.SelectData, images, indices):
        idx = evt.index
        if images is None or idx >= len(images):
            return None, None, ""
        selected_idx = indices[idx]
        selected_tensor = first_100_images[selected_idx][0]
        return selected_tensor, images[idx], ""

    gallery.select(
        fn=on_image_select,
        inputs=[gallery, image_indices],
        outputs=[selected_image_tensor, selected_img, answer_box]
    )

    for i, btn in enumerate(q_buttons):
        btn.click(
            fn=lambda img_tensor, i=i: generate_answer(img_tensor, i),
            inputs=[selected_image_tensor],
            outputs=answer_box
        )

demo.queue(max_size=1).launch(show_error=True)
