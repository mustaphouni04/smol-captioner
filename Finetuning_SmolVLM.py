import os
import pickle
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force use of the first GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# **1. Setup: Constants and Model Configuration**
USE_QLORA = True
MODEL_ID = "dish_title_model"

# Initialize processor and model
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure QLoRA settings
if USE_QLORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
        init_lora_weights="gaussian",
        use_dora=False,
        inference_mode=False,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bf16 precision for the model
    )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0"
        
    ).to(DEVICE)

    if "default" not in model.peft_config:
        model.add_adapter(lora_config)
        model.enable_adapters()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config).to(DEVICE)  # Explicitly move to DEVICE
else:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # Ensure the model uses bf16 precision
        device_map="cuda:0",
    ).to(DEVICE)  # Ensure the model uses bf16 precision


# **2. Dataset Preparation**
class DishTitleDataset(Dataset):
    def __init__(self, pickle_path, image_dir, processor):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = list(self.data.keys())[idx]
        title = self.data[image_name]
        image_path = os.path.join(self.image_dir, image_name + ".jpg") # maybe issue lies here
        image = Image.open(image_path).convert("RGB")
        return {"image": image, "title": title}

    def collate_fn(self, examples):
        texts = []
        images = []
        for example in examples:
            title = example["title"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the title of this dish?"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": title}],
                },
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([example["image"]])

        # Process on CPU and keep tensors on CPU
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch

# **3. Training**
def train_model(pickle_path, image_dir, output_dir, num_epochs=1):
    dataset = DishTitleDataset(pickle_path, image_dir, processor)

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="epoch",
        output_dir=output_dir,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.collate_fn,
    )
    print("Training model...")
    
    # Explicitly move model to device before training
    trainer.model.to(DEVICE)
    
    # Train model
    trainer.train(resume_from_checkpoint=True)
    print("Training complete!")
    
    # Save the model locally
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved locally at {output_dir}")
# **4. Inference**
def infer_title(image_path, model_dir):
    # Load saved model and processor
    model = Idefics3ForConditionalGeneration.from_pretrained(model_dir).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_dir)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the title of this dish?"},
                {"type": "image"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text.strip()], images=[[image]], return_tensors="pt", padding=True).to(DEVICE)

    outputs = model.generate(**inputs)
    predicted_title = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return predicted_title

# **5. Run Training and Inference**
if __name__ == "__main__":
    # Paths to dataset and images
    PICKLE_PATH = "train_titles.pkl"
    IMAGE_DIR = "dataset/food_images/food_images"
    OUTPUT_DIR = "dish_title_model"

    # Train the model
    train_model(PICKLE_PATH, IMAGE_DIR, OUTPUT_DIR, num_epochs=2)

    '''
    # Example inference
    test_image_path = "./images/sample_dish.jpg"
    predicted_title = infer_title(test_image_path, OUTPUT_DIR)
    print(f"Predicted Title: {predicted_title}")
    '''
