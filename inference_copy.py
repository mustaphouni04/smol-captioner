import os
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt

# Importing necessary models
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, CLIPModel, CLIPProcessor
from tqdm import tqdm

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

def compute_image_embeddings(clip_model, clip_processor, images):
    image_inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**image_inputs)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    return image_embeddings

def select_most_similar_examples(clip_model, clip_processor, test_image, train_titles, top_k=2):
    train_image_embeddings = []
    train_image_names = []
    batch_size = 64
    train_image_keys = list(train_titles.keys())
    
    for i in range(0, len(train_image_keys), batch_size):
        batch_keys = train_image_keys[i:i+batch_size]
        batch_images = [
            Image.open(os.path.join("dataset/food_images/food_images", f"{img_name}.jpg")).convert("RGB")
            for img_name in batch_keys
        ]
        batch_embeddings = compute_image_embeddings(clip_model, clip_processor, batch_images)
        train_image_embeddings.append(batch_embeddings)
        train_image_names.extend(batch_keys)
        for img in batch_images:
            img.close()
    
    train_image_embeddings = torch.cat(train_image_embeddings, dim=0)
    test_image_embedding = compute_image_embeddings(clip_model, clip_processor, [test_image])
    similarities = torch.mm(test_image_embedding, train_image_embeddings.T).squeeze()
    top_k_indices = similarities.topk(k=top_k).indices.cpu().numpy()
    
    selected_examples = []
    for idx in top_k_indices:
        img_name = train_image_names[idx]
        selected_examples.append({
            "image_path": os.path.join("dataset/food_images/food_images", f"{img_name}.jpg"),
            "title": train_titles[img_name]
        })
    return selected_examples

def infer_title_with_icl(test_image, train_titles, model_dir="dish_title_model"):
    clip_model, clip_processor = load_clip_model()
    selected_examples = select_most_similar_examples(clip_model, clip_processor, test_image, train_titles)
    model = Idefics3ForConditionalGeneration.from_pretrained(model_dir).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_dir)
    messages = [{"role": "user", "content": []}]
    input_images = []
    for i, example in enumerate(selected_examples, 1):
        example_image = Image.open(example["image_path"]).convert("RGB")
        messages[0]["content"].extend([
            {"type": "text", "text": f"Example {str(i)}: {example['title']}."},
            {"type": "image"},
        ])
        input_images.append(example_image)
    messages[0]["content"].extend([
        {"type": "text", "text": "What is the title of this dish based on the example above?"},
        {"type": "image"},
    ])
    input_images.append(test_image)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text.strip()], 
        images=[input_images], 
        return_tensors="pt", 
        padding=True
    ).to(DEVICE)
    outputs = model.generate(**inputs)
    predicted_title = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    for img in input_images:
        img.close()
    return predicted_title[6:]

def compute_metrics(reference, prediction):
    # Tokenize the reference and prediction
    reference_tokens = nltk.word_tokenize(reference.lower())
    prediction_tokens = nltk.word_tokenize(prediction.lower())
    
    # Compute BLEU scores
    smoothing_function = SmoothingFunction().method1
    bleu_1 = sentence_bleu([reference_tokens], prediction_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    bleu_2 = sentence_bleu([reference_tokens], prediction_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    
    # Compute ROUGE-L
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(reference, prediction)['rougeL'].fmeasure
    
    # Compute METEOR
    meteor = meteor_score([' '.join(reference_tokens)], ' '.join(prediction_tokens))
    
    return bleu_1, bleu_2, rouge, meteor

def plot_image_with_caption(image, predicted_caption, ground_truth, metrics, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_caption}\n"
              f"Ground Truth: {ground_truth}\n"
              f"BLEU-1: {metrics[0]:.2f}, BLEU-2: {metrics[1]:.2f}, ROUGE-L: {metrics[2]:.2f}, METEOR: {metrics[3]:.2f}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_image_captions(train_titles_path='train_titles.pkl', 
                            test_titles_path='test_titles.pkl', 
                            results_json_path='icl_finetuned_results.json', 
                            output_dir='icl_caption_evaluation_results'):
    os.makedirs(output_dir, exist_ok=True)
    with open(train_titles_path, "rb") as f:
        train_titles = pickle.load(f)
    with open(test_titles_path, "rb") as f:
        test_titles = pickle.load(f)
    results = {}
    for image_name, title in tqdm(test_titles.items(), desc="Inferring titles"):
        image_path = os.path.join("dataset/food_images/food_images", f"{image_name}.jpg")
        test_image = Image.open(image_path).convert("RGB")
        predicted_title = infer_title_with_icl(test_image, train_titles)
        bleu_1, bleu_2, rouge, meteor = compute_metrics(title, predicted_title)
        results[image_name] = {
            "actual_title": title,
            "predicted_title": predicted_title,
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "rouge_l": rouge,
            "meteor": meteor
        }
        output_image_path = os.path.join(output_dir, f"{image_name}.png")
        plot_image_with_caption(test_image, predicted_title, title, (bleu_1, bleu_2, rouge, meteor), output_image_path)
        test_image.close()
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
    return results

def main():
    process_image_captions()

if __name__ == "__main__":
    main()
