import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nltk
from rouge_score import rouge_scorer
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from Finetuning_SmolVLM import infer_title

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def calculate_metrics(actual_title, predicted_title):
    """
    Calculate evaluation metrics for captions
    
    Args:
        actual_title (str): Ground truth caption
        predicted_title (str): Model predicted caption
    
    Returns:
        dict: Metrics including BLEU, ROUGE, and METEOR scores
    """
    # Tokenize titles
    actual_title_tkns = nltk.word_tokenize(actual_title.lower())
    predicted_title_tkns = nltk.word_tokenize(predicted_title.lower())
    
    # BLEU Scores
    bleu_1 = nltk.translate.bleu_score.sentence_bleu(
        [actual_title_tkns], predicted_title_tkns, 
        weights=(1, 0, 0, 0)
    )
    bleu_2 = nltk.translate.bleu_score.sentence_bleu(
        [actual_title_tkns], predicted_title_tkns, 
        weights=(0.5, 0.5, 0, 0)
    )
    
    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(actual_title, predicted_title)
    
    # METEOR Score
    meteor = nltk.translate.meteor_score.meteor_score(
        [actual_title_tkns], predicted_title_tkns
    )
    
    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'METEOR': meteor
    }

def create_results_visualization(image_path, actual_title, predicted_title, metrics):
    """
    Create a visualization of the image with captions and metrics
    
    Args:
        image_path (str): Path to the image file
        actual_title (str): Ground truth caption
        predicted_title (str): Model predicted caption
        metrics (dict): Evaluation metrics
    
    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    plt.figure(figsize=(12, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    
    # Display captions and metrics
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    metrics_text = "Evaluation Metrics:\n"
    for metric, score in metrics.items():
        metrics_text += f"{metric}: {score:.4f}\n"
    
    plt.text(0.1, 0.8, f"Ground Truth:\n{actual_title}", fontsize=10, verticalalignment='top')
    plt.text(0.1, 0.5, f"Predicted:\n{predicted_title}", fontsize=10, verticalalignment='top')
    plt.text(0.1, 0.1, metrics_text, fontsize=10, verticalalignment='bottom', 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    return plt.gcf()

def process_image_captions(test_titles_path='test_titles.pkl', 
                            results_json_path='finetuned_2_epoch.json', 
                            output_dir='caption_evaluation_results2'):
    """
    Process image captions, generate metrics, and create visualizations
    
    Args:
        test_titles_path (str): Path to test titles pickle file
        results_json_path (str): Path to save/load results JSON
        output_dir (str): Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test titles
    with open(test_titles_path, "rb") as f:
        test_titles = pickle.load(f)
    
    # Results container
    results = {}
    metrics_summary = {
        'BLEU-1': [], 'BLEU-2': [], 
        'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': [], 
        'METEOR': []
    }
    
    # Check if results already exist
    if os.path.exists(results_json_path):
        with open(results_json_path, "r") as f:
            results = json.load(f)
    else:
        # Generate predictions if not exist
        for image_name, title in tqdm(test_titles.items(), desc="Inferring titles"):
            image_path = os.path.join("dataset/food_images/food_images", f"{image_name}.jpg")
            predicted_title = infer_title(image_path, "dish_title_model")
            print(f"Actual title: {image_name}, Predicted title: {predicted_title}")
            results[image_name] = {"actual_title": title, "predicted_title": predicted_title}
        
        # Save results
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=4)
    
    # Process each result
    for image_name, data in tqdm(results.items(), desc="Processing results"):
        image_path = os.path.join("dataset/food_images/food_images", f"{image_name}.jpg")
        actual_title = data['actual_title']
        predicted_title = data['predicted_title'][55:]  
        
        # Calculate metrics
        metrics = calculate_metrics(actual_title, predicted_title)
        
        # Store metrics for summary
        for metric, score in metrics.items():
            metrics_summary[metric].append(score)
        
        # Create visualization
        fig = create_results_visualization(image_path, actual_title, predicted_title, metrics)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{image_name}_result.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Calculate and print average metrics
    avg_metrics = {metric: np.mean(scores) for metric, scores in metrics_summary.items()}
    print("Average Metrics:")
    for metric, avg_score in avg_metrics.items():
        print(f"{metric}: {avg_score:.4f}")
    
    # Save summary metrics
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)

def main():
    process_image_captions()

if __name__ == "__main__":
    main()
