import os
import google.generativeai as genai
import json
import pickle
from tqdm import tqdm

with open("test_titles.pkl", "rb") as f:
    test_titles = pickle.load(f)

with open("train_titles.pkl", "rb") as f:
    train_titles = pickle.load(f)

with open("validation_titles.pkl", "rb") as f:
    validation_titles = pickle.load(f)


genai.configure(api_key="basurita")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Make your response short, generate a candidates for alternative titles that mean the same as this dish:\nGolden and Crimson Beet Salad with Oranges, Fennel, and Feta\n\nJust return the results, don't say nothing else",
      ],
    },
    {
      "role": "model",
      "parts": [
        "* Jeweled Beet & Citrus Salad\n* Orange-Fennel Beet Salad\n* Two-Tone Beet Salad with Feta\n* Sweet & Savory Beetroot Salad\n* Citrus Beetroot & Feta Salad\n",
      ],
    }
  ]
)

alternatives_train = {}
alternatives_test = {}
alternatives_validation = {}
splits = ["train", "test", "validation"]

for split in splits:
    if split == "train":
        titles = train_titles
        alternatives = alternatives_train
    elif split == "test":
        titles = test_titles
        alternatives = alternatives_test
    else:
        titles = validation_titles
        alternatives = alternatives_validation
    
    img_titles = list(titles.items())
    
    def split(list_a, chunk_size):
      for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]

    img_titles = list(split(img_titles, 75))

    for img_title in tqdm(img_titles, desc=f"Generating alternatives for {split}"):
      image_names = [img_title[i][0] for i in range(len(img_title))]
      titles = [img_title[i][1] for i in range(len(img_title))]

      combined_titles = "\n".join(titles)
      response = chat_session.send_message([f"""Make your response short, 
                           generate candidates 
                           for alternative titles 
                           that mean the same as these dishes:
                           \n{combined_titles}\n\n

                           Put the original title first, 
                           then the alternatives.
                           Just return the results, don't say nothing else"""])

      print(response.text)

      for i, title in enumerate(titles):
        # response is the ith line in the response
        alternatives[image_names[i]] = response.text.split("\n")[i]
      
      print(list(alternatives.items())[:4])
    
    # save the dictionary to a file
    with open(f"alternatives_{split}.json", "w") as f:
        json.dump(alternatives, f)
        



