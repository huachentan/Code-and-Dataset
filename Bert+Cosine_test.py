import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
model = BertModel.from_pretrained("bert-base-uncased").to(device)  
model.eval() 

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device) 
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  

with open("Example of PlagiarismCheck/Originaltext.json", "r", encoding="utf-8") as f:
    original_texts_data = json.load(f)
    original_texts = [item["text"] for item in original_texts_data["Original text"]]
with open("Example of PlagiarismCheck/AItext.json", "r", encoding="utf-8") as f:
    ai_texts = json.load(f)

results = {}
for i, orig_text in enumerate(original_texts): 
    orig_embedding = get_embedding(orig_text) 
    similarities = []
    key = f"Original text{i+1}"
    if key in ai_texts:
        ai_text_list = ai_texts[key]
        for ai_text in ai_text_list: 
            ai_embedding = get_embedding(ai_text["text"])
            similarity_score = cosine_similarity([orig_embedding], [ai_embedding])[0][0]
            similarities.append(float(similarity_score))  
        results[f"original_{i+1}"] = similarities
    else:
        print(f"Warning: No corresponding AI texts found for Original text {i+1}")

results_file = "/home/wzx/similarity_text/text/similarity_results.json"
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)  

print(f"The similarity results are calculated and have been saved to {results_file}")