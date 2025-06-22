import faiss
import time
from PIL import Image
import torch
from torch.nn import functional as F
import os

from capture import take_screenshot
from utils import device, max_file

dim = 3072
index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
index_bak = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
ids = []

def decode_task(model, transform):
    global index, index_bak, ids, max_file

    next_id = 0
    paths = []
    
    img_prompt = '<|user|>\n<|image_1|>\nSummary above image in one word: <|end|>\n<|assistant|>\n'

    while True:
        try:
            pth = take_screenshot()
        except:
            print("Error taking screenshot")
            time.sleep(0.5)
            continue
        paths.append(pth)

        if len(paths) > max_file:
            oldest_file = paths.pop(0)
            try:
                os.remove(oldest_file)
            except Exception as e:
                print(f"Error removing file {oldest_file}: {e}")

        input_image = [Image.open(pth)]
        ids.append((next_id, pth))

        inputs_image = transform(text=img_prompt,
                    images=input_image, 
                    return_tensors="pt", 
                    padding=True).to(device)
        
        with torch.no_grad():
            emb_image = model(**inputs_image, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            emb_image = F.normalize(emb_image, dim=-1).cpu()

        index.add_with_ids(emb_image, next_id)
        print(f"Added image with ID {next_id} to index. Current index size: {index.ntotal}")
        if len(ids) > max_file:
            index_bak.add_with_ids(emb_image, next_id)
        if len(ids) > 2 * max_file:
            index = index_bak
            index_bak = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
            ids = ids[-max_file:]

        next_id = (next_id + 1) % 1000

def query(model, transform, text, k=3):
    global index, ids

    if len(ids) == 0:
        return []

    text_prompt = '<|user|>\n<sent>\nSummary above sentence in one word: <|end|>\n<|assistant|>\n'
    input_texts = text_prompt.replace('<sent>', text)
    inputs_text = transform(text=input_texts,
                    images=None,
                    return_tensors="pt", 
                    padding=True)
    for key in inputs_text:
        inputs_text[key] = inputs_text[key].to(device)

    with torch.no_grad():
        emb_text = model(**inputs_text, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        emb_text = F.normalize(emb_text, dim=-1).cpu()

    D, I = index.search(emb_text, k=k if len(ids) > k else len(ids))
    d_i = list(zip(D[0], I[0]))
    d_i = sorted(d_i, key=lambda x: x[0], reverse=True)
    result_ids = [item[1] for item in d_i]
    result_paths = []
    for id, pth in ids:
        if id in result_ids:
            result_paths.append(pth)

    return result_paths