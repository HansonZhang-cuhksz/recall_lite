import faiss
import time
from PIL import Image
import torch
from torch.nn import functional as F

from utils import device
import shared

dim = 3072
index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
ids = []

def decode_task(model, transform):
    global index, ids

    next_id = 0
    
    img_prompt = '<|user|>\n<|image_1|>\nSummary above image in one word: <|end|>\n<|assistant|>\n'

    while True:
        start_time = time.time()

        if shared.pth is None:
            time.sleep(0.5)
            continue

        input_image = [Image.open(shared.pth)]
        ids.append((next_id, shared.pth))
        shared.pth = None

        inputs_image = transform(text=img_prompt,
                    images=input_image, 
                    return_tensors="pt", 
                    padding=True).to(device)
        
        with torch.no_grad():
            emb_image = model(**inputs_image, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            emb_image = F.normalize(emb_image, dim=-1).cpu()

        index.add_with_ids(emb_image, next_id)
        print(f"Added image with ID {next_id} to index. Current index size: {index.ntotal}")
        if len(ids) > 100:
            index.remove_ids(ids.pop(0))
        next_id = (next_id + 1) % 1000

        shared.period = max(time.time() - start_time + 1, 5)

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