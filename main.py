import streamlit as st
import time

from infer import decode_task, query, paths
from utils import device, memory_dir

st.title("Recall Lite")

@st.cache_resource
def init():
    global memory_dir, device

    import threading
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    import torch
    import os
    
    for file in os.listdir(os.path.join(memory_dir, "images")):
        file_path = os.path.join(memory_dir, "images", file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    base_model_path = r".\phi"
    print("Model Loading...")
    transform = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.float16, quantization_config=quantization_config)
    print("Model Loaded.")
    transform.tokenizer.padding_side = "left"
    transform.tokenizer.padding = True

    decode_thread = threading.Thread(target=decode_task, args=(model, transform))
    decode_thread.daemon = True
    decode_thread.start()

    st.session_state.default_timestamp = 0

    time.sleep(5)

    return model, transform

model, transform = init()

if len(paths) > 0:
    timestamp = st.slider("Timestamp", 0, len(paths)-1, st.session_state.default_timestamp, 1)
    st.image(paths[timestamp], use_container_width=True)
    st.session_state.default_timestamp = timestamp

# Place the chat bar at the bottom
with st.form("Ask Recall", clear_on_submit=True):
    user_input = st.text_input("Ask Qwen Recall", "")
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        result = query(model, transform, user_input, k=3)
        print(result)
        for image in result:
            try:
                st.image(image, use_container_width=True)
            except:
                print(f"Error displaying image: {image}")
            st.write(image)