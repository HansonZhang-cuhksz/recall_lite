# recall_lite

This is a lite version of qwen_recall. This program only embeds images and retrieve at user query.

## Quick Start
Install dependencies:
```
pip install -r requirements.txt
```

Download UniME-Phi3.5-V-4.2B. e.g. From Modelscope:
```
modelscope download --model AI-ModelScope/UniME-Phi3.5-V-4.2B --local_dir ./phi
```

Run the program:
```
streamlit run main.py
```

Tested on windows, using RTX4060 GPU, approximately 3s/item(program limited at 5s/item).