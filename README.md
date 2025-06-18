# artworks_search_engine

Semantic similarity search application for fine art (paintings, drawings).

---

## How to Run (Development Mode)

### 1. Create a Python virtual environment
```bash
python -m venv env
```

### ⚠️ Important
This application is configured to work with private bi-encoder and cross-encoder models, and a private embedding dataset. You can proceed in one of the following ways:
- upload your own models and dataset to Hugging Face Hub with the same interface;
- follow the fine-tuning example notebook in fine_tune_models.ipynb to train and host your own SentenceTransformer models.

### 2. Create a `.env` file
Paste the configuration below.  
Confidential variables are marked with `[...]`.  
⚠️ **Do not expose `.env` publicly**.

<details>
<summary>Example .env</summary>

```env
# PYTHON
VENV_PYTHON=.\env\Scripts\python.exe

# FLASK
FLASK_APP=app.py
FLASK_DEBUG=true
SESSION_TYPE=redis
SESSION_REDIS=redis://redis:6379
SESSION_PERMANENT=0
SESSION_USE_SIGNER=1

# HUGGING FACE
HF_TOKEN=[PASTE YOUR TOKEN]
TF_ENABLE_ONEDNN_OPTS=0
DATASET_CHECKPOINT=anna-bozhenko/artworks
EMBEDDED_ARTWORKS_FIELD=[PASTE YOUR PATH]
BI_ENCODER_CHECKPOINT=[PASTE YOUR MODEL]
CROSS_ENCODER_CHECKPOINT=[PASTE YOUR MODEL]
TOP_K=20            # extract K values from cross-encoded <query, description>
FAISS_TOP_K=50      # extract K values from bi-encoded descriptions
DEVICE=cpu

# DEPLOYMENT
PORT=8000
HOST=0.0.0.0
```

</details>

---

### 🔐 Embedding Descriptions and Uploading Dataset to Hugging Face Hub

After setting up your .env file with the necessary Hugging Face credentials and paths, you can embed the artwork descriptions using your own bi-encoder and push the resulting embeddings to a private Hugging Face dataset repository.

<details>
<summary>Example of embedding data</summary>

```from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Login to Hugging Face Hub using your token
login(os.getenv('HF_TOKEN'))

# Load dataset from Hugging Face Hub (or your own uploaded dataset)
artworks = load_dataset(os.getenv('DATASET_CHECKPOINT'), split="train")

# Load your fine-tuned SentenceTransformer bi-encoder model
bi_encoder_checkpoint = SentenceTransformer(os.getenv('BI_ENCODER_CHECKPOINT'))  # 👈 PASTE your BI-ENCODER MODEL NAME or local path
device = "cuda"  # or "cpu"
private = True   # whether your output dataset should be private
repo_name = ""   # 👈 PASTE your Hugging Face DATASET NAME

# Crop input text length (e.g., limit to N words)
max_words_n = 128  # adjust if needed
artworks_full_info = artworks.map(
    lambda batch: {
        "cropped_full_info": [" ".join(x.split()[:max_words_n]) for x in batch["full_info"]]
    },
    batched=True,
    batch_size=1000
)["cropped_full_info"][:]

# Generate sentence embeddings
encoded_data = bi_encoder_checkpoint.encode(
    artworks_full_info,
    show_progress_bar=True,
    device=device,
    convert_to_numpy=True
)

# Wrap and push to Hugging Face Hub
embeddings = Dataset.from_dict({"bert-base-dot-v5": encoded_data})  # 👈 keep field name consistent with `app.py`
embeddings.push_to_hub(repo_name, private=private)
```
</details>
---

### 3. Dockerize and run the application

#### 3.1 Build Docker image
```bash
docker compose build
```

#### 3.2 Run the container
```bash
docker compose up -d
```

#### 3.3 Enter the container
```bash
docker exec -it [your-container-id] sh
```

#### 3.4 Run the app inside the container
```bash
python app.py
```

#### 3.5 Stop the app
Press `Ctrl+C`

#### 3.6 Exit the container shell
Press `Ctrl+D` or type `exit`

#### 3.7 Shut down the container
```bash
docker compose down
```

---

### 4. View the application

Open in browser:

- http://127.0.0.1:8000
- http://172.19.0.3:8000 *(Docker internal network, may vary)*

---

## Dataset

The dataset of artworks is hosted on the Hugging Face Hub:  
🔗 https://huggingface.co/datasets/anna-bozhenko/artworks
