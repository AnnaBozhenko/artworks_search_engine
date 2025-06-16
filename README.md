# artworks_search_engine

Semantic similarity search application for fine art (paintings, drawings).

---

## How to Run (Development Mode)

### 1. Create a Python virtual environment
```bash
python -m venv env
```

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
BI_ENCODER_CHECKPOINT=anna-bozhenko/artworks-bert-base-dot-v5
CROSS_ENCODER_CHECKPOINT=cross-encoder/ms-marco-MiniLM-L-6-v2
TOP_K=20            # extract K values from cross-encoded <query, description>
FAISS_TOP_K=50      # extract K values from bi-encoded descriptions
DEVICE=cpu

# DEPLOYMENT
PORT=8000
HOST=0.0.0.0
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
