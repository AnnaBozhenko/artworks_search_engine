from flask import Flask, session, render_template, request, redirect, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModel

import numpy as np 
import torch

import faiss

app = Flask(__name__)
# set a config for secret key
app.secret_key = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app)

# dataset_checkpoint = "anna-bozhenko/aic-dataset"
dataset_checkpoint = "anna-bozhenko/artworks"
# bi_encoder_checkpoint = "multi-qa-MiniLM-L6-cos-v1"
bi_encoder_checkpoint = "anna-bozhenko/artworks-search-MiniLM-L6-cos-v1"
cross_encoder_checkpoint = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K=20
FAISS_TOP_K = 1000

# bi_encoder = SentenceTransformer(bi_encoder_checkpoint)
# bi_encoder.max_seq_length = 512
cross_encoder = CrossEncoder(cross_encoder_checkpoint)
ds = load_dataset(dataset_checkpoint, split="train") 
# ds = ds.select(range(100))# FOR DEBUG MODE ONLY 
embeddings = torch.from_numpy(np.array(ds['embeddings']).astype(np.float32)).cpu() # cpu powered, casted to np.float32

dim = len(embeddings[0])

# set up FAISS
n_seeds = 10
quantizer = faiss.IndexFlatL2(dim)
faiss_index = faiss.IndexIVFFlat(quantizer, dim, n_seeds)
# train
faiss_index.train(embeddings)
faiss_index.add(embeddings)
# add indices
faiss_index.add_with_ids(embeddings, np.arange(len(embeddings)))

bi_encoder = AutoModel.from_pretrained(bi_encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(bi_encoder_checkpoint)



# utilities (move to other dirs)
def get_indices(query, bi_encoder, cross_encoder, corpus, corpus_bi_embeddings, similarity_threshold=0.5, top_k=10, device="cuda"):
  q_embedding = bi_encoder.encode(query, convert_to_tensor=True).cpu() # cpu powered
  hits = util.semantic_search(q_embedding, corpus_bi_embeddings, top_k=1000)[0]
  crosses = [[query, corpus[hit["corpus_id"]]] for hit in hits]
  scores = cross_encoder.predict(crosses)
  cross_score_hits  = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
  return [corpus_id for corpus_id, _ in cross_score_hits[:top_k]]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = bi_encoder(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])


def get_results(indices, ds):
  try:
    return [ds[i] for i in indices]
  except Exception as e:
    print(f"Error while results {e}")


def extract_relevant_info(x):
  if x is None:
     return None
  return {
      'title': x['title'],
      'artist': x['artist'],
      'museum': x['museum'],
      'museum_region': x['museum_region'],
      'museum_country': x['museum_country'],
      'url': x['url'],
      'date_start': x['date_start'],
      'date_end': x['date_end'],
      'image_url': x[ 'image_url']
  }


@app.route("/")
def index():
    if 'search-history' not in session:
        session['search-history'] = []
    
    return render_template('index.html', 
                          results=None, 
                          query="", 
                          has_history=len(session['search-history']) > 0)


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    consider_history = request.form.get('consider-history', 'off') == 'on'
    
    if not query:
        return redirect(url_for('index'))
    else:
        # result_idxs = get_indices(query=query,
        #                    bi_encoder=bi_encoder, cross_encoder=cross_encoder,
        #                    corpus=ds["full_info"],
        #                    corpus_bi_embeddings=embeddings,
        #                    similarity_threshold=0.7,
        #                    top_k=top_k)
        # results = [extract_relevant_info(r) for r in get_results(result_idxs, ds)]
        
        # bi encoder layer
        question_embedding = get_embedding(query).detach().cpu().numpy()
        scores, indices = faiss_index.search(question_embedding, FAISS_TOP_K)
        indices = indices[0]
        crosses = [[query, full_info] for full_info in ds[indices]["full_info"]]
        scores = cross_encoder.predict(crosses)
        cross_score_hits  = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        corpus_ids = [corpus_id for corpus_id, _ in cross_score_hits[:TOP_K]]

        results = [extract_relevant_info(x) for x in ds.select(corpus_ids)]
        
        return render_template('index.html', 
                          results=results,  # Limit to top 20 results
                          query=query, 
                          has_history=len(session['search-history']) > 0,
                          history=session['search-history'])
       
        # return render_template('index.html', 
        #                   results=[],  # Limit to top 20 results
        #                   query=query, 
        #                   has_history=len(session['search_history']) > 0,
        #                   history=session['search_history'])

@app.route('/reset')
def reset():
    # Clear search history
    session['search-history'] = []
    return redirect(url_for('index'))


@app.route('/help')
def help_page():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)
