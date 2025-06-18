from flask import session, Flask, render_template, request, redirect, url_for, jsonify
from flask_session import Session
from redis import Redis
from werkzeug.middleware.proxy_fix import ProxyFix
import os
from time import time
from dotenv import load_dotenv

from datasets import load_dataset, load_from_disk
from sentence_transformers import CrossEncoder, SentenceTransformer
from huggingface_hub import login

import numpy as np 

import faiss

from datetime import datetime
import csv

import logging
from logging import FileHandler
import time

load_dotenv() 
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
login(token=os.getenv('HF_TOKEN'))
PREDEFINED_QUERIES = ['Paintings in the Baroque style',
 'Impressionist landscape artworks',
 'Renaissance portraits of noble families',
 'Cubist representations of still life',
 'Surrealist depictions',
 'Italian art from the 15th century',
 'Italian art',
 'Dutch Golden Age paintings',
 'Japanese woodblock prints',
 'Ancient Egyptian frescoes',
 'French neoclassical graphs',
 "Pablo Picasso's early works",
 'Portraits of a noble woman',
 'portrait of a peasant girl',
 'Mythological scenes with Greek gods']

# Налаштування логування у файл
def setup_logger():
    logger = logging.getLogger("fine_arts_search_engine")
    logger.setLevel(logging.INFO)

    # Створення file handler із кодуванням UTF-8
    file_handler = FileHandler("search_engine_app.log", encoding='utf-8')
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Щоб уникнути дублювання записів
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()


def log_request(func):
    def wrapper(*args, **kwargs):
        log_info = [f"\nfunction: {func.__name__}()"]    
        for k, v in kwargs.items():
            if k in ["bi_encoder", "cross_encoder"]:
                log_info.append(f"{k}: {v.name}")
            else:
                log_info.append(f"{k}: {v}")
        
        log_info = '\n'.join(log_info)

        start_time = time.time()
        logger.info(f"Початок обробки | {log_info}")
        
        result = func(*args, **kwargs)
        
        duration = time.time() - start_time
        logger.info(f"Завершено обробку | Час виконання: {duration:.3f} сек")
        return result
    return wrapper


app = Flask(__name__)
# set a config for secret key
app.secret_key = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app)

# set up session server 
app.config['SESSION_TYPE'] = os.getenv('SESSION_TYPE', "redis")
app.config['SESSION_REDIS'] = Redis.from_url(os.getenv('SESSION_REDIS'))
app.config['SESSION_PERMANENT'] = bool(int(os.getenv('SESSION_PERMANENT', False)))
app.config['SESSION_USE_SIGNER'] = bool(int(os.getenv('SESSION_USE_SIGNER', True)))
Session(app)

#
dataset_checkpoint = os.getenv('DATASET_CHECKPOINT')
# bi_encoder_checkpoint = "anna-bozhenko/artworks-search-MiniLM-L6-cos-v1"
bi_encoder_checkpoint = os.getenv('BI_ENCODER_CHECKPOINT')
cross_encoder_checkpoint = os.getenv('CROSS_ENCODER_CHECKPOINT')
TOP_K = int(os.getenv('TOP_K'))
FAISS_TOP_K = int(os.getenv('FAISS_TOP_K'))
device = int(os.getenv('DEVICE', "cpu"))

bi_encoder = SentenceTransformer(bi_encoder_checkpoint)
bi_encoder.name = bi_encoder_checkpoint
# bi_encoder.max_seq_length = 512
cross_encoder = CrossEncoder(cross_encoder_checkpoint)
cross_encoder.name = cross_encoder_checkpoint
ds = load_dataset(dataset_checkpoint, split="train") 

# embeddings = np.array(ds['embeddings']).astype(np.float32)
# TODO: need to leverage another storage for embeddings 
embeddings = load_dataset(os.getenv('EMBEDDED_ARTWORKS_FIELD'), split="train").with_format("numpy")[:]['bert-base-dot-v5']
# embeddings = np.array(embeddings['mpnet-base-dot-v1'][:]).astype("float32")
dim = embeddings.shape[1]

# set up FAISS
faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
faiss.normalize_L2(embeddings)
faiss_index.add_with_ids(embeddings, np.arange(embeddings.shape[0]))
faiss.write_index(faiss_index, 'artworks_description.index')


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


# @log_request
# def extract_results(query, bi_encoder=bi_encoder, cross_encoder=cross_encoder, faiss_top_k=FAISS_TOP_K, top_k=TOP_K):
#     query_vec = bi_encoder.encode([query], convert_to_numpy=True, device=device, normalize_embeddings=True).astype('float32')
#     D, I = faiss_index.search(query_vec, k=faiss_top_k)
#     crosses = [(query, ds["full_info"][i]) for i in I[0]]
#     scores = cross_encoder.predict(crosses)
#     cross_score_hits  = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
#     cross_score_hits = [I[0][i] for i, _ in cross_score_hits]
#     corpus_ids = cross_score_hits[:top_k]
#     results = [extract_relevant_info(x) for x in ds.select(corpus_ids)]
#     return results


@log_request
def extract_results(query, bi_encoder=bi_encoder, cross_encoder=cross_encoder, faiss_top_k=FAISS_TOP_K, top_k=TOP_K):
    query_vec = bi_encoder.encode([query], convert_to_numpy=True, device=device, normalize_embeddings=True).astype('float32')
    D, I = faiss_index.search(query_vec, k=faiss_top_k)
    results = [extract_relevant_info(x) for x in ds.select(I[0])]
    return results


@app.route("/feedback")
def feedback_page():
    return render_template('feedback.html', total_queries=len(PREDEFINED_QUERIES))


@app.route("/start_feedback")
def start_feedback():
    session["feedback_active"] = True
    session["current_query_index"] = 0
    return generate_feedback_query()
    


def generate_feedback_query():
    current_query_index = session.get('current_query_index', 0)
    
    if current_query_index >= len(PREDEFINED_QUERIES):
        # All queries completed
        session['feedback_active'] = False
        return render_template('feedback_complete.html')
    
    query = PREDEFINED_QUERIES[current_query_index]
    # Generate search results for this query
    results = extract_results(query, bi_encoder=bi_encoder, cross_encoder=cross_encoder, faiss_top_k=50, top_k=20)
    session['current_feedback_results'] = results

    return render_template('feedback_query.html',
                          query=query,
                          results=results,
                          query_number=current_query_index + 1,
                          total_queries=len(PREDEFINED_QUERIES),
                          progress_percentage=int((current_query_index / len(PREDEFINED_QUERIES)) * 100))


def save_feedback_to_csv(query, result_url, result_rank, is_relevant):
    feedback_dir = 'users_feedback'
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)
    
    csv_file = os.path.join(feedback_dir, 'feedback_data.csv')

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        fieldnames = ['timestamp', 'query', 'result_url', 'result_rank', 'is_relevant']
        writer = csv.DictWriter(file, fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result_url': result_url,
            'result_rank': result_rank,
            'is_relevant': 1 if is_relevant else 0
        })


@app.route('/submit_query_feedback', methods=['POST'])
def submit_query_feedback():
    if not session.get('feedback_active'):
        return redirect(url_for('feedback_page'))
    
    relevant_results = request.form.getlist('relevant_results')
    relevant_indices = [int(idx) for idx in relevant_results]

    current_query_i = session.get('current_query_index', 0)
    current_query = PREDEFINED_QUERIES[current_query_i]
    feedback_results = session.get('current_feedback_results', [])

    for feedback_result_i, feedback_result in enumerate(feedback_results):
        is_relevant = feedback_result_i in relevant_indices
        save_feedback_to_csv(query=current_query,
                             result_url=feedback_result['url'], #
                             result_rank = feedback_result_i + 1,
                             is_relevant=is_relevant
            )
    # move to next query
    session['current_query_index'] = current_query_i + 1
    return generate_feedback_query()


@app.route('/stop_feedback')
def stop_feedback():
    session['feedback_active'] = False
    session.pop('current_query_index', None)
    session.pop('current_feedback_results', None)
    return redirect(url_for('feedback_page'))



@app.route("/")
def index():
    if 'search_history' not in session:
        session['search_history'] = []
    
    return render_template('index.html', 
                          results=None, 
                          query="", 
                          has_history=len(session['search_history']) > 0)


@app.route('/search', methods=['POST', 'GET'])
def search():
    if 'search_history' not in session:
        session['search_history'] = []
    
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    query = request.form.get('query', '')
    query_str = query
    consider_history = request.form.get('consider_history', 'off') == 'on'
    
    if not query:
        print("redirected to index")
        return redirect(url_for('index'))

    session["last_query"] = query
    session["consider_history"] = consider_history
    if consider_history and session['search_history']:
        history_context = " ".join(session['search_history'][-3:])
        query = f"{history_context} {query}" # 

    if query not in session['search_history']:
        session['search_history'] = session['search_history'] + [query_str]
        # keep last 5 qrs
        if len(session['search_history']) > 5:
            session['search_history'] = session['search_history'][-5:]
    
    results = extract_results(query, bi_encoder=bi_encoder, cross_encoder=cross_encoder, faiss_top_k=FAISS_TOP_K, top_k=TOP_K)
    # store all results for pagintion
    session["all_results"] = results
    print(f"fst result's title: `{results[0]['title']}`")
    batch_size = 10
    first_batch = results[:batch_size]
    print(f"Total results n: {len(results)}, tail n: {len(results)}")
    print(f"Session keys: {list(session.keys())}")
    # if 'search-history' not in session:
    #     session['search-history'] = []
    return render_template('index.html', 
                      results=first_batch,  # Limit to top 20 results
                      query=query, 
                      has_history=len(session['search_history']) > 0,
                      history=session['search_history'],
                      has_more=len(results) > batch_size,
                      total_count=len(results))


@app.route("/load_more", methods=["GET"])
def load_more():
    offset = int(request.args.get("offset", 0))
    batch_size = int(request.args.get("batch_size", 10))
    print(f"Session keys: {list(session.keys())}")
    all_results = session.get("all_results", [])
    next_batch = all_results[offset:offset + batch_size] if all_results else all_results
    has_more = len(all_results) > (offset + batch_size)
    print(f"Total results n: {len(all_results)}, tail n: {len(all_results[offset:])}")
    return jsonify({
        'results': next_batch,
        'has_more': has_more,
        'total_count': len(all_results),
        'next_offset': offset + batch_size if has_more else -1
    })

@app.route('/reset')
def reset():
    # Clear search history
    session['search_history'] = []
    return redirect(url_for('index'))


@app.route('/help')
def help_page():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
    # app.run(debug=True)
                # host=os.getenv('HOST', "0.0.0.0"), 
        # port=int(os.getenv('PORT', 5000)), 
