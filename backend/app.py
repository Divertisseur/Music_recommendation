from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from flask_cors import CORS
import torch
from torchvggish import vggish, vggish_input
import time
import faiss
import json
import math

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper to extract VGGish embedding from an audio file
vgg = vggish()
vgg.eval()

# Load FAISS index and metadata at startup
INDEX_PATH = 'music_index.faiss'
META_PATH = 'music_metadata.json'

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        music_metadata = json.load(f)
    print(f'Loaded FAISS index with {faiss_index.ntotal} vectors and {len(music_metadata)} metadata entries.')
else:
    faiss_index = None
    music_metadata = []
    print('No FAISS index or metadata found. Please run build_music_db.py.')

def extract_vggish_embedding(filepath):
    examples = vggish_input.wavfile_to_examples(filepath)
    if isinstance(examples, np.ndarray):
        examples_tensor = torch.from_numpy(examples)
    elif torch.is_tensor(examples):
        examples_tensor = examples
    else:
        raise ValueError(f"Unexpected type for examples: {type(examples)}")
    with torch.no_grad():
        emb = vgg(examples_tensor)
    return emb.mean(dim=0).numpy()

def similarity_to_percent(sim, min_sim=0.95, max_sim=1.0):
    sim = max(min_sim, min(max_sim, sim))
    norm = (sim - min_sim) / (max_sim - min_sim)
    percent = 100 * (norm ** 3)
    return round(percent)

# Dummy music database with VGGish embeddings (replace with real vectors)
music_db = [
    {
        'artist': 'Artist 1',
        'title': 'Song A',
        'vector': np.random.rand(128),  # Replace with real VGGish vectors
    },
    {
        'artist': 'Artist 2',
        'title': 'Song B',
        'vector': np.random.rand(128),
    },
    {
        'artist': 'Artist 3',
        'title': 'Song C',
        'vector': np.random.rand(128),
    },
]

@app.route('/upload', methods=['POST'])
def upload_file():
    print('Received upload request')
    start_time = time.time()
    n = request.args.get('n', default=3, type=int)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename) if file.filename else 'audio.wav'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    if faiss_index is None or not music_metadata:
        return jsonify({'error': 'Music database not loaded. Please build the database.'}), 500
    try:
        print('Starting VGGish inference...')
        vggish_start = time.time()
        vggish_vec = extract_vggish_embedding(filepath)
        vggish_end = time.time()
        print(f'VGGish inference took {vggish_end - vggish_start:.2f} seconds')
    except Exception as e:
        return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500
    # Search FAISS for nearest neighbors
    query = vggish_vec.astype('float32').reshape(1, -1)
    D, I = faiss_index.search(query, n)
    recommendations = []
    similarities = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(music_metadata):
            continue
        meta = music_metadata[idx]
        db_vec = faiss_index.reconstruct(int(idx))
        sim = np.dot(vggish_vec, db_vec) / (np.linalg.norm(vggish_vec) * np.linalg.norm(db_vec) + 1e-8)
        similarities.append(sim)
        percent = similarity_to_percent(sim)
        print(f"Recommendation: {meta.get('artist','')} - {meta.get('title','')}, similarity: {sim:.4f}, percentage: {percent}%")
        # Try to get filename from metadata, fallback to constructing from artist/title
        filename = meta.get('filename')
        if not filename:
            # Try to reconstruct filename as in music_db
            artist = meta.get('artist', '').strip()
            title = meta.get('title', '').strip()
            # Try both dash and en dash
            possible_filenames = [
                f"{artist} - {title}.wav",
                f"{artist} – {title}.wav"
            ]
            for fname in possible_filenames:
                if os.path.exists(os.path.join('music_db', fname)):
                    filename = fname
                    break
            if not filename:
                filename = possible_filenames[0]  # fallback
        recommendations.append({
            'artist': meta.get('artist', ''),
            'title': meta.get('title', ''),
            'similarity': float(sim),
            'percent': percent,
            'filename': filename
        })
    print(f"All similarities in this batch: {similarities}")
    total_time = time.time() - start_time
    print(f'Total request time: {total_time:.2f} seconds')
    return jsonify({'message': 'File received', 'filename': filename, 'recommendations': recommendations})

@app.route('/music/<path:filename>')
def serve_music(filename):
    return send_from_directory('music_db', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port) 