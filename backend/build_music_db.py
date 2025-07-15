import os
import json
import numpy as np
import faiss
import torch
from torchvggish import vggish, vggish_input

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# Helper to extract VGGish embedding from an audio file
def extract_vggish_embedding(filepath, vgg):
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

def scan_audio_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return files

def main(audio_folder, index_path='music_index.faiss', meta_path='music_metadata.json'):
    vgg = vggish()
    vgg.eval()
    audio_files = scan_audio_files(audio_folder)
    print(f'Found {len(audio_files)} audio files.')
    vectors = []
    metadata = []
    for path in audio_files:
        print(f'Processing: {path}')
        try:
            vec = extract_vggish_embedding(path, vgg)
            vectors.append(vec.astype(np.float32))
            # Use filename as title, or parse artist/title from filename if you want
            metadata.append({
                'filename': os.path.basename(path),
                'fullpath': path,
                'artist': '',
                'title': os.path.splitext(os.path.basename(path))[0],
            })
        except Exception as e:
            print(f'Failed to process {path}: {e}')
    if not vectors:
        print('No vectors extracted. Exiting.')
        return
    vectors_np = np.stack(vectors).astype(np.float32)
    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)
    faiss.write_index(index, index_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved FAISS index to {index_path} and metadata to {meta_path}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python build_music_db.py <audio_folder>')
    else:
        main(sys.argv[1]) 