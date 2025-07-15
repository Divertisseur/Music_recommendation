# Music Recommendation System

A modern music recommendation system that leverages deep audio embeddings (VGGish), efficient similarity search (FAISS), and a simple web interface to recommend similar songs based on an uploaded audio file.

## Features

- **Audio-based Recommendations:** Upload a music clip and get similar tracks recommended using deep learning audio embeddings.
- **Deep Learning Embeddings:** Uses [VGGish](https://github.com/harritaylor/torchvggish) (PyTorch) to extract semantic audio features.
- **Efficient Search:** Utilizes [FAISS](https://github.com/facebookresearch/faiss) for fast nearest neighbor search over large music databases.
- **REST API:** Flask backend with endpoints for uploading audio and retrieving recommendations.
- **Frontend:** Simple HTML frontend for easy interaction.
- **Cross-Origin Support:** CORS enabled for frontend-backend communication.

## Project Structure

```
music_recommendation/
  backend/
    app.py                # Flask API server
    build_music_db.py     # Script to build FAISS index and metadata
    requirements.txt      # Python dependencies
  frontend/
    index.html            # Simple web interface
```

## How It Works

1. **Audio Upload:** User uploads a music file via the frontend or API.
2. **Feature Extraction:** The backend extracts a VGGish embedding from the audio.
3. **Similarity Search:** The embedding is compared to a database of music embeddings using FAISS.
4. **Recommendations:** The most similar tracks are returned, along with similarity scores and metadata.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/music_recommendation.git
cd music_recommendation
```

### 2. Backend Setup

#### a. Create a Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### b. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### c. Build the Music Database

Prepare your music files (WAV format recommended) in a folder (e.g., `music_db/`). Then run:

```bash
python build_music_db.py
```

This will extract VGGish embeddings for each track, build a FAISS index, and save metadata.

#### d. Run the Backend Server

```bash
python app.py
```

The Flask server will start on `http://127.0.0.1:5000` by default.

### 3. Frontend Usage

Simply open `frontend/index.html` in your browser. Make sure the backend is running and accessible.

## API Endpoints

### `POST /upload`
- **Description:** Upload an audio file and receive music recommendations.
- **Params:**
  - `file`: Audio file (form-data)
  - `n` (optional): Number of recommendations (default: 3)
- **Response:**
  - `recommendations`: List of similar tracks with artist, title, similarity, and filename.

### `GET /music/<filename>`
- **Description:** Download or stream a music file from the database.

## Technologies Used

- **PyTorch**: Deep learning framework for VGGish model
- **torchvggish**: Pretrained VGGish audio embedding model
- **FAISS**: Fast similarity search and clustering
- **Flask**: Lightweight Python web framework
- **librosa**: Audio processing
- **NumPy, SciPy**: Scientific computing
- **HTML/JS**: Simple frontend

## Notes
- Ensure your music files are in the correct format (WAV recommended) and placed in the `music_db/` directory before building the database.
- The system is designed for educational/demo purposes. For production, consider security, scalability, and advanced frontend features.

## Credits
- VGGish model: [Google Research](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
- torchvggish: [harritaylor/torchvggish](https://github.com/harritaylor/torchvggish)
- FAISS: [Facebook Research](https://github.com/facebookresearch/faiss)

## License

MIT License. See [LICENSE](LICENSE) for details. 