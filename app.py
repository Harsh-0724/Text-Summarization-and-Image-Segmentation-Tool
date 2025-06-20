import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from pyvis.network import Network
import cv2

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Make sure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Text summarization functions

def rl_summarize(sentences, top_fraction=0.2):
    if not sentences:
        return ["No content to summarize."]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)

        if X.shape[1] == 0:
            return ["Content is mostly stop words or empty."]

        similarity_matrix = cosine_similarity(X)
        scores = similarity_matrix.sum(axis=1)

        ranked_sentences = [sentence for _, sentence in sorted(zip(scores, sentences), reverse=True)]

        top_n = max(1, int(len(ranked_sentences) * top_fraction))  # top 20% sentences
        summary = ranked_sentences[:top_n]
        return summary

    except ValueError:
        return ["Failed to process document. It may be empty or invalid."]


def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def create_similarity_graph(sentences):
    if not sentences:
        return ""

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(X)

    G = nx.Graph()

    for i, sent in enumerate(sentences):
        # Use short preview as label
        G.add_node(i, label=sent[:50] + ("..." if len(sent) > 50 else ""))

    # Add edges with weight > threshold
    threshold = 0.1
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            weight = similarity_matrix[i][j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])

    # Generate and return HTML string (no file)
    return net.generate_html()


# Image segmentation using KNN (KMeans)

def segment_image_knn(image_path, output_path, k=3):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image.")

    # Convert to RGB for better color segmentation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_vals = img_rgb.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # Define criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img_rgb.shape)

    # Save segmented image in BGR (OpenCV default)
    segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, segmented_bgr)
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf_file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['pdf_file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    sentences = sent_tokenize(text)

    summary = rl_summarize(sentences)
    graph_html = create_similarity_graph(sentences)

    if isinstance(summary, str):
        summary = sent_tokenize(summary)

    return render_template('summary.html', summary=summary, graph_html=graph_html)


@app.route('/segment', methods=['POST'])
def segment():
    if 'image_file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image_file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    segmented_filename = "segmented_" + filename
    segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)

    try:
        segment_image_knn(filepath, segmented_path)
    except Exception as e:
        return f"Failed to segment image: {e}"

    return render_template('segmentation.html', original=filename, segmented=segmented_filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
