import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Function to load pre-trained word embeddings from a file
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


# Function to compute cosine similarity between two word embeddings
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    return similarity

# Load pre-trained word embeddings from the GloVe file
embeddings = load_embeddings("glove.6B.50d.txt")

# List of example words for visualization
word_list = ["happiness", "success", "sadness", "failure", "joy", "achievement", "defeat", "unhappiness","sorrow"]

# Compute cosine similarity between pairs of example words
similarities = {}
for word1 in word_list:
    similarities[word1] = {}
    for word2 in word_list:
        similarity = cosine_similarity(embeddings[word1], embeddings[word2])
        similarities[word1][word2] = similarity

# Reduce dimensionality to 3 using PCA
pca = PCA(n_components=3)
word_vectors = np.array([embeddings[word] for word in word_list])
word_embeddings_3d = pca.fit_transform(word_vectors)

# Plot the reduced-dimensional embeddings in 3D space using Plotly
fig = go.Figure()
for i, word in enumerate(word_list):
    x, y, z = word_embeddings_3d[i]
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers+text", marker=dict(size=10), text=[word]))

# Set layout for the plot
fig.update_layout(scene=dict(xaxis_title='X Label', yaxis_title='Y Label', zaxis_title='Z Label'),
                  title='Word Embeddings in 3D Space')

# Show the plot
fig.show()