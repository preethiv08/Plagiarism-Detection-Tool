from flask import Flask, render_template, request
import pandas as pd
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

df = pd.read_csv("C:\\Users\\Dell\\Downloads\\metadata\\metadata.csv")
df.dropna(subset=['abstract'], inplace=True)
abstracts = df['abstract'].tolist()[:100]

model = SentenceTransformer('average_word_embeddings_glove.6B.300d')

# Load vectors
with open('vectors.pkl', 'rb') as f:
    vectors = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plagiarism', methods=['POST'])
def plagiarism():
    user_input = request.form['user_input']

    corpus_embeddings = model.encode(abstracts, convert_to_tensor=True)
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # Calculate cosine similarity score
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(5, len(abstracts))
    top_results = torch.topk(cos_scores, k=top_k)
    cos_scores_float = cos_scores.numpy().astype(float)

    most_similar_index = top_results.indices[0].item()

    most_similar_abstract = abstracts[most_similar_index]

    output1_float = cos_scores_float[0]
    output2_float = vectors[most_similar_index].max()
    average = (output1_float + output2_float) / 2

    return render_template('results.html', average=average, most_similar_abstract=most_similar_abstract)

if __name__ == '__main__':
    app.run(debug=True)
