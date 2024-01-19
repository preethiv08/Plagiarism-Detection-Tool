import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

def preprocess_data(data_path, sample_size=100):
    data = pd.read_csv(data_path, low_memory=False)

    data = data.dropna(subset=['abstract']).reset_index(drop=True)

    data = data.sample(sample_size)[['abstract']]

    return data

from sklearn.metrics.pairwise import cosine_similarity

model_path = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, output_attentions=False, output_hidden_states=True
)

def create_vector_from_text(tokenizer, model, text, MAX_LEN=510):
    input_ids = tokenizer.encode(
        text, add_special_tokens=True, max_length=MAX_LEN,
    )

    results = pad_sequences(
        [input_ids],
        maxlen=MAX_LEN,
        dtype="long",
        truncating="post",
        padding="post",
    )

    input_ids = results[0]

    # Create attention masks
    attention_mask = [int(i > 0) for i in input_ids]

    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

    model.eval()

    # from all 12 layers.
    with torch.no_grad():
        logits, encoded_layers = model(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            return_dict=False,
        )

    layer_i = 12  
    batch_i = 0  
    token_i = 0  

    vector = encoded_layers[layer_i][batch_i][token_i]

    vector = vector.detach().cpu().numpy()

    return vector

def create_vector_database(data):
    vectors = []
    for text in tqdm(data):
        vector = create_vector_from_text(tokenizer, model, text)
        vectors.append(vector)

    return vectors

df = pd.read_csv("C:\\Users\\Dell\\Downloads\\metadata\\metadata.csv")
df.dropna(subset=['abstract'], inplace=True)
abstracts = df['abstract'].tolist()[:100]

#Sentence Transformer Model
model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=False, output_hidden_states=True)

# Create vectors and save to a pickle file
vectors = create_vector_database(abstracts)
with open('vectors.pkl', 'wb') as f:
    pickle.dump(vectors, f)

def process_document(text):
    # Create a vector for given text and adjust it for cosine similarity search
    text_vect = create_vector_from_text(tokenizer, model, text)
    text_vect = np.array(text_vect)
    text_vect = text_vect.reshape(1, -1)

    return text_vect

def is_plagiarism(similarity_score, plagiarism_threshold):
    return similarity_score >= plagiarism_threshold

def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):
    top_N = 3
    
    query_vect = process_document(query_text)

    data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
    data["similarity"] = data["similarity"].apply(lambda x: x[0][0])

    similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N + 1]
    formated_result = similar_articles[["abstract", "similarity"]].reset_index(drop=True)

    similarity_score = formated_result.iloc[0]["similarity"]
    most_similar_article = formated_result.iloc[0]["abstract"]
    is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)

    plagiarism_decision = {
        'similarity_score': similarity_score,
        'is_plagiarism': is_plagiarism_bool,
        'most_similar_article': most_similar_article,
        'article_submitted': query_text,
    }

    return plagiarism_decision
