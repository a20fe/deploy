import pandas as pd
import spacy
import string
import re
from spacy.lang.en import English
from spacy import load
from spacy.lang.en.stop_words import STOP_WORDS
import pickle
from pymongo import MongoClient
from fastapi import FastAPI, Request, Form
import uvicorn 
from sklearn.preprocessing import  LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


app = FastAPI()

app.secret_key = 'PASS@152bn'
nlp = spacy.load("en_core_web_sm")
label_encoder = LabelEncoder()
v = TfidfVectorizer()

# Load the saved model
try:
    with open('recommender.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
    model = None

# Connect to MongoDB
client = MongoClient('mongodb+srv://mahmoudrdwan32:123123123@elmentor.fphwgku.mongodb.net/?retryWrites=true&w=majority')
db = client['elmentor']
collection = db['users']

# Function to retrieve data from MongoDB and store it in the session
def store_data_in_session():
    cursor = collection.find({'mentor': True})
    data_list = list(cursor)
    for item in data_list:
        item['_id'] = str(item['_id'])
    data = pd.DataFrame(data_list)
    return data


# Preprocessing function
stop_words = STOP_WORDS
def preprocess(text):
    if isinstance(text, list):  # Check if text is a list
        text = ' '.join(text)  # Convert list to string
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_alpha and token.text.lower() not in stop_words:
            token_text = token.text.lower()
            token_text = token_text.translate(str.maketrans('', '', string.punctuation))
            lemmatized_token = token.lemma_
            filtered_tokens.append(lemmatized_token)
    return " ".join(filtered_tokens)

# Route to recommend
@app.get("/recommend")
async def recommend(request: Request):
    # Receive new data
    data = store_data_in_session()

    professional_titles = data['professionalTitle'].astype(str)
    specializations = data['specialization'].astype(str)
    tech_stacks = data['techStack'].astype(str)
    
    data['preprocessed_data'] = ' '.join(professional_titles) + ' ' + ' '.join(specializations) + ' ' + ' '.join(tech_stacks)
    data['preprocessed_data'] = data['preprocessed_data'].apply(preprocess)
    data['preprocessed_data'] = data['preprocessed_data'].replace('\n', ' ').replace('||', ' ').replace(',', ' ').replace('  ', ' ').replace(':', ' ')
    data['levelOfExperience'] = label_encoder.fit_transform(data['levelOfExperience'])
    
    # Vectorize preprocessed data
    vectorized_data = v.fit_transform(data['preprocessed_data'])
    tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=v.get_feature_names_out())
    df_processed = pd.concat([data[['levelOfExperience','userName']], tfidf_df], axis=1)
    df_processed.set_index('userName', inplace=True)

    #fitting data into model
    df_processed_matrix = csr_matrix(df_processed.values)
    model.fit(df_processed_matrix)

    sample_data_point = df_processed.iloc[5]
    sample_data_point_matrix = csr_matrix(sample_data_point.values.reshape(1, -1))
    distances, indices = model.kneighbors(sample_data_point_matrix, n_neighbors=4)

    neighbor_data_list = []

    for i in range(len(distances.flatten())):
       neighbor_index = int(indices.flatten()[i])  # Convert numpy.int64 to int
       neighbor_distance = float(distances.flatten()[i])  # Convert numpy.float64 to float

       neighbor_info = {
        "Neighbor": i + 1,
        "Index in DataFrame": neighbor_index,
        "Distance": neighbor_distance,
       }
       neighbor_data_list.append(neighbor_info)

    return {"neighbors": neighbor_data_list}


if __name__=="__main__":
   uvicorn.run(app, host='localhost', port=8000)