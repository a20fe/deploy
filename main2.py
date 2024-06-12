import pandas as pd
import spacy
import string
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import uvicorn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
import logging
from typing import List

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
    logging.error(f"Unexpected {err=}, {type(err)=}")
    model = None

# Preprocessing function
stop_words = spacy.lang.en.stop_words.STOP_WORDS
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

# Define Pydantic model for input validation
class UserData(BaseModel):
    professionalTitle: str
    specialization: str
    techStack: List[str]  # Accepting techStack as a list of strings
    levelOfExperience: str
    userName: str

# Route to recommend
@app.post("/recommend")
async def recommend(data: List[UserData]):
    try:
        # Convert input data to DataFrame
        data_dicts = [user.dict() for user in data]
        data_df = pd.DataFrame(data_dicts)
    except ValidationError as e:
        logging.error(f"Data validation error: {e}")
        raise HTTPException(status_code=422, detail="Invalid data format")

    try:
        professional_titles = data_df['professionalTitle'].astype(str)
        specializations = data_df['specialization'].astype(str)
        tech_stacks = data_df['techStack'].apply(lambda x: ' '.join(x))  # Convert list to string
    except KeyError as e:
        logging.error(f"Missing key in the JSON data: {e}")
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")

    data_df['preprocessed_data'] = professional_titles + ' ' + specializations + ' ' + tech_stacks
    data_df['preprocessed_data'] = data_df['preprocessed_data'].apply(preprocess)
    data_df['preprocessed_data'] = data_df['preprocessed_data'].str.replace('\n', ' ').str.replace('||', ' ').str.replace(',', ' ').str.replace('  ', ' ').str.replace(':', ' ')
    data_df['levelOfExperience'] = label_encoder.fit_transform(data_df['levelOfExperience'])
    
    # Vectorize preprocessed data
    vectorized_data = v.fit_transform(data_df['preprocessed_data'])
    tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=v.get_feature_names_out())
    df_processed = pd.concat([data_df[['levelOfExperience', 'userName']], tfidf_df], axis=1)
    df_processed.set_index('userName', inplace=True)

    # Fitting data into model
    df_processed_matrix = csr_matrix(df_processed.values)
    model.fit(df_processed_matrix)

    sample_data_point = df_processed.iloc[0]  # Use the first sample for testing
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

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
