



import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))  # Obtener las stopwords en inglés
    words = word_tokenize(text)  # Tokenizar el texto
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Filtrar las stopwords
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(" ".join(filtered_words)) 
    return " ".join([token.lemma_ for token in doc])  # Unir las palabras filtradas en un solo string

def create_corpus(texts,schemas):
    corpus = []  # Inicializar el corpus
    for i,text in enumerate(texts):
        # Concatenar el recurso, descripción y estructura en un solo string y convertirlo a minúsculas
        content = text["resource"].lower() + " " + text["description"].lower() + " "+ schemas[i].lower()
        preprocessed_text = preprocess_text(content)
        corpus.append(preprocessed_text)  # Añadir el contenido al corpus
    return corpus

# Load the NDJSON file with UTF-8 encoding and parse each line separately
def load_json(path):
    # Load the NDJSON file with UTF-8 encoding and parse each line separately
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            data.append(json.loads(line))
    return data

# Usage
schemas = load_json("C:\\Users\\Alvaro\\ChatGPT\\ChatGPT\\Fase3\\json_schemas\\dataset_schemas.ndjson")

def load_dataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        texts = []
        for line in lines:
            json_line = json.loads(line)  # Convertir cada línea a un objeto JSON
            texts.append(json_line)       # Añadir el objeto JSON a la lista de textos
    return texts

schemas = load_json("C:\\Users\Alvaro\\ChatGPT\\ChatGPT\\Fase3\\json_schemas\\dataset_schemas.ndjson")
texts = load_dataset("C:\\Users\Alvaro\\ChatGPT\\ChatGPT\\Fase3\\datasetRecursos.ndjson")

for i,schema in enumerate(texts):
    print(str(schemas[i]).lower())