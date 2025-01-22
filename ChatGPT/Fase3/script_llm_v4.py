# Importar las librerías necesarias
import os
import json
import time
import numpy as np
import pandas as pd
import nltk
import requests
from typing import Literal
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from together import Together

nltk.download('stopwords')
nltk.download('punkt')  # Necesario para tokenizar

def load_dataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        texts = []
        for line in lines:
            json_line = json.loads(line)  # Convertir cada línea a un objeto JSON
            texts.append(json_line)       # Añadir el objeto JSON a la lista de textos
    return texts

def create_corpus(texts, schemas):
    corpus = []  # Inicializar el corpus
    for i, text in enumerate(texts):
        # Concatenar el recurso, descripción y estructura en un solo string y convertirlo a minúsculas
        content = "Resource name : "+ text["resource"].lower() + "Attribute definitions : " + " "+ str(schemas[i]).lower()
        preprocessed_text = preprocess_text(content)
        corpus.append(preprocessed_text)  # Añadir el contenido al corpus
    return corpus

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))  # Obtener las stopwords en inglés
    words = word_tokenize(text)  # Tokenizar el texto
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Filtrar las stopwords
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(" ".join(filtered_words))
    return " ".join([token.lemma_ for token in doc])  # Unir las palabras filtradas en un solo string

def tfidf_faiss_similarity(corpus, query):
    vectorizer = TfidfVectorizer()  # Crear el vectorizador TF-IDF
    tfidf_matrix = vectorizer.fit_transform(corpus).toarray().astype(np.float32)  # Vectorizar el corpus

    query_vec = vectorizer.transform([query]).toarray().astype(np.float32)  # Vectorizar la consulta

    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)[0]  # Calcular la similitud del coseno
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}  # Crear un diccionario con los puntajes de similitud
    return similarity_dict

def sent_transformer_encoder(corpus, query):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Obtener el embedding de la consulta
    query_vec = model.encode([query])[0]
    # Obtener los embeddings del corpus
    corpus_vecs = model.encode(corpus)

    # Calcular la similitud del coseno entre el embedding de la consulta y los embeddings del corpus
    similarity_scores = cosine_similarity([query_vec], corpus_vecs)[0]

    # Crear un diccionario con los puntajes de similitud
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}
    return similarity_dict

def bm25_similarity(corpus, query):
    tokenized_corpus = [doc.split(" ") for doc in corpus]  # Tokenizar el corpus
    tokenized_query = query.split(" ")  # Tokenizar la consulta
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.75, b=0.75)  # Crear el índice BM25

    bm25_scores = bm25_index.get_scores(tokenized_query)  # Obtener los puntajes BM25 para la consulta
    similarity_dict = {i: score for i, score in enumerate(bm25_scores)}  # Crear un diccionario con los puntajes de similitud

    return similarity_dict

def reciprocal_rank_fusion(rank_lists, k=60):
    rrf_scores = defaultdict(float)  # Crear un diccionario para almacenar los puntajes RRF

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            rrf_scores[doc_id] += 1.0 / (k + rank)  # Calcular la recíproca del rango y sumar al puntaje del documento

    sorted_rrf_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)  # Ordenar los puntajes de mayor a menor
    return sorted_rrf_scores

def transformers_similarity(corpus, query):

    model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")  # Cargar el modelo de Transformers

    # Obtener el embedding de la consulta
    query_vec = model.encode([query])[0]
    # Obtener los embeddings del corpus
    corpus_vecs = model.encode(corpus)

    # Calcular la similitud del coseno entre el embedding de la consulta y los embeddings del corpus
    similarity_scores = cosine_similarity([query_vec], corpus_vecs)[0]

    # Crear un diccionario con los puntajes de similitud
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}

    return similarity_dict

def fasttext_similarity(corpus, query):
    # Preprocesar el corpus y la consulta
    processed_corpus = [word_tokenize(doc) for doc in corpus]
    processed_query = word_tokenize(query)

    # Entrenar el modelo FastText
    model = SentenceTransformer('fasttext-wiki-news-subwords-300')

    # Obtener el vector de la consulta
    query_vec = model.encode([" ".join(processed_query)])[0]

    # Obtener los vectores de cada documento en el corpus
    corpus_vecs = model.encode([" ".join(doc) for doc in processed_corpus])

    # Calcular la similitud del coseno
    similarity_scores = cosine_similarity([query_vec], corpus_vecs)[0]
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}  # Crear un diccionario con los puntajes de similitud
    return similarity_dict

def get_top_texts(vec_similarity_dict, bm25_similarity_dict, univ_sent_encoder_similarity_dic, fasttext_similarities, profile_corpus, top_k=5):
    # Obtener los índices de los textos más similares usando TF-IDF y ordenar de mayor a menor similitud
    vec_indexes = sorted(vec_similarity_dict, key=vec_similarity_dict.get, reverse=True)
    # Obtener los índices de los textos más similares usando BM25 y ordenar de mayor a menor similitud
    bm25_indexes = sorted(bm25_similarity_dict, key=bm25_similarity_dict.get, reverse=True)
    # Obtener los índices de los textos más similares usando Universal Sentence Encoder y ordenar de mayor a menor similitud
    univ_sent_encoder_indexes = sorted(univ_sent_encoder_similarity_dic, key=univ_sent_encoder_similarity_dic.get, reverse=True)
    # Obtener los índices de los textos más similares usando FastText y ordenar de mayor a menor similitud
    fasttext_indexes = sorted(fasttext_similarities, key=fasttext_similarities.get, reverse=True)

    # Se combinan los rankings de los tres métodos utilizando la fusión de rango recíproco (RRF)
    combined_rankings = reciprocal_rank_fusion([vec_indexes, bm25_indexes, fasttext_indexes, univ_sent_encoder_indexes], k=top_k)

    # Se procesan los índices, obteniendo únicamente los k mejores
    final_indexes = [index for index, _ in combined_rankings[:top_k]]

    # Recuperar los textos relevantes del corpus utilizando los índices obtenidos
    relevant_texts = [profile_corpus[i] for i in final_indexes]

    return relevant_texts, final_indexes




def generate_description(attributes):
    query = f"Given the following table attributes: {', '.join(attributes)}, generate a general description about the dataset and its contents. Aiming to give a great context for understanding the data"
    GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT_FHIR")
    payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert medical data analyst. Your task is to generate a general description of the dataset based on the provided table attributes."
                        "The description should provide an overview of the dataset and its contents, giving context for understanding the data."                        
                    )
                },
                {
                    "role": "user",
                    "content": f"{query}"
                }
            ],
            "temperature": 0,
        }

    response = send_request(payload, GPT4V_ENDPOINT)
    response_message = response['choices'][0]['message']
    if 'function_call' in response_message:
        response_content = response_message['function_call']['arguments']
    else:
        response_content = response_message['content']
    
    return response_content

def send_request(payload, endpoint):
        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }
        response = requests.post(endpoint, headers=headers, json=payload)
        return json.loads(response.content)
        
def generate_response(query, modelName, context, json_schema, num_schemas, iterations=5):
    
    if modelName == "GPT":
        GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT_FHIR")
        functions = []
        for i in range(num_schemas):
            schema_title = json_schema[i].get("title", f"Mapping_{i + 1}")
            schema_title_formatted = schema_title.replace(" ", "_").replace("-", "_")
            functions.append({
                "name": f"{schema_title_formatted}",
                "description": "Generate a mapping between table column names and the top 3 FHIR resource attributes based on similarity.",
                "parameters": json_schema[i]
            })

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"""
                        ##################### CONTEXT ####################  
                        You are a domain expert assistant specializing in mapping clinical tabular data to FHIR resources (FHIR R4).  
                        You have been given information about table attributes (including their names, descriptions, and sample values) and your task is to determine the 3 most specific FHIR attributes that best correspond to each table attribute. You may choose attributes from any FHIR resource as you see fit.  

                        Review the given attributes thoroughly and choose the mappings that most accurately reflect the semantic meaning and structure of the table attributes in the context of FHIR. Reason about the potential matches internally, but do not include the reasoning steps in the final answer.
                        The documents you have to consider to answer the query is: {context}
                        ##################### TASK ####################  
                        Based on the provided table attribute information and standard FHIR attribute definitions, determine the top 3 most specific FHIR attributes that correspond to each table attribute, listed in order of relevance (most specific first).
                    
                        ##################### OUTPUT FORMAT ####################  
                        Provide your answer as a JSON object, using the following structure:

                        
                                    "table_attribute_name": "attributeName",
                                    "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
                        

                        Each key corresponds to the table attribute name, and its value is an array of exactly three strings. If fewer than three matches are found, use "No additional attribute found" to fill the empty slots.

                        ##################### EXAMPLE ####################  
                        For example, if the table attribute is "patient_birthDate":

                        
                            "table_attribute_name": "patient_birthDate",
                            "fhir_attribute_name": "Patient.birthDate", "Encounter.birthDate", "Patient.anchorAge"
                        

                        ##################### ADDITIONAL INSTRUCTIONS ####################  
                        - Consider the attribute name, description, and sample values when determining the best FHIR attributes.
                        - Return only the final JSON object without additional commentary.
                        - Use the FHIR R4 specification as the reference (https://www.hl7.org/fhir/).
                        - Double-check your mappings before returning the final result.
                        """
                        
                    )
                },
                {
                    "role": "user",
                    "content": f"{query}"
                }
            ],
            "temperature": 0,
            "top_p": 0,
            "functions": functions,
            "function_call": "auto"
        }

        response = send_request(payload, GPT4V_ENDPOINT)
        print(response)
        response_message = response['choices'][0]['message']
        if 'function_call' in response_message:
            response_content = response_message['function_call']['arguments']
        else:
            response_content = response_message['content']

        time.sleep(30)
        for i in range(iterations):
            reflection_payload = {
                "messages": [
                    {"role": "system", "content": f"Iteration {i+1}: Initial mapping of top 3 attributes for each column."},
                    {"role": "assistant", "content": response_content},
                    {
                        "role": "system",
                        "content": (
                            f"""
                            #######CONTEXT#######
                            You are a domain expert assistant specializing in mapping clinical tabular data to FHIR resources (FHIR R4). 
                            You have been given information about table attributes (including their names, descriptions, and sample values) and your task is to determine the 3 most specific FHIR attributes that best correspond to each table attribute. You may choose attributes from any FHIR resource as you see fit.
                            You previously provided a mapping between table column names and the top 3 FHIR resource attributes based on similarity.
                            {response_content}
                            #######INPUT_DATA#######
                            The documents you have to consider to answer the query is: {context}
                            #######TASK#######
                            Review the mapping above for completeness and accuracy. Ensure the three best-matched attributes are selected.
                            #######OUTPUT_FORMAT#######
                            Provide your answer as a JSON object in the following format:
                                    {{
                                    "table_attribute_name": "attributeName",
                                    "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
                                    }}
                            """
                            

                        )
                    },
                ],
                "temperature": 0,
                "top_p": 0,
                "functions": functions,
                "function_call": "auto"
            }

            response = send_request(reflection_payload, GPT4V_ENDPOINT)
            response_message = response['choices'][0]['message']
            response_content = response_message['function_call']['arguments'] if 'function_call' in response_message else response_message['content']
            time.sleep(30)


        return response_content

    elif modelName == "Llama":
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        tools = []
        for i in range(num_schemas):
            schema_title = json_schema[i].get("title", f"Mapping_{i + 1}")
            schema_title_formatted = schema_title.replace(" ", "_").replace("-", "_")
            tools.append({
                "function": {
                    "name": f"{schema_title_formatted}",
                    "description": "Generate a mapping between table column names and the top 3 FHIR resource attributes based on similarity.",
                    "parameters": {"type": "object", "properties": json_schema[i]}
                },
                "type": "function"
            })

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {
                    'role': 'system',
                    'content': (
                        f"""
                        ##################### CONTEXT ####################  
                        You are a domain expert assistant specializing in mapping clinical tabular data to FHIR resources (FHIR R4).  
                        You have been given information about table attributes (including their names, descriptions, and sample values) and your task is to determine the 3 most specific FHIR attributes that best correspond to each table attribute. You may choose attributes from any FHIR resource as you see fit.  

                        Review the given attributes thoroughly and choose the mappings that most accurately reflect the semantic meaning and structure of the table attributes in the context of FHIR. Reason about the potential matches internally, but do not include the reasoning steps in the final answer.
                        The documents you have to consider to answer the query is: {context}
                        ##################### TASK ####################  
                        Based on the provided table attribute information and standard FHIR attribute definitions, determine the top 3 most specific FHIR attributes that correspond to each table attribute, listed in order of relevance (most specific first).
                    
                        ##################### OUTPUT FORMAT ####################  
                        Provide your answer as a JSON object, using the following structure:

                        
                                    "table_attribute_name": "attributeName",
                                    "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
                        

                        Each key corresponds to the table attribute name, and its value is an array of exactly three strings. If fewer than three matches are found, use "No additional attribute found" to fill the empty slots.

                        ##################### EXAMPLE ####################  
                        For example, if the table attribute is "patient_birthDate":

                        
                            "table_attribute_name": "patient_birthDate",
                            "fhir_attribute_name": "Patient.birthDate", "Encounter.birthDate", "Patient.anchorAge"
                        

                        ##################### ADDITIONAL INSTRUCTIONS ####################  
                        - Consider the attribute name, description, and sample values when determining the best FHIR attributes.
                        - Return only the final JSON object without additional commentary.
                        - Use the FHIR R4 specification as the reference (https://www.hl7.org/fhir/).
                        - Double-check your mappings before returning the final result.
                        """
                        f"The query is: {query}"
                    )
                },
                {
                    'role': 'user',
                    'content': "Follow the query instructions"
                }
            ],
            temperature=0,
            tool_choice="auto",
            tools=tools
        )

        response_message = response.choices[0].message
        function_arguments = response_message.tool_calls[0].function.arguments if response_message.tool_calls else response_message.content
        time.sleep(30)


        for i in range(iterations):
            reflection_prompt = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": (
                        f"""
                            #######CONTEXT#######
                            You are a domain expert assistant specializing in mapping clinical tabular data to FHIR resources (FHIR R4). 
                            You have been given information about table attributes (including their names, descriptions, and sample values) and your task is to determine the 3 most specific FHIR attributes that best correspond to each table attribute. You may choose attributes from any FHIR resource as you see fit.
                            You previously provided a mapping between table column names and the top 3 FHIR resource attributes based on similarity.
                            {function_arguments}
                            #######INPUT_DATA#######
                            The documents you have to consider to answer the query is: {context}
                            #######TASK#######
                            Review the mapping above for completeness and accuracy. Ensure the three best-matched attributes are selected.
                            #######OUTPUT_FORMAT#######
                            Provide your answer as a JSON object in the following format:
                                    {{
                                    "table_attribute_name": "attributeName",
                                    "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
                                    }}
                            """
                    )},
                    {
                        "role": "user",
                        "content": (
                            "Reflect on the mapping and ensure the top 3 attributes are accurately selected for each column."
                        )
                    }
                ],
                temperature=0,
                tools=tools,
                tool_choice="auto"
            )

            response_message = reflection_prompt.choices[0].message
            function_arguments = response_message.tool_calls[0].function.arguments if response_message.tool_calls else response_message.content
            time.sleep(30)

        return function_arguments

def load_data(path):
    # Cargar los datos tabulares desde un archivo CSV
    data = pd.read_csv(path)
    # Se convierte la data a formato JSON
    data_json = data.to_json(orient='records')
    return data_json

def load_json(path):
    # Load the JSON file with UTF-8 encoding to avoid encoding issues
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_ndjson(path):
    # Load the NDJSON file with UTF-8 encoding and parse each line separately
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            data.append(json.loads(line))
    return data

def load_clusters(path):
    with open(path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    return clusters

def filter_texts_and_schemas_by_resources(resources, texts, schemas_data):
    filtered_texts = []
    filtered_schemas = []
    for idx, text in enumerate(texts):
        if text['resource'] in resources:
            filtered_texts.append(text)
            filtered_schemas.append(schemas_data[idx])
    return filtered_texts, filtered_schemas

# ------------------------------------------------------------
path = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/"
filename = "filtered_data_attributes.json"  # Nombre del archivo JSON
data_structured = load_json(path + filename)  # Cargar los datos estructurados desde un archivo JSON
# ------------------------------------------------------------
texts = load_dataset('C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/datasetRecursos.ndjson')  # Cargar el conjunto de datos desde un archivo NDJSON

# Cargar los esquemas enriquecidos
schemas_data = load_ndjson("C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/json_schemas/enriched_dataset_schemas.ndjson")
# ------------------------------------------------------------

# Cargar los datos de los clusters
clusters_data = load_json("C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/clusters_v10.json")

# Iterar sobre cada cluster
# Define el cluster y atributo desde donde quieres empezar
start_cluster_name = "Cluster 5"  # Reemplaza con el nombre del cluster deseado

# Bandera para indicar si ya hemos llegado al punto de inicio
start_processing = False

# Iterar sobre cada cluster
for cluster_name, cluster_info in clusters_data.items():
    attributes = cluster_info.get('Attributes', [])
    resources = [res['Resource'] for res in cluster_info.get('Top Resources', [])]

    print(f"Processing Cluster: {cluster_name}")
    print("Resources:", resources)
    print("Attributes:", attributes)
    print("------------------------------------------------------------")

    # Verificar si hemos llegado al punto de inicio
    if not start_processing:
        if cluster_name == start_cluster_name:
            start_processing = True
        else:
            continue  # Si no hemos llegado, continúa al siguiente cluster

    # Buscar la información de los atributos en data_structured
    attr_infos = []
    for attribute_name in attributes:
        attr_info = next((item for item in data_structured if item.get("Attribute name") == attribute_name), None)
        if attr_info:
            attr_infos.append(attr_info)
        else:
            print(f"Attribute {attribute_name} not found in data_structured")
            continue

    if not attr_infos:
        print(f"No attributes found for cluster {cluster_name}")
        continue

    # Construir la consulta para todos los atributos del cluster
    query = "##################### INPUT DATA ##################\n"
    for attr_info in attr_infos:
        attribute_name = attr_info.get("Attribute name")
        description = attr_info.get("Description")
        values = attr_info.get("Values")
        query += f"Attribute Name: {attribute_name}\n"
        query += f"Description: {description}\n"
        query += f"Sample Values: {values}\n"
        query += "----------------------------------------\n"

    #table_description = generate_description(query)
    #print("Table Description:", table_description)

    query += """
    ##################### TASK ##################
    Based on the provided attribute information and the FHIR attribute definitions, determine the most specific FHIR attribute that corresponds to each table attribute. Be as precise as possible in your mapping.
    ##################### OUTPUT FORMAT ##################
    Provide your answer as a JSON object in the following format:
         {
         "table_attribute_name": "attributeName",
         "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
         },
         {
         "table_attribute_name": "attributeName",
         "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
         },

    }
    """
    # Filtrar texts y schemas_data basados en los recursos
    filtered_texts, filtered_schemas = filter_texts_and_schemas_by_resources(resources, texts, schemas_data)

    # Crear el corpus
    context = create_corpus(filtered_texts, schemas=filtered_schemas)

    # Generar la respuesta usando el modelo GPT o Llama
    with open("C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/mapeo_clusters_v12_llama.json", 'a', encoding='utf-8') as archivo:
        archivo.write(f"Cluster: {cluster_name}\n")

        print("------------------------LLAMA RESPONSE------------------------------------")
        response = generate_response(query, "Llama", context,filtered_schemas, len(filtered_schemas), 1)
        print(response)  # Imprimir la respuesta generada por Llama
        archivo.write("LLAMA Response:\n")
        archivo.write(str(response))
        archivo.write("\n")

        #print("------------------------GPT RESPONSE---------------------------------------")
        #response = generate_response(query, "GPT", context,filtered_schemas, len(filtered_schemas), 1)
        #print(response)  # Imprimir la respuesta generada por GPT
        #archivo.write("GPT Response:\n")
        #archivo.write(str(response))
        #archivo.write("\n\n")

