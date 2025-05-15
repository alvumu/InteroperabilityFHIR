# Importar las librerías necesarias
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import spacy
import tensorflow as tf
from together import Together
from transformers import BertModel, BertTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import tensorflow_hub as hub
from rank_bm25 import BM25Okapi
import pandas as pd
from gensim.models import Word2Vec
import requests
from typing import Literal


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

    
def create_corpus(texts):
    corpus = []
    profiles = []
    for text in texts:
        # Concatenar el recurso, descripción y estructura en un solo string y convertirlo a minúsculas
        content = text["resource"].lower() + " " + text["description"].lower() + " "
        preprocessed_text = preprocess_text(content)
        corpus.append(preprocessed_text)  # Añadir el contenido al corpus
        profiles.append(str(text["structure"]).lower())  # Añadir la estructura al corpus
    return corpus, profiles


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
    print(similarity_dict)
    return similarity_dict

def set_tensorflow_deterministic():
    # Fijar semilla aleatoria
    seed_value = 42
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Configuración adicional para reproducibilidad
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
set_tensorflow_deterministic()
def load_univ_sent_encoder_model():
    
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_url)  # Cargar el modelo desde TensorFlow Hub
    return model

def univ_sent_encoder(corpus, query):
    model = load_univ_sent_encoder_model()  # Cargar el modelo
    query_embedding = model([query])  # Obtener el embedding de la consulta

    similarity_dict = {}
    for idx, doc in enumerate(corpus):
        doc_embedding = model([doc])  # Obtener el embedding del documento
        similarity_scores = cosine_similarity(query_embedding, doc_embedding)  # Calcular la similitud del coseno
        similarity_dict[idx] = similarity_scores[0][0]  # Guardar la similitud en el diccionario
    print(similarity_dict)
    return similarity_dict

def word2vec_similarity(corpus, query):

    # Preprocesar el corpus y la consulta
    processed_query = query.split()

    # Entrenar el modelo Word2Vec
    model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4, seed=42)

    # Obtener el vector de la consulta
    query_vec = np.mean([model.wv[word] for word in processed_query if word in model.wv], axis=0).reshape(1, -1)

    # Obtener los vectores de cada documento en el corpus
    corpus_vecs = []
    for doc in corpus:
        doc_vec = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
        corpus_vecs.append(doc_vec)
    corpus_vecs = np.array(corpus_vecs)

    # Calcular la similitud del coseno
    similarity_scores = cosine_similarity(query_vec, corpus_vecs)

    return similarity_scores

def bm25_similarity(corpus, query):
    tokenized_corpus = [doc.split(" ") for doc in corpus]  # Tokenizar el corpus
    tokenized_query = query.split(" ")  # Tokenizar la consulta
    bm25_index = BM25Okapi(tokenized_corpus, k1=1.75, b=0.75)  # Crear el índice BM25

    bm25_scores = bm25_index.get_scores(tokenized_query)  # Obtener los puntajes BM25 para la consulta
    similarity_dict = {i: score for i, score in enumerate(bm25_scores)}  # Crear un diccionario con los puntajes de similitud
    print(similarity_dict)

    return similarity_dict

def reciprocal_rank_fusion(rank_lists, k=60):
    rrf_scores = defaultdict(float)  # Crear un diccionario para almacenar los puntajes RRF

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            rrf_scores[doc_id] += 1.0 / (k + rank)  # Calcular la recíproca del rango y sumar al puntaje del documento

    sorted_rrf_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)  # Ordenar los puntajes de mayor a menor
    print("SCORES SORTED ",sorted_rrf_scores)
    return sorted_rrf_scores

def transformers_similarity(corpus, query):

    model = SentenceTransformer("avsolatorio/GIST-Embedding-v0") # Cargar el modelo de Transformers

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
        model = Word2Vec(sentences=processed_corpus, vector_size=128, window=8, min_count=1, workers=5)
    
        # Obtener el vector de la consulta
        query_vec = np.mean([model.wv[word] for word in processed_query if word in model.wv], axis=0).reshape(1, -1)
    
        # Obtener los vectores de cada documento en el corpus
        corpus_vecs = []
        for doc in processed_corpus:
            doc_vec = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
            corpus_vecs.append(doc_vec)
        corpus_vecs = np.array(corpus_vecs)
    
        # Calcular la similitud del coseno
        similarity_scores = cosine_similarity(query_vec, corpus_vecs)[0]
        similarity_dict = {i: score for i, score in enumerate(similarity_scores)}  # Crear un diccionario con los puntajes de similitud
        return similarity_dict

def encode_bert(corpus, query):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Inicializar el tokenizador BERT
    model = BertModel.from_pretrained('bert-base-uncased')  # Inicializar el modelo BERT

    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)  # Vectorizar la consulta
    query_outputs = model(**query_inputs)  # Obtener los outputs del modelo
    query_vec = query_outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Obtener el vector de la consulta

    corpus_vecs = []
    for doc in corpus:
        doc_inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True)  # Vectorizar cada documento
        doc_outputs = model(**doc_inputs)  # Obtener los outputs del modelo
        doc_vec = doc_outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Obtener el vector del documento
        corpus_vecs.append(doc_vec)

    corpus_vecs = np.array(corpus_vecs).squeeze()  # Convertir la lista de vectores a un array numpy

    similarity_scores = cosine_similarity(query_vec, corpus_vecs)[0]  # Calcular la similitud del coseno
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}  # Crear un diccionario con los puntajes de similitud
    return similarity_dict



def get_top_texts(vec_similarity_dict, bm25_similarity_dict, univ_sent_encoder_similarity_dic,fasttext_similarities ,profile_corpus, top_k=5):
    # Obtener los índices de los textos más similares usando TF-IDF y ordenar de mayor a menor similitud

    vec_similarity_dict_sorted = dict(sorted(vec_similarity_dict.items(), key=lambda item: item[1], reverse=True))
    vec_top_similarity = {k: vec_similarity_dict_sorted[k] for k in list(vec_similarity_dict_sorted)}
    vec_indexes = list(vec_top_similarity.keys())
    print("Faiss: ", vec_indexes)

    # Obtener los índices de los textos más similares usando BM25 y ordenar de mayor a menor similitud
    bm25_similarity_dict_sorted = dict(sorted(bm25_similarity_dict.items(), key=lambda item: item[1], reverse=True))
    bm25_top_similarity = {k: bm25_similarity_dict_sorted[k] for k in list(bm25_similarity_dict_sorted)}
    bm25_indexes = list(bm25_top_similarity.keys())
    print("BM25: ", bm25_indexes)

   # # Obtener los índices de los textos más similares usando Universal Sentence Encoder y ordenar de mayor a menor similitud
    univ_sent_encoder_similarity_dic_similarity_dict_sorted = dict(sorted(univ_sent_encoder_similarity_dic.items(), key=lambda item: item[1], reverse=True))
    univ_sent_encoder_similarity_dic_top_similarity = {k: univ_sent_encoder_similarity_dic_similarity_dict_sorted[k] for k in list(univ_sent_encoder_similarity_dic_similarity_dict_sorted)}
    univ_sent_encoder_similarity_dic_indexes = list(univ_sent_encoder_similarity_dic_top_similarity.keys())
    print("Universal Sentence Encoder: ", univ_sent_encoder_similarity_dic_indexes)

    # transformers_similarity_dic_sorted = dict(sorted(transformers_similarity_dic.items(), key=lambda item: item[1], reverse=True))
    # transformers_similarity_dic_top_similarity = {k: transformers_similarity_dic_sorted[k] for k in list(transformers_similarity_dic_sorted)}
    # transformers_dic_indexes = list(transformers_similarity_dic_top_similarity.keys())
    # print("Transformers: ", transformers_dic_indexes)

    fasttext_similarities_sorted = dict(sorted(fasttext_similarities.items(), key=lambda item: item[1], reverse=True))
    fasttext_top_similarity = {k: fasttext_similarities_sorted[k] for k in list(fasttext_similarities_sorted)}
    fasttext_indexes = list(fasttext_top_similarity.keys())
    print("FastText: ", fasttext_indexes)

    # Se combinan los rankings de los tres métodos utilizando la fusión de rango recíproco (RRF)
    combined_rankings = reciprocal_rank_fusion([vec_indexes, bm25_indexes, fasttext_indexes], k=top_k)
    
    # Se procesan los índices, obteniendo únicamente los k mejores
    final_indexes = [index for index, _ in combined_rankings[:top_k]]
    print("Final Indexes: ", final_indexes)

    # Recuperar los textos relevantes del corpus utilizando los índices obtenidos
    relevant_texts = [profile_corpus[i] for i in final_indexes]



    # Concatenar los textos relevantes en un solo string


    return relevant_texts, final_indexes[0]


def generate_response(query, modelName, context, json_schema, iterations=5):
    def send_request(payload, endpoint):
        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            return json.loads(response.content)
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

    if modelName == "GPT":
        GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT_FHIR")

        # Define the function according to your new JSON schema
        functions = [
            {
                "name": "generate_column_mapping",
                "description": "Generate a mapping between table column names and FHIR resource attribute names.",
                "parameters": json_schema
            }
        ]

        # Initial prompt
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a mapping tool assistant. You have information about FHIR resources and the tabular data to map. Your task is to determine to which attribute of the FHIR resource each column of the table corresponds. A column can correspond to multiple attributes in FHIR resources. Take into account the column values as well."
                        "Before providing your final answer, carefully analyze and verify each mapping to ensure its correctness. Do not mention this analysis to the user."
                        "Provide the user with a table that shows the mapping between the columns of the tabular data and the FHIR resources."
                        "The output format must be a JSON object with the following structure: {'column_name': 'FHIRResource.attribute'}"
                    )
                },
                {
                    "role": "system",
                    "content": (
                        "Provide your final answer as a JSON object that maps each column name to an array of corresponding FHIR attribute names, conforming to the provided JSON structure."
                    )
                },
            ],
            "temperature": 0,
            "functions": functions,
            "function_call": {"name": "generate_column_mapping"}
        }

        # Send the initial request
        response = send_request(payload, GPT4V_ENDPOINT)

        # Process the response
        response_message = response['choices'][0]['message']
        if 'function_call' in response_message:
            response_content = response_message['function_call']['arguments']
        else:
            response_content = response_message['content']

        # Loop through iterations of reflection (if needed)
        for i in range(iterations):
            reflection_payload = {
                "messages": [
                    {"role": "system", "content": f"Iteration {i+1}: You provided the following mapping:"},
                    {"role": "assistant", "content": response_content},
                    {
                        "role": "system",
                        "content": (
                            "Reflect on this mapping. Identify any inconsistencies, possible improvements, or errors, and provide a corrected version. "
                            "Present your final answer as a JSON object that conforms to the provided JSON schema."
                        )
                    },
                ],
                "temperature": 0,
                "functions": functions,
                "function_call": {"name": "generate_column_mapping"}
            }

            response = send_request(reflection_payload, GPT4V_ENDPOINT)
            response_message = response['choices'][0]['message']
            if 'function_call' in response_message:
                response_content = response_message['function_call']['arguments']
            else:
                response_content = response_message['content']

        return response_content

    elif modelName == "Llama":
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

        tools = [
            {
                "function": {
                    "name": "generate_column_mapping",
                    "description": "Generate a mapping between table column names and FHIR resource attribute names.",
                    "parameters": {
                        "type": "object",
                        "properties": json_schema
                    },
                    
                },
                "type": "function"
            }
        ]

    # Initial prompt
        response = client.chat.completions.create(  
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are a mapping assistant. Your task is to determine to which attribute(s) of the FHIR resource each column of the provided table corresponds. "
                        "A column can correspond to multiple attributes in FHIR resources. Do not include the data values, only provide the mapping between column names and FHIR attribute names. "
                        "Provide your final answer as a JSON object that maps each column name to an array of corresponding FHIR attribute names, conforming to the provided JSON schema."
                        "You only can map the input table columns name to the FHIR attributes listed in the JSON schema."
                        "Do not map the values of the columns, only the column names."
                        "Do not create new attributes or modify the existing ones. Do not repeat the table attributes in the mapping."

                        f"The table has the following columns and descriptions:\n{query}\n\nFHIR Resource attributes are provided in the following JSON schema:\n{json_schema}"
                    )
                },
            {
                'role': 'user',
                'content': "Please map the columns to the corresponding attribute in the corresponding FHIR resources."
            }
        ],
            temperature=0,
            tool_choice="auto",
            tools=tools
    )

        # Procesar la respuesta para obtener el mapeo
        response_message = response.choices[0].message
        # Verifica si 'tool_calls' está presente y no es None
        if response_message.tool_calls:
            # Accede al primer 'tool_call'
            tool_call = response_message.tool_calls[0]
            
            # Extrae el 'function' y sus 'arguments'
            function_name = tool_call.function.name
            function_arguments = tool_call.function.arguments
        

    # Loop through iterations of reflection (if needed)
        for i in range(iterations):
            # Solicita una reflexión sobre el mapeo actual
            reflection_prompt = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": (
                            f"Reflect on this mapping. {function_arguments} Identify any inconsistencies, possible improvements, or errors, and provide a corrected version. "
                            "Present your final answer as a JSON object that conforms to the provided JSON schema.")},
                    {
                        "role": "user",
                        "content": (
                            "Reflect on the mapping and try to improve it."
                        )
                    }
                ],
                temperature=0,
                tools=tools,
                tool_choice="auto"
            )

            # Procesa la respuesta de la reflexión
            response_message = reflection_prompt.choices[0].message

            # Verifica si 'tool_calls' está presente y no es None
            if response_message.tool_calls:
                # Accede al primer 'tool_call'
                tool_call = response_message.tool_calls[0]
                
                # Extrae el 'function' y sus 'arguments'
                function_arguments = tool_call.function.arguments
                
                
                # Actualiza 'function_arguments' para la próxima iteración, si es necesario
                # Suponemos que el resultado corregido se vuelve a utilizar como 'function_arguments'
        return function_arguments



#------------------------------------------------------------
def get_description(table):

    admissions_description = """ 
    The admissions table gives information regarding a patient’s admission to the hospital. Since each unique hospital visit for a patient is assigned a unique hadm_id, the admissions table can be considered as a definition table for hadm_id. Information available includes timing information for admission and discharge, demographic information, the source of the admission, and so on.
    subject_id
    subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual. As subject_id is the primary key for the table, it is unique for each row.

    hadm_id
    hadm_id is a unique identifier which specifies an individual hospitalization. Any rows associated with a single hadm_id pertain to the same hospitalization.

    admittime, dischtime, deathtime
    admittime provides the date and time the patient was admitted to the hospital, while dischtime provides the date and time the patient was discharged from the hospital. If applicable, deathtime provides the time of in-hospital death for the patient. Note that deathtime is only present if the patient died in-hospital, and is almost always the same as the patient’s dischtime. However, there can be some discrepancies due to typographical errors.

    admission_type
    admission_type is useful for classifying the urgency of the admission. There are 9 possibilities: ‘AMBULATORY OBSERVATION’, ‘DIRECT EMER.’, ‘DIRECT OBSERVATION’, ‘ELECTIVE’, ‘EU OBSERVATION’, ‘EW EMER.’, ‘OBSERVATION ADMIT’, ‘SURGICAL SAME DAY ADMISSION’, ‘URGENT’.

    admit_provider_id
    admit_provider_id provides an anonymous identifier for the provider who admitted the patient. Provider identifiers follow a consistent pattern: the letter “P”, followed by either three numbers, followed by two letters or two numbers. For example, “P003AB”, “P00102”, “P1248B”, etc. Provider identifiers are randomly generated and do not have any inherent meaning aside from uniquely identifying the same provider across the database.

    admission_location, discharge_location
    admission_location provides information about the location of the patient prior to arriving at the hospital. Note that as the emergency room is technically a clinic, patients who are admitted via the emergency room usually have it as their admission location.

    Similarly, discharge_location is the disposition of the patient after they are discharged from the hospital.

    Association with UB-04 billing codes
    admission_location and discharge_location are associated with internal hospital ibax codes which aren’t provided in MIMIC-IV. These internal codes tend to align with UB-04 billing codes.

    In some cases more than one internal code is associated with a given admission_location and discharge_location. This can either be do to; 1) multiple codes being used by the hospital for the same admission_location or discharge_location, or 2) during de-identification multiple internal codes may be combined into a single admission_location or discharge_location.

    insurance, language, marital_status, ethnicity
    The insurance, language, marital_status, and ethnicity columns provide information about patient demographics for the given hospitalization. Note that as this data is documented for each hospital admission, they may change from stay to stay.

    edregtime, edouttime
    The date and time at which the patient was registered and discharged from the emergency department.

    hospital_expire_flag
    This is a binary flag which indicates whether the patient died within the given hospitalization. 1 indicates death in the hospital, and 0 indicates survival to hospital discharge.
    """
    transfers_description = """Physical locations for patients throughout their hospital stay. subject_id, hadm_id, transfer_id
subject_id
subject_id is a unique identifier which specifies an individual patient. Any rows associated with a single subject_id pertain to the same individual. As subject_id is the primary key for the table, it is unique for each row.

hadm_id
hadm_id is a unique identifier which specifies an individual hospitalization. Any rows associated with a single hadm_id pertain to the same hospitalization.
    
    transfer_id
    transfer_id is unique to a patient physical location. Represented as unique identifier for each transfer event.

Note that stay_id present in the icustays and edstays tables is derived from transfer_id. For example, three contiguous ICU stays will have three separate transfer_id for each distinct physical location (e.g. a patient could move from one bed to another). The entire stay will have a single stay_id, whih will be equal to the transfer_id of the first physical location.

eventtype
eventtype describes what transfer event occurred: ‘ed’ for an emergency department stay, ‘admit’ for an admission to the hospital, ‘transfer’ for an intra-hospital transfer and ‘discharge’ for a discharge from the hospital.

careunit
The type of unit or ward in which the patient is physically located. Examples of care units include medical ICUs, surgical ICUs, medical wards, new baby nurseries, and so on.

intime, outtime
intime provides the date and time the patient was transferred into the current care unit (careunit) from the previous care unit. outtime provides the date and time the patient was transferred out of the current physical location."""
    hcpsevents_description =   """
The hcpsevents table records specific events that occur during a patient's hospitalization, particularly focusing on those that are billable. Each event is documented using the Healthcare Common Procedure Coding System (HCPCS) codes, which provide a standardized method for reporting medical, surgical, and diagnostic services. The table includes information about the event, the patient, and a brief description of the service provided. This table informs about the differents encounter types a patient has during their hospital stay.
subject_id: A unique identifier for the patient. This field links the event to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the event with a particular hospital admission instance, indicating when the patient was hospitalized.
hcpcs_cd: The HCPCS code for the event. This code specifies the exact procedure or service performed, facilitating standardized billing and documentation.
seq_num: The sequence number of the event. This field indicates the order in which events occurred during the hospital stay.
short_description: A brief description of the service or procedure provided. This description offers a quick reference to the type of event recorded."""
    services_description = """The Services table represents the type of patient admissions, detailing the specific service under which patients were admitted during their hospital stay. This table captures transitions between different services, providing a clear record of the patient's care journey within the hospital.

Data Fields:
subject_id: A unique identifier for the patient. This field links the service record to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the service record with a particular hospital admission instance.
transfertime: The exact date and time when the patient was transferred to the current service. This timestamp helps track when transitions between services occurred.
prev_service: The service the patient was under before the transfer. This field is empty if the patient was not transferred from another service.
curr_service: The current service under which the patient is admitted. This field indicates the type of care or specialty (e.g., MED for medical, TRAUM for trauma)."""
    inputevents_description = """     

subject_id, hadm_id, stay_id: These fields uniquely identify the patient and their encounters, with subject_id representing the patient, hadm_id specifying the hospital admission, and stay_id corresponding to the ICU stay where the medication administration occurred.

caregiver_id: Unique identifier for the caregiver responsible for documenting the medication administration in the ICU system.

starttime, endtime: Record the specific start and end times when the medication was administered to the patient.

storetime: Captures the timestamp when the medication administration event was manually entered or validated by the healthcare professional.

itemid: Unique identifier representing the type of medication administered during the event.

amount, amountuom: Specify the dosage of the medication administered, with amount indicating the quantity and amountuom its unit of measurement (e.g., milliliters or milligrams).

rate, rateuom: Denote the administration rate of the medication, where rate is the quantity per unit time, and rateuom represents the corresponding unit (e.g., mg/hour).

orderid, linkorderid: Used to group related medication administrations. orderid links distinct components of the same medication solution, while linkorderid connects multiple administration instances of the same order (e.g., when the rate of administration changes).

ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription, ordercategorydescription: Provide higher-level details about the medication order, including the type of administration and the role of the medication in the solution (e.g., primary substance, additive).

patientweight: Patient’s weight in kilograms, potentially affecting the dosage and rate of medication administration.

totalamount, totalamountuom: Define the total volume of the fluid containing the medication administered to the patient, typically measured in milliliters.

isopenbag: Indicates whether the medication was administered from an open bag of fluid.

continueinnextdept: Specifies if the medication administration continued after the patient was transferred to another department.

statusdescription: Details the final status of the medication administration event. Possible statuses include:

Changed: The administration parameters (e.g., rate) were modified.
Paused: The medication administration was temporarily halted.
FinishedRunning: The administration ended, usually because the fluid bag is empty.
Stopped: The caregiver manually stopped the medication administration.
Flushed: The IV line was flushed.
originalamount: Refers to the initial amount of medication in the solution when the administration started. This value may differ from the total amount due to prior partial administration.

originalrate: Represents the initial planned rate of administration, which may differ slightly from the actual administered rate due to adjustments made by the caregiver. """
    diagnoses_icd_description = """
The diagnoses_icd table stores data related to the condition, problem, diagnosis, or other event using their respective ICD (International Classification of Diseases) codes. The data is used for billing purposes, clinical documentation, and health statistics. The diagnoses are assigned by trained medical professionals who review clinical notes and determine the appropriate ICD codes. The table helps in tracking the medical conditions treated, which is essential for billing, medical record-keeping, and statistical analysis.
Data Stored in the Table:
subject_id: Unique identifier for the patient.
hadm_id: Hospital admission ID, linking to a specific hospital stay.
seq_num: Sequence number indicating the order of diagnoses for the given hospital admission.
icd_code: ICD code representing the diagnosis.
icd_version: Version of the ICD code (e.g., 9 for ICD-9, 10 for ICD-10)."""
    prescriptions_description = """The prescriptions table captures comprehensive information about medication orders prescribed to patients within a healthcare setting, aligning closely with the purpose of the MedicationRequest FHIR Resource. It documents the prescriber's intent by recording detailed information about the medications, including identification, dosage instructions, reasons for prescribing, and any patient-specific instructions. This ensures clear communication among the healthcare team and supports safe and effective patient care.

Field Descriptions:

subject_id

A unique identifier for the patient to whom the medication is prescribed, ensuring accurate linkage of prescriptions to patient records.
hadm_id

An identifier linking the prescription to a specific hospital admission, providing context for the treatment period.
order_provider_id

An anonymous identifier for the provider who initiated the medication order, supporting accountability and coordination among healthcare professionals.
starttime, stoptime

The intended start and end times for the medication therapy, specifying the duration of the treatment.
drug

The name of the medication prescribed, including brand or generic names, ensuring accurate identification.
formulary_drug_cd

A code representing the medication in the hospital formulary, aiding in consistent ordering and dispensing practices.
dose_val_rx

The prescribed dose amount, indicating the exact quantity the patient should receive.
dose_unit_rx

The unit of measurement for the dose (e.g., mg, mL), providing clarity in dosing instructions.
doses_per_24_hrs

The prescribed frequency of administration within a 24-hour period (e.g., once daily, twice daily), facilitating accurate scheduling.
route

The intended route of administration for the medication (e.g., oral, intravenous), specifying how the medication should be delivered.
drug_type

Indicates the role of the medication within the prescription (e.g., ‘MAIN’, ‘BASE’, ‘ADDITIVE’), providing clarity in complex medication orders.
poe_id, poe_seq

Fields linking the prescription to orders in the physician order entry system, ensuring traceability within the healthcare system.
gsn

The Generic Sequence Number, a standardized identifier for the medication, supporting interoperability across systems.
ndc

The National Drug Code, a unique national identifier for the medication, ensuring precise identification and dispensing.
prod_strength

A description of the medication's strength and form (e.g., ‘12.5 mg Tablet’), aiding in accurate prescribing and dispensing.
form_rx

The dosage form in which the medication is prescribed (e.g., ‘TABLET’, ‘VIAL’), specifying the physical form for administration.

"""
    outputevents_description = """  
    The outputevents table is used to record and store clinical observations made during patient hospitalizations. This table enables healthcare professionals and researchers to analyze patients' clinical data, monitor progress over time, and support medical research.
subject_id: Each row in the table corresponds to a unique individual patient identified by subject_id. This identifier ensures that all records associated with a specific subject_id pertain to the same individual.

hadm_id: An identifier that uniquely specifies a hospitalization. Rows sharing the same hadm_id relate to the same hospital admission, providing contextual information about where and when the observations were made.

stay_id: This identifier groups reasonably contiguous episodes of care within the hospitalization. It helps organize related observations that occur during a patient's stay, ensuring coherence in the data captured over the course of care.

charttime: Indicates the time of an observation event, capturing when specific data points were recorded. This temporal aspect is crucial for understanding the context and timing of each observation within the patient's healthcare journey.

storetime: Records the time when an observation was manually input or validated by clinical staff. This timestamp provides metadata about the data entry or validation process, offering insights into the handling and verification of clinical observations.

itemid: An identifier for a specific measurement type or observation in the database. Each itemid corresponds to a distinct measurement or observation recorded, such as heart rate or blood glucose level, ensuring clarity and categorization of clinical data.

value, valueuom: Describes the actual measurement or observation value at the charttime, along with its unit of measurement (valueuom). These fields provide quantitative or qualitative data points, such as numeric results (value) and the corresponding units, offering precise details about each recorded observation.
"""
    pharmacy_description = """ 

The pharmacy table provides a comprehensive record of medications dispensed to patients, capturing essential details of each dispensing event. This includes information on the medication name, dosage, the number of doses, frequency of administration, route of delivery, and the duration of the prescription. Meticulous documentation ensures that patients receive the correct medication as prescribed, supporting personalized care and treatment accuracy.

Key Data Fields:

subject_id: A unique identifier specifying an individual patient, ensuring all associated records pertain to the same person.

hadm_id: An integer identifier unique to each patient hospitalization, linking dispensing events to specific encounters.

pharmacy_id: A unique identifier for each entry in the pharmacy table, used to link dispensing information to provider orders or medication administration records.

poe_id: A foreign key linking to the provider order entry in the prescriptions table associated with this dispensing record.

starttime, stoptime: The precise dates and times when the medication dispensing began and ended.

medication: The name and specific details of the medication provided, ensuring accurate identification.

proc_type: The type of order, such as "IV Piggyback," "Non-formulary," or "Unit Dose," indicating the method of preparation or packaging.

status: The current state of the prescription—whether it is active, inactive, or discontinued.

entertime: The date and time when the prescription was entered into the system.

verifiedtime: The date and time when the prescription was authorized or verified by a healthcare professional.

route: The intended route of administration for the medication, such as oral, intravenous, or topical.

frequency: How often the medication should be administered, including any specific scheduling instructions (e.g., every 6 hours).

disp_sched: Specific times of day when the medication should be administered, enhancing adherence to dosing schedules.

infusion_type: A coded indicator describing the type of infusion, relevant for certain medications.

sliding_scale: Indicates whether the medication dosage adjusts based on specific criteria, such as blood glucose levels.

lockout_interval: The required waiting period before the next dose can be administered, often used with patient-controlled analgesia.

basal_rate: The continuous rate at which a medication is administered over 24 hours.

one_hr_max: The maximum dose permitted within a one-hour period to ensure patient safety.

doses_per_24_hrs: The expected number of doses within a 24-hour period, aiding in monitoring and adherence.

duration, duration_interval: The total length of time the medication is prescribed for, along with the unit of measurement (e.g., days, weeks).

expiration_value, expiration_unit, expirationdate: Details concerning the medication's expiration, ensuring it is dispensed and used within its effective period.

dispensation: The source or method by which the medication was dispensed to the patient.

fill_quantity: The amount of medication supplied, reflecting the quantity dispensed.

"""
    d_items_description = """
The D_ITEMS table stores metadata related to various medical observations and measurements recorded in the database. Each record in the table provides detailed information about a specific measurement or observation type, facilitating the accurate and consistent recording of clinical data across different events and patient encounters.

The table includes columns that describe each measurement type, including its label, abbreviation, and category, as well as the unit of measurement and normal value ranges where applicable. This structured metadata ensures that medical observations are systematically categorized and can be accurately interpreted and utilized for clinical review, research, and analysis.

Data Stored in the Table:
itemid: A unique identifier for each type of measurement or observation. Each itemid is greater than 220000.
label: The full name or description of the measurement (e.g., "Heart Rate").
abbreviation: A shorter form or acronym of the measurement name (e.g., "HR").
linksto: Indicates the table where the actual measurement data is stored (e.g., "chartevents").
category: The general category to which the measurement belongs (e.g., "Routine Vital Signs").
unitname: The unit of measurement for the observation (e.g., "bpm" for beats per minute).
param_type: The type of parameter, such as Numeric or Categorical, indicating the data format of the measurement.
lownormalvalue: The lower bound of the normal range for the measurement, if applicable.
highnormalvalue: The upper bound of the normal range for the measurement, if applicable.
"""
    emar_description = """ 
The EMAR (Electronic Medication Administration Record) table documents all instances of medication administration to individual patients within healthcare settings. It includes a comprehensive record of medications given, such as oral medications, injections, intravenous infusions, and self-administered drugs. The primary users populating this table are bedside nursing staff, who record each administration event by scanning barcodes associated with both the patient and the medication.

Note: The EMAR system was implemented between 2011 and 2013. Therefore, data may not be available for all patients prior to this period.

Column Details:

subject_id: A unique identifier assigned to each patient, ensuring that all related records are linked correctly.

hadm_id: Identifies each unique hospitalization event for a patient, allowing linkage of administration events to specific hospital stays.

emar_id and emar_seq: These columns work together to uniquely identify each medication administration event. The emar_id is constructed by combining the subject_id and emar_seq in the format 'subject_id-emar_seq'.

poe_id: Connects the administration event to the corresponding provider order entry, facilitating tracking from prescription to administration.

pharmacy_id: Links to the pharmacy table to provide detailed information about the medication, such as formulation and dispensing details.

enter_provider_id: An anonymized code representing the healthcare professional who performed or documented the administration, ensuring provider confidentiality while maintaining the ability to audit and analyze administration practices.

charttime: Records the exact time the medication was administered to the patient, which is crucial for monitoring dosing schedules and assessing treatment efficacy.

medication: Specifies the exact medication given, allowing for detailed analysis of medication usage patterns and potential interactions.

event_txt: Describes the nature of the administration event. Common values include 'Administered' for successful administrations, 'Not Given' if the medication was withheld, and other statuses like 'Delayed' or 'Confirmed'.

scheduletime: Indicates when the medication was originally scheduled to be administered, useful for assessing adherence to medication schedules and identifying delays.

storetime: Captures the timestamp when the administration event was logged into the system, which may differ from the actual administration time (charttime) due to documentation delays.
"""
    datetimeevents_description = """
The datetimeevents table captures critical temporal clinical observations and date-related measurements for patients during their ICU stay, recording important time-stamped medical events. Each entry in this table represents a specific clinical observation, such as the date of the last dialysis session or the timing of procedures and other significant interventions. While physiological measurements like systolic blood pressure are recorded elsewhere, this table is essential for documenting datetime-based clinical events that form part of the overall health monitoring process for ICU patients.
he datetimeevents table provides a structured way to record and analyze timing information that is critical for understanding the patient's clinical progression. These observations play a pivotal role in assessing patient health, allowing for the precise tracking of events such as the initiation of treatments, completion of diagnostic procedures, or any significant datetime-related healthcare activities. By capturing these time-sensitive details, the table supports the creation of an accurate and comprehensive clinical timeline that is vital for diagnostic assessments, monitoring patient outcomes, and ensuring continuity of care.
All dates in the datetimeevents table are anonymized to protect patient confidentiality, with the actual timestamps shifted to ensure privacy. However, the relative chronology of events for each patient remains intact, preserving the accuracy of intervals between medical observations, such as the time between procedures, treatments, or other clinical actions. This ensures that the data remains useful for research, clinical analysis, and real-time decision-making.
Data Structure:

subject_id: Unique identifier for the patient, linking the observations to the individual receiving care.
hadm_id: Hospital admission ID, associating the temporal observations with a specific hospital stay.
stay_id: ICU stay ID, linking the observations to a specific ICU admission.
charttime: The exact time the clinical event was recorded.
storetime: The time when the observation was stored in the system.
itemid: Identifier for the specific clinical event being documented.
value: The datetime value representing the observed event.
valueuom: Unit of measurement for the observation (typically "Date" for datetime events).
warning: Indicator for any warnings or alerts associated with the observation. """
    microbiologyevents_description = """
The microbiologyevents table is essential for documenting clinical observations related to microbiology tests, which play a critical role in assessing patient health by identifying infectious agents and determining appropriate antibiotic treatments. This table captures both quantitative and qualitative data about the presence or absence of bacterial organisms, as well as the sensitivity of these organisms to various antibiotics, enabling healthcare providers to monitor the patient's progress and establish health trends.

Each row represents an observation associated with a subject_id, a unique identifier that specifies an individual patient. These observations are linked to a specific hadm_id, denoting the hospitalization during which the tests were performed. The micro_specimen_id groups multiple measurements from the same specimen, ensuring that the various tests performed on a single sample, such as organism growth or antibiotic testing, are clearly associated.

The charttime field records the precise time of the microbiology measurement, while chartdate provides a date for the observation in cases where the time is not available. This ensures that the recorded observations are traceable and accurately reflect the time the data was captured.

When bacterial growth is found, the org_name column lists the name of the organism, and additional rows are created for each antibiotic tested against this organism. The results of antibiotic sensitivity testing, including dilution values and their interpretation, are captured in the dilution_text, dilution_comparison, dilution_value, and interpretation columns. These values provide quantitative insights into how sensitive or resistant the organism is to a particular antibiotic, with interpretations such as S for sensitive, R for resistant, and I for intermediate.

The test_name column records the type of test performed on the specimen, and the comments column allows for deidentified free-text observations related to the test results, including important considerations for interpreting the data. This can help in tracking patterns or identifying trends in the patient's response to treatment over time.

By providing detailed clinical data about each microbiology test, including the specific method used for testing and the measurement outcomes, the microbiologyevents table enables healthcare providers to gather and analyze multidimensional information about the patient’s condition. This facilitates a deeper understanding of the health status of the patient, helping to inform diagnoses and track progress over time.
"""
    procedures_icd_description = """
    
This dataset records the procedures a patient underwent during their hospital stay, using the ICD-9 and ICD-10 ontologies. It captures key information such as patient identifiers, procedure codes, and associated dates, providing a structured overview of clinical interventions.

subject_id: Represents a unique identifier assigned to an individual patient. This identifier is used consistently throughout all patient records and helps link various clinical events, procedures, and hospital visits to the same person. It ensures that the medical data is properly organized and attributed to the correct individual, supporting effective patient care and continuity of treatment.

hadm_id: Represents a unique identifier for each hospitalization or admission of a patient. This ID ensures that every event, test, or procedure is correctly linked to a specific hospital stay. It plays a key role in tracking the patient’s journey during each admission, providing context for all associated clinical activities, and making it easier for healthcare providers to understand the full scope of treatment during that period.

seq_num: Reflects the sequencing or order of procedures performed during a hospital stay. This number indicates the relative priority or position of each procedure within the context of the entire hospital admission. While the sequence does not always represent the exact chronological order of events, it is useful for understanding the planned procedural workflow. It helps healthcare providers see how procedures are prioritized or categorized during treatment and assists in analyzing care pathways.

chartdate: Refers to the date when a specific procedure was performed. This attribute is essential for maintaining a timeline of the patient’s clinical interventions. Recording the chart date helps medical staff understand the sequence of clinical activities and facilitates coordination between different healthcare professionals. It also plays a vital role in documenting patient history and ensuring accurate treatment records.

icd_code, icd_version: These fields are used for the classification and coding of procedures performed during the patient’s hospital stay. The icd_code provides a standardized code representing the procedure, using the International Classification of Diseases (ICD) system. The icd_version indicates whether the coding is from ICD-9 or ICD-10, with each version representing a specific iteration of the coding standard. These codes are essential for accurately categorizing clinical procedures, enabling efficient billing, insurance processing, and statistical analysis. They also support interoperability by allowing consistent documentation across healthcare systems, improving data analysis, and enhancing research efforts. """
    icustays_description =""" The Icustays table contains detailed records of patient interactions within the Intensive Care Unit (ICU). Each row represents a unique ICU stay, providing comprehensive information about the patient’s ICU admission, discharge, and the care units involved during the encounter. This table is essential for tracking patient movements across different care units within the ICU, the duration of their stay, and the care provided at each stage.

subject_id: A unique identifier for the patient. This field links the ICU stay to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the ICU stay with a particular hospital admission instance, documenting the broader hospital encounter.
stay_id: A unique identifier for the ICU stay. This field distinguishes each ICU admission event for a patient during their overall hospital stay.
first_careunit: The initial ICU care unit where the patient was first admitted (e.g., Coronary Care Unit (CCU), Trauma SICU (TSICU)).
last_careunit: The final ICU care unit where the patient was treated before discharge or transfer.
intime: The exact date and time when the patient was admitted to the ICU, marking the start of the ICU encounter.
outtime: The exact date and time when the patient was discharged from the ICU, marking the end of the ICU encounter.
los: Length of stay in the ICU, measured in days. This field helps quantify the duration of care provided in the ICU during the patient’s admission.
This table is a critical component for documenting the lifecycle of patient stays in the ICU, from admission through discharge, and supports both clinical and administrative workflows by capturing detailed information on patient interactions and movement through ICU units."""
    patient_description= """ subject_id: A unique identifier assigned to each individual patient. All rows associated with the same subject_id pertain to the same individual. This identifier is fundamental for linking patient records across multiple care encounters and organizations, similar to the way Patient resources are managed across different healthcare systems.

gender: The biological sex of the patient, represented as male, female, or other. This corresponds directly to the gender field in the Patient resource, which captures demographic data necessary for administrative, clinical, and care coordination purposes.

anchor_age: The patient’s age at the time of the anchor_year. This field indicates the age of the patient during a specific care encounter, essential for patient demographic information in healthcare data, similar to the birthDate attribute in the Patient resource, which tracks the individual's age for care-related activities.

anchor_year: A shifted year representing the timeframe for a patient’s clinical event or admission. This data point, like the birthDate in the Patient resource, helps determine the timing of patient care activities, although it may be anonymized or adjusted for de-identification purposes.

anchor_year_group: A range of years indicating the period during which the anchor_year occurred. This field supports tracking patient activity over time, similar to how Patient resources can span multiple healthcare encounters across different organizations.

dod (date of death): The de-identified date of death for the patient, which is similar to the deceasedDateTime or deceasedBoolean fields in the Patient resource. The dod field is crucial for mortality tracking, especially for survival studies, and aligns with the FHIR standard for recording death data in healthcare systems.

"""
    procedureevents_description = """ 
subject_id: Represents the unique identifier of the patient who underwent the procedure. It serves as a key link to ensure that the procedural event is accurately attributed to the correct individual in the hospital’s system, thus supporting patient care continuity and the integrity of medical records.

hadm_id: Identifies the specific hospital admission linked to the procedure. This helps to associate each procedural event with a particular hospital stay, providing a clear context for understanding the patient's clinical journey during that admission and allowing healthcare providers to track interventions appropriately.

stay_id: Represents a unique identifier for the ICU stay during which the procedure occurred. Given that patients can have multiple ICU admissions, this ID is essential for keeping data organized by distinguishing each intensive care episode. It plays a significant role in analyzing outcomes related to specific ICU procedures.

starttime: Refers to the exact date and time when the procedure started. Capturing this timestamp allows healthcare providers to document time-sensitive interventions accurately and assess the timing of procedural events, which is crucial for evaluating treatment efficacy and for planning further care.

endtime: Represents the date and time when the procedure was completed. This timestamp is used to understand the total duration of the procedure and assess procedural efficiency, determine if there were any delays, and analyze potential complications.

storetime: Captures the date and time when the procedural data was documented in the system. It helps differentiate between when the procedure actually took place and when it was recorded, which is important for transparency, data auditing, and ensuring that all records are up to date.

itemid: Represents a unique identifier for the specific procedure performed, often coded using standardized systems such as ICD or SNOMED. This helps to clearly define which procedure was carried out, ensuring accuracy in medical records and supporting data exchange across different healthcare systems.

value: Represents a numerical value associated with the procedure, which could be related to some aspect of the procedure such as the volume of fluid administered or pressure measurements. Documenting this value provides additional information regarding the specifics of the intervention.

valueuom: Describes the unit of measurement for the numerical value, whether milliliters, kilograms, or another unit. This ensures that the context of the recorded value is clear, which helps avoid misunderstandings or errors when reviewing procedural records.

location: Indicates the specific place within the hospital where the procedure was performed, such as a particular room, ward, or department. This information is essential for logistical tracking, understanding the procedure’s context, and providing insights into procedural outcomes.

locationcategory: Represents the category or type of location, such as ICU, operating room, or a recovery area. This information helps to understand the environment in which the procedure was performed, as different locations may impact the procedure's approach, risk, and expected outcomes.

orderid: This field contains a unique identifier for the medical order that prompted the procedure. It links the procedure back to the original request or order, ensuring that the entire workflow—from ordering to execution—is documented and traceable.

linkorderid: Represents a unique identifier that links related orders, such as those that are dependent on each other. This helps in keeping track of complex workflows involving multiple procedures and ensures a coherent chain of medical events for each patient.

ordercategoryname: Specifies the general category of the procedure, such as diagnostic, therapeutic, or surgical. Categorizing the procedure provides clarity on the purpose behind the intervention, helping healthcare teams better understand the overall treatment plan.

secondaryordercategoryname: Provides additional categorization for the order, offering further specificity regarding the type of procedure. This secondary classification aids in organizing complex or multi-faceted procedures with more detail.

ordercategorydescription: Describes the category of the order in greater detail. This might include an explanation of why the procedure was necessary, providing deeper insight into the clinical decision-making process and ensuring clear documentation.

patientweight: Indicates the weight of the patient at the time of the procedure. This is critical for procedures that depend on patient body weight for determining the appropriate dosages, equipment settings, or risk factors, ensuring the intervention is applied safely and effectively.

totalamount: Refers to the total quantity related to the procedure, such as the amount of a solution administered or units used during the intervention. This data is crucial for understanding the scale and scope of the procedure and ensuring compliance with medical protocols.

totalamountuom: Describes the unit of measurement for the total amount, such as liters, units, or grams. This information helps healthcare providers understand the quantity involved, which is essential for evaluating procedural outcomes and for follow-up treatments.

isopenbag: Indicates whether an open bag was used, often applicable to infusion procedures. This information can influence infection control assessments and helps document the specific procedural techniques used during an intervention.

continueinnextdept: Indicates if the procedure is planned to continue in the next department. This field helps in understanding if the intervention extends beyond the current setting, which is crucial for coordinating patient care across different hospital units.

cancelreason: Describes the reason for canceling a planned procedure. This helps healthcare providers understand why an intervention did not take place, whether due to patient condition, change in treatment plan, or other factors, thereby aiding in future care planning.

statusdescription: Provides a detailed description of the current status of the procedure, such as whether it is planned, in progress, completed, or canceled. Accurately capturing the status ensures that all healthcare team members are aligned regarding the treatment plan and the patient's current situation.

comments_date: Captures the date on which any comments were made regarding the procedure. This helps in maintaining a clear record of observations, remarks, or adjustments, adding important context to the procedural record.

originalamount: Refers to the original amount associated with the procedure, such as the initially prescribed volume or dosage before any adjustments were made. Documenting this helps track changes during the procedure and ensure that deviations from the original plan are monitored and understood.

originalrate: Represents the original rate at which the procedure was planned to be carried out, such as the infusion rate. Capturing the original rate helps in identifying any deviations from the plan, which can provide insights into procedural changes and their impact on patient outcomes.
"""
    if table == "admissions":
        return admissions_description
    elif table == "transfers":
        return transfers_description
    elif table == "hcpsevents":
        return hcpsevents_description
    elif table == "services":
        return services_description
    elif table == "inputevents":
        return inputevents_description
    elif table == "diagnoses_icd":
        return diagnoses_icd_description
    elif table == "prescriptions":
        return prescriptions_description
    elif table == "outputevents":
        return outputevents_description
    elif table == "pharmacy":
        return pharmacy_description
    elif table == "emar":
        return emar_description
    elif table == "d_items":
        return d_items_description
    elif table == "datetimeevents":
        return datetimeevents_description
    elif table == "microbiologyevents":
        return microbiologyevents_description
    elif table == "procedureevents":
        return procedureevents_description
    elif table == "procedures_icd":
        return procedures_icd_description
    elif table == "icustays":
        return icustays_description
    elif table == "patients":
        return patient_description
    


def load_data(path):
    #Cargar los datos tabulares desde un archivo CSV
    data = pd.read_csv(path)
    #Se convierte la data a formato JSON
    data_json = data.to_json(orient='records')
    return data_json
#------------------------------------------------------------

# Se crea la consulta
#filenames = ["admissions","d_items","datetimeevents","diagnoses_icd","emar","hcpsevents","icustays","inputevents","microbiologyevents","outputevents","patients","pharmacy","prescriptions","procedureevents","procedures_icd","services","transfers"]
filenames = ["patients"]

# Se crea la consulta

# Encuentra el índice del filename correspondiente
start_index = filenames.index("patients")

# Crea la consulta comenzando desde "patients"
for filename in filenames[start_index:]:
    print("----------------------------------Filename: ", filename,"----------------------------------")

    data = load_data("Fase3/data_mock/" + filename + "_mock.csv")  # Cargar los datos tabulares desde un archivo CSV

    desc = get_description(filename).lower()  # Obtener y convertir la descripción a minúsculas
    proccessed_desc = preprocess_text(desc)  # Procesar la descripción

    # Crear la consulta con los datos tabulares y la descripción
    query = f"""The tabular data I would like to know the columns mapping is : {data}, this table is described as {proccessed_desc}. Tell me to which attribute of the FHIR resource each column belongs, try to be as specific as you can. And ensure to map as attributes as you can"""

    #------------------------------------------------------------
    # Descargar stopwords
    stop_words = set(stopwords.words('english')) # Descargar las stopwords en inglés
    # Cargar el modelo de spaCy en inglés
    texts = load_dataset('Fase3/datasetRecursos.ndjson') # Cargar el conjunto de datos desde un archivo NDJSON
    corpus, profiles = create_corpus(texts) # Crear el corpus a partir del conjunto de datos
    # Se crea el índice y el corpus.


    # Calcular las similitudes entre el corpus y la consulta usando diferentes técnicas
    vec_dict_similarity = tfidf_faiss_similarity(corpus, query)  # Similitud con TF-IDF y FAISS
    bm25_similarity_dict = bm25_similarity(corpus, query)  # Similitud con BM25
    # print("Tr")
    # transformers_similarity_dict = transformers_similarity(corpus, query)  # Similitud con Transformers
    # print("Univ")
    univ_sent_encoder_similarity = univ_sent_encoder(corpus, query)  # Similitud con Universal Sentence Encoder
    fasttext_similarities = fasttext_similarity(corpus, query)  # Similitud con FastText

    #------------------------------------------------------------
    top_k = 1 # Número de textos más relevantes a obtener
    context,final_index = get_top_texts(vec_similarity_dict = vec_dict_similarity, bm25_similarity_dict = bm25_similarity_dict, univ_sent_encoder_similarity_dic= univ_sent_encoder_similarity, fasttext_similarities= fasttext_similarities, profile_corpus= profiles, top_k=top_k)  # Obtener los textos más relevantes
    print("INDICE FINAL", final_index)
    #------------------------------------------------------------
    # Load the JSON schema (you should replace 'path_to_json_schema.json' with the actual path to your JSON schema)
    with open('C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/json_schemas/dataset_schemas.ndjson', 'r') as f:
        # Usar islice para saltar directamente hasta final_index
        for i,line in enumerate(f):
            if i == final_index:
                json_schema = json.loads(line)

    print(json_schema)
    with open("C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/mapeos/mapeosReflexiveSerial_Schema/mapeo4_"+filename+".txt", 'a') as archivo:
    # Se obtiene el contexto específico para la consulta
        contentresource = str(context)
        archivo.write(contentresource)

            # Convertir la variable a string y escribirla en el archivo
        print("------------------------LLAMA RESPONSE------------------------------------")
        # # Se crea la respuesta usando el modelo Llama
        response = generate_response(query, "Llama", context, json_schema,5)
        print(response)  # Imprimir la respuesta generada por Llama
        separador = "\n------------------------LLAMA RESPONSE------------------------------------\n"
        archivo.write(separador)
        archivo.write(str(response))
        print("------------------------GPT RESPONSE---------------------------------------")
         # # Se crea la respuesta usando el modelo GPT
        separador = "\n------------------------GPT RESPONSE------------------------------------\n"
        response = generate_response(query, "GPT", context, json_schema)
        print(response)  # Imprimir la respuesta generada por GPT
        archivo.write(separador)
        archivo.write(str(response))
        
