# Importar las librerías necesarias
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import spacy
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
    model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4)

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
        model = Word2Vec(sentences=processed_corpus, vector_size=128, window=8, min_count=1, workers=4)
    
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



def get_top_texts(vec_similarity_dict, bm25_similarity_dict, univ_sent_encoder_similarity_dic, fasttext_similarities ,profile_corpus, top_k=5):
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
    combined_rankings = reciprocal_rank_fusion([vec_indexes, bm25_indexes, fasttext_indexes, univ_sent_encoder_similarity_dic_indexes], k=top_k)
    
    # Se procesan los índices, obteniendo únicamente los k mejores
    final_indexes = [index for index, _ in combined_rankings[:top_k]]
    print("Final Indexes: ", final_indexes)

    # Recuperar los textos relevantes del corpus utilizando los índices obtenidos
    relevant_texts = [profile_corpus[i] for i in final_indexes]



    # Concatenar los textos relevantes en un solo string


    return relevant_texts, final_indexes[0]


def generate_response(query, modelName, context, iterations=5):
    import time
    def send_request(payload, endpoint, retries=3, delay=5):
        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }
        for attempt in range(retries):
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return json.loads(response.content)
            except requests.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    raise SystemExit(f"Failed to make the request after {retries} attempts. Error: {e}")

    if modelName == "GPT":
        GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT_FHIR")
        # Initial prompt
        payload = {
            "messages": [
            {'role': 'system', 'content': "You are a mapping tool assistant. You have information about FHIR resources and the tabular data to map. Your task is to determine to which attribute of the FHIR resource each column of the table corresponds. A column can correspond to multiple attributes in FHIR resources. Take into account the column values as well."},
            {'role': 'system', 'content': "Before providing your final answer, carefully analyze and verify each mapping to ensure its correctness. Do not mention this analysis to the user."},
            {'role': 'system', 'content': "Provide the user with a table that shows the mapping between the columns of the tabular data and the FHIR resources."},
            {'role': 'system', 'content': "The output format must be a JSON object with the following structure: {'column_name': 'FHIRResource.attribute'}"},
            {'role': 'system', 'content': f"Profile information: {context}"},
            {'role': 'user', 'content': f"{query}"}],
            "temperature": 0,

        }

        # Send the initial request
        response = send_request(payload, GPT4V_ENDPOINT)

        # Process the response
        response_message = response['choices'][0]['message']
        if 'function_call' in response_message:
            response_content = response_message['function_call']['arguments']
        else:
            response_content = response_message['content']

        return response_content

    elif modelName == "Llama":
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        # Solicitud inicial
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
            {'role': 'system', 'content': (
            "You are a mapping tool assistant. You have information about FHIR resources and the tabular data to map. Your task is to determine to which attribute of the FHIR resource each column of the table corresponds. A column can correspond to multiple attributes in FHIR resources. Take into account the column values as well."
            "Before providing your final answer, carefully analyze and verify each mapping to ensure its correctness. Do not mention this analysis to the user."
            "Provide the user with a table that shows the mapping between the columns of the tabular data and the FHIR resources."
            "The output format must be a JSON object with the following structure: {'column_name': 'FHIRResource.attribute'}"
            f"The context information is : {context} And the query is: {query}")},
            {'role': 'user', 'content': "Please map the columns to the corresponding attribute in the corresponding FHIR resources."}],
            temperature=0,
        )
        # Procesar la respuesta para obtener el mapeo
        response_content = response.choices[0].message.content
        
        return response_content



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
The EMAR (Electronic Medication Administration Record) table is used to record the administration of medications to an individual patient. This table plays a crucial role in tracking medication administration, ensuring that every medication given to a patient is accurately documented, reflecting details such as the route of administration, the time, and the status of the administration. Records in this table are populated by bedside nursing staff using barcode scanning to link the medication and the patient, ensuring safe and accurate medication delivery.

Each row in the table is linked to a specific subject_id, a unique identifier representing the patient, ensuring all administration events related to that individual are associated. The hadm_id uniquely identifies each hospitalization for the patient, and fields like emar_id and emar_seq track administration orders in chronological order.

The medication field records the name of the medication that was administered, reflecting the specific medication given to the patient. The event_txt field captures additional details about the administration, with common values such as ‘Administered,’ ‘Applied,’ or ‘Not Given,’ which could correspond to reasons for a medication not being administered.

The table also contains identifiers such as poe_id and pharmacy_id, which allow linking to orders and prescriptions in the poe and pharmacy tables, ensuring that the entire medication administration process is documented from ordering to administration. Additionally, enter_provider_id uniquely identifies the healthcare provider who entered the data into the EMAR system, ensuring accountability in the administration process.

The charttime field records the exact time of the medication administration, while scheduletime reflects when the administration was originally scheduled, supporting accurate tracking of the occurence and adherence to prescribed schedules. The storetime field documents when the administration event was recorded in the EMAR system, allowing for complete visibility of the administration process.

In addition, the table supports tracking the dose of the medication administered and other relevant details such as the route (e.g., oral, intravenous). This ensures that the correct amount and method of administration are clearly documented, reflecting the performer (e.g., nurse or another healthcare provider) responsible for the administration.

By capturing detailed information about the medication administration process, the EMAR table plays a vital role in ensuring that medications are administered safely and effectively, with accurate documentation of the status, reason, time, and method of administration, thus supporting the overall medication therapy of the patient.
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

subject_id: A unique identifier representing an individual patient. All rows linked to a specific subject_id relate to the same person.

hadm_id: An identifier for each unique hospitalization of a patient, similar to how hospital encounters are tracked during clinical care.

seq_num: Reflects the priority or order assigned to procedures during the hospital stay. While it doesn’t always align with exact chronological events, it indicates procedural sequencing.

chartdate: The date when a specific procedure occurred, providing a timeline for clinical interventions.

icd_code, icd_version: These fields specify the procedure using ICD-9 or ICD-10 codes. The icd_code serves as a classification of the procedure performed, and icd_version distinguishes between coding systems. Both versions are widely used to accurately categorize clinical procedures.
 """
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
The procedureevents table stores data representing various procedure events that a patient undergoes during their ICU stay. Each procedure in the table provides detailed information, including timing, type, and related specifics.

Data Fields:
subject_id: A unique identifier for the patient. This field links the procedural event to a specific patient within the hospital system.
hadm_id: The hospital admission ID. This field associates the procedural event with a particular hospital admission instance.
stay_id: A unique identifier for the ICU stay. This field distinguishes each ICU admission event for a patient.
starttime: The date and time when the procedure started. This timestamp helps track the initiation of the procedure.
endtime: The date and time when the procedure ended. This timestamp helps track the completion of the procedure.
storetime: The date and time when the procedure was recorded in the system.
itemid: A unique identifier for the specific procedure performed.
value: The numerical value associated with the procedure, if applicable.
valueuom: The unit of measurement for the value, if applicable.
location: The specific location where the procedure took place.
locationcategory: The category of the location where the procedure was performed.
orderid: A unique identifier for the order associated with the procedure.
linkorderid: A unique identifier linking related orders.
ordercategoryname: The name of the category to which the order belongs.
secondaryordercategoryname: The name of the secondary category to which the order belongs.
ordercategorydescription: A description of the order category.
patientweight: The weight of the patient at the time of the procedure.
totalamount: The total amount related to the procedure, if applicable.
totalamountuom: The unit of measurement for the total amount, if applicable.
isopenbag: Indicates whether an open bag was used for the procedure.
continueinnextdept: Indicates if the procedure will continue in the next department.
cancelreason: The reason for canceling the procedure, if applicable.
statusdescription: A description of the status of the procedure.
comments_date: The date of any comments related to the procedure.
originalamount: The original amount associated with the procedure.
originalrate: The original rate associated with the procedure."""

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
#------------------------------------------------------------

# Se crea la consulta
filenames = ["admissions","d_items","datetimeevents","diagnoses_icd","emar","hcpsevents","icustays","inputevents","microbiologyevents","outputevents","patients","pharmacy","prescriptions","procedureevents","procedures_icd","services","transfers"]
# Se crea la consulta

# Encuentra el índice del filename correspondiente
start_index = filenames.index("admissions")

# Crea la consulta comenzando desde "patients"
for filename in filenames[start_index:]:
    print("----------------------------------Filename: ", filename,"----------------------------------")

    data = load_data("Fase3/" + filename + "_mock.csv")  # Cargar los datos tabulares desde un archivo CSV

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
    context,final_index = get_top_texts(vec_dict_similarity, bm25_similarity_dict, univ_sent_encoder_similarity, fasttext_similarities, profiles, top_k)  # Obtener los textos más relevantes   
    print(context)
    # Se obtiene el contexto específico para la consulta
    for i in range(5):  # Realizar el proceso de generación de respuesta 5 veces
        # Abrir el archivo en modo de escritura ('w')
        with open("C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/mapeos/mapeosCoTParallel/mapeo3_"+filename+".txt", 'a') as archivo:
            # Código para escribir en el archivo
                # Convertir la variable a string y escribirla en el archivo
            print("---------------------------------", i)
            print("------------------------LLAMA RESPONSE------------------------------------")
            # Se crea la respuesta usando el modelo Llama
            separador = "------------------------LLAMA RESPONSE------------------------------------"
            archivo.write(str(separador))
            response = generate_response(query, "Llama", context)
            print(response)  # Imprimir la respuesta generada por Llama
            archivo.write(str(response))
            print("------------------------GPT RESPONSE---------------------------------------")
            # Se crea la respuesta usando el modelo GPT
            #response = generate_response(query, "GPT", context)
            #print(response)  # Imprimir la respuesta generada por GPT
            #archivo.write(str(response))
            #print("---------------------------------------------------------------------------")