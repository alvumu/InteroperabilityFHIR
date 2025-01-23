import os
import json
import numpy as np
import requests
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import nltk
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import ParameterGrid
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel

nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))  # Stopwords en inglés

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct]
    filtered_tokens = [w for w in tokens if w not in stop_words and w.strip() != '']
    return " ".join(filtered_tokens)

def detect_value_type(values):
    if all(isinstance(v, (int, float)) for v in values if v is not None):
        return 'Numeric'
    elif all(isinstance(v, str) for v in values if v is not None):
        if all('-' in v for v in values):
            return 'Date/Time'
        else:
            return 'Categorical'
    else:
        return 'Mixed'

def cluster_and_evaluate(attribute_embeddings, attribute_texts, method_params):
    clustering_results = []
    clustering_algorithms = {
        'KMeans': {
            'model': KMeans,
            'params': {'n_clusters': range(3, 15), 'random_state': [42]}
        },
        'AgglomerativeClustering': {
            'model': AgglomerativeClustering,
            'params': {'n_clusters': range(3, 15), 'metric': ['euclidean', 'cosine'], 'linkage': ['ward', 'complete', 'average', 'single']}
        },
        'DBSCAN': {
            'model': DBSCAN,
            'params': {'eps': [0.5, 0.4, 0.3, 0.2, 0.1], 'min_samples': [6, 7, 8], 'metric': ['euclidean', 'cosine']}
        },
        'Birch': {
            'model': Birch,
            'params': {'n_clusters': range(3, 15), 'threshold': [0.1, 0.5, 1.0]}
        },
        'OPTICS': {
            'model': OPTICS,
            'params': {'min_samples': [6, 7, 8], 'metric': ['euclidean', 'cosine']}
        },
        'SpectralClustering': {
            'model': SpectralClustering,
            'params': {'n_clusters': range(3, 15), 'affinity': ['nearest_neighbors', 'rbf'], 'random_state': [42]}
        }
    }

    for algo_name, algo in clustering_algorithms.items():
        Model = algo['model']
        param_grid = ParameterGrid(algo['params'])
        for params in param_grid:
            if algo_name == 'AgglomerativeClustering':
                linkage = params.get('linkage')
                metric = params.get('metric')
                if linkage == 'ward' and metric != 'euclidean':
                    continue
                if linkage == 'single' and metric == 'cosine':
                    continue
            try:
                model = Model(**params)
                if algo_name == 'SpectralClustering':
                    cluster_labels = model.fit_predict(attribute_embeddings)
                else:
                    cluster_labels = model.fit(attribute_embeddings).labels_
                
                n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                if n_clusters_ < 3:
                    continue

                cluster_counts = Counter(cluster_labels)
                if -1 in cluster_counts:
                    del cluster_counts[-1]
                if any(size <= 5 for size in cluster_counts.values()):
                    continue

                silhouette_avg = silhouette_score(attribute_embeddings, cluster_labels)
                ch_score = calinski_harabasz_score(attribute_embeddings, cluster_labels)
                db_score = davies_bouldin_score(attribute_embeddings, cluster_labels)

                clustering_results.append({
                    'algorithm': algo_name,
                    'params': params,
                    'n_clusters': n_clusters_,
                    'cluster_sizes': cluster_counts,
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': ch_score,
                    'davies_bouldin_score': db_score,
                    'labels': cluster_labels
                })

                print(f"{algo_name} con parámetros {params}: n_clusters = {n_clusters_}, cluster_sizes = {cluster_counts}, silhouette = {silhouette_avg:.4f}, CH = {ch_score:.4f}, DB = {db_score:.4f}")
            except Exception as e:
                print(f"Error con {algo_name} y parámetros {params}: {e}")

    return clustering_results

def select_best_clustering(clustering_results):
    if not clustering_results:
        print("No se encontraron resultados de clustering que cumplan con los requisitos.")
        return None, None
    best_result = max(clustering_results, key=lambda x: x['silhouette_score'])
    print(f"\nMejor método: {best_result['algorithm']} con parámetros {best_result['params']}")
    print(f"n_clusters: {best_result['n_clusters']}, cluster_sizes: {best_result['cluster_sizes']}")
    print(f"Silhouette Score: {best_result['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Score: {best_result['calinski_harabasz_score']:.4f}")
    print(f"Davies-Bouldin Score: {best_result['davies_bouldin_score']:.4f}")
    return best_result['labels'], best_result

def reduce_dimensionality(embeddings, method='pca', n_components=50):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError("Método de reducción de dimensionalidad no soportado.")
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized = embeddings / norms
    return normalized

def get_transformers_embedding(model, tokenizer, texts):
    embeddings = []
    model.eval()
    import torch
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**encoded)
            # Asumimos un modelo tipo BERT: outputs.last_hidden_state existe
            last_hidden = outputs.last_hidden_state
            cls_vector = last_hidden[:,0,:].squeeze(0).numpy()
            embeddings.append(cls_vector)
    embeddings = np.array(embeddings)
    return embeddings

def cluster_attributes(attributes):
    attribute_texts = []
    for attr in attributes:
        v_type = detect_value_type(attr.get('Values', []))
        combined_text = f"{attr['Attribute name']}. {attr['Description']} ValueType: {v_type}"
        attribute_texts.append(combined_text)

    preprocessed_texts = [preprocess_text(text) for text in attribute_texts]

    # Diccionario con modelos
    # Valores pueden ser o bien instancias de SentenceTransformer o strings (nombres de modelos Transformers)
    embedding_models = {
        'neuml/pubmedbert-base-embeddings': SentenceTransformer('neuml/pubmedbert-base-embeddings'),
        'abhinand/MedEmbed-large-v0.1': SentenceTransformer('abhinand/MedEmbed-large-v0.1'),
        'medicalai/ClinicalBERT': "medicalai/ClinicalBERT",
        'dmis-lab/biobert-v1.1': 'dmis-lab/biobert-v1.1',
    }

    # Cargar los modelos Transformers una sola vez
    # Si es string => cargar tokenizer y model
    for key, val in embedding_models.items():
        if isinstance(val, str):
            tokenizer = AutoTokenizer.from_pretrained(val)
            model = AutoModel.from_pretrained(val)
            embedding_models[key] = (model, tokenizer)

    all_clustering_results = []
    for model_name, model_obj in embedding_models.items():
        print(f"\nGenerando embeddings con el modelo: {model_name}")
        if isinstance(model_obj, SentenceTransformer):
            attribute_embeddings = model_obj.encode(preprocessed_texts, show_progress_bar=True)
        else:
            # model_obj es una tupla (model, tokenizer)
            model, tokenizer = model_obj
            attribute_embeddings = get_transformers_embedding(model, tokenizer, preprocessed_texts)

        attribute_embeddings = normalize_embeddings(attribute_embeddings)
        # Si quieres, puedes reducir dimensionalidad:
        # reduced_embeddings = reduce_dimensionality(attribute_embeddings, method='pca', n_components=50)
        # Usaremos attribute_embeddings directamente:
        clustering_results = cluster_and_evaluate(attribute_embeddings, attribute_texts, method_params=None)

        for result in clustering_results:
            result['embedding_model'] = model_name
        all_clustering_results.extend(clustering_results)

    if not all_clustering_results:
        print("No se encontraron clusterings que cumplan con los requisitos.")
        return {}, None, embedding_models

    selected_labels, best_result = select_best_clustering(all_clustering_results)
    if selected_labels is None:
        return {}, None, embedding_models

    cluster_labels = selected_labels
    cluster_embedding_model = best_result['embedding_model']

    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(attributes[idx])

    return clusters, cluster_embedding_model, embedding_models

def send_request(payload, endpoint):
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
    }
    response = requests.post(endpoint, headers=headers, json=payload)
    return json.loads(response.content)

def generate_description(attributes):
    query = f"Given the following table attributes: {', '.join(attributes)}, generate a general description about the dataset and its contents."
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

def reciprocal_rank_fusion(rank_lists, k=60):
    rrf_scores = defaultdict(float)
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    sorted_rrf_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_rrf_scores

def tfidf_faiss_similarity(resource_texts, cluster_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(resource_texts).toarray().astype(np.float32)
    cluster_vec = vectorizer.transform([cluster_text]).toarray().astype(np.float32)
    similarity_scores = cosine_similarity(cluster_vec, tfidf_matrix)[0]
    similarity_dict = {i: score for i, score in enumerate(similarity_scores)}
    return similarity_dict

def compute_transformer_embeddings(cluster_text, resource_texts, embedding_models, model_name):
    model_obj = embedding_models[model_name]
    if isinstance(model_obj, SentenceTransformer):
        model = model_obj
        resource_embeddings = model.encode(resource_texts, show_progress_bar=True)
        resource_embeddings = normalize_embeddings(resource_embeddings)

        cluster_embedding = model.encode([cluster_text])[0]
        cluster_embedding = cluster_embedding / (np.linalg.norm(cluster_embedding) + 1e-9)
    else:
        # model_obj es (model, tokenizer)
        model, tokenizer = model_obj
        resource_embeddings = get_transformers_embedding(model, tokenizer, resource_texts)
        resource_embeddings = normalize_embeddings(resource_embeddings)

        cluster_emb = get_transformers_embedding(model, tokenizer, [cluster_text])
        cluster_emb = normalize_embeddings(cluster_emb)
        cluster_embedding = cluster_emb[0]

    similarity_scores = cosine_similarity([cluster_embedding], resource_embeddings)[0]
    print("Similitudes entre cluster y recursos:")
    print(similarity_scores)
    return {i: score for i, score in enumerate(similarity_scores)}

def compute_similarity(cluster_text, resource_texts, embedding_models, model_name):
    print(f"Computing similarity with model: {model_name}")
    transformer_similarity = compute_transformer_embeddings(cluster_text, resource_texts, embedding_models, model_name)
    return transformer_similarity

def find_top_similar_resources(cluster, cluster_model_name, embedding_models, resources, json_schemas, top_k=3):
    query = "##################### INPUT DATA ##################\n"
    for attr_info in cluster:
        attribute_name = attr_info.get("Attribute name")
        description = attr_info.get("Description")
        values = attr_info.get("Values")
        query += f"Attribute Name: {attribute_name}\n"
        query += f"Description: {description}\n"
        query += f"Sample Values: {values}\n"
        query += "----------------------------------------\n"

    cluster_text = preprocess_text(query)

    resource_texts = []
    for i, res in enumerate(resources):
        resource_text = f"{res['resource']}. {res['description']} {str(json_schemas[i])}"
        resource_texts.append(resource_text)

    preprocessed_resource_texts = [preprocess_text(text) for text in resource_texts]
    transformer_similarity = compute_similarity(cluster_text, preprocessed_resource_texts, embedding_models, cluster_model_name)
    top_k_indices = sorted(transformer_similarity, key=transformer_similarity.get, reverse=True)[:top_k]
    top_resources = [resources[i] for i in top_k_indices]

    return top_resources

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_ndjson(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def main():
    resource_path = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/datasetRecursos.ndjson"
    json_schemas_path = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/json_schemas/enriched_dataset_schemas.ndjson"
    attribute_path = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/filtered_data_attributes.json"
    output_path = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/clusters_v10.json"

    resources = load_dataset(resource_path)
    attributes = load_json(attribute_path)
    json_schemas = load_ndjson(json_schemas_path)

    clusters, cluster_embedding_model, embedding_models = cluster_attributes(attributes)

    if not clusters:
        print("No se pudo generar clusters que cumplan con los requisitos.")
        return

    rag_results = {}
    for i, (cluster_label, cluster_attrs) in enumerate(clusters.items()):
        top_resources = find_top_similar_resources(cluster_attrs, cluster_embedding_model, embedding_models, resources, json_schemas, top_k=5)
        [print(res["resource"]) for res in top_resources]
        rag_results[f"Cluster {i + 1}"] = {
            "Attributes": [attr["Attribute name"] for attr in cluster_attrs],
            "Top Resources": [{"Resource": res["resource"]} for res in top_resources]
        }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(rag_results, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
