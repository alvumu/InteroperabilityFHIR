import pandas as pd

# Función para verificar si el mapeo de GPT es válido respecto al mapeo teórico
def es_valido(teorical_correct, gpt_mapped):
    """
    Comprueba si el mapeo realizado por GPT (gpt_mapped) es válido en relación con el mapeo teórico (teorical_correct).
    Retorna YES si es válido, NO en caso contrario.
    """
    # Dividir los valores por punto para jerarquías
    teorical_parts = teorical_correct.split('.')
    gpt_parts = gpt_mapped.split('.')
    
    # Validar si GPT es un prefijo de TeoricalCorrect
    return "YES" if gpt_parts == teorical_parts[:len(gpt_parts)] else "NO"

# Generar una lista de combinaciones de GPT y TeoricalCorrect
def validate_gpt_row(row):
    gpt_values = [x.strip().lower() for x in row['GPT'].split(';')]
    teorical_correct = [x.strip().lower() for x in row['TeoricalCorrect'].split('/')]
    for gpt_value in gpt_values:
        for correct in teorical_correct:
            if es_valido(correct, gpt_value) == "YES":
                return "YES"
    return "NO"

def validate_llama_row(row):
    print(row["attribute"]) 
    llama_values = [x.strip().lower() for x in row['Llama'].split(';')]
    teorical_correct = [x.strip().lower() for x in row['TeoricalCorrect'].split('/')]
    for llama_value in llama_values:
        for correct in teorical_correct:
            if es_valido(correct, llama_value) == "YES":
                return "YES"
    return "NO"

def validate_mappings(archivo1_path, archivo2_path, output_path):
    # Leer archivos
    df1 = pd.read_csv(archivo1_path)
    df2 = pd.read_csv(archivo2_path)

    # Realizar merge en base a 'attribute'
    df = pd.merge(df1, df2[['attribute', 'TeoricalCorrect']], on='attribute', how='right')

    # Asegurar que las columnas no tengan valores NaN
    df['Llama'] = df['Llama'].fillna('')
    df['TeoricalCorrect'] = df['TeoricalCorrect'].fillna('')



    # Aplicar validación fila por fila
    #df['CorrectGPT'] = df.apply(validate_gpt_row, axis=1)
    df['CorrectLlama'] = df.apply(validate_llama_row, axis=1)

    # Guardar el resultado en un archivo CSV
    df.to_csv(output_path, index=False)

# Uso del script
archivo1_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/formatted_output_clusters_v12_llama.csv'  # Reemplaza con tu ruta real
archivo2_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/correct_attributes_v2.csv'  # Reemplaza con tu ruta real
output_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/formatted_output_clusters_v12_llama_validated.csv'

validate_mappings(archivo1_path, archivo2_path, output_path)
