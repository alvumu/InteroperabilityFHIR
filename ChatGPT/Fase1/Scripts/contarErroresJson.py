import pandas as pd
import json

def uploadResults(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_line = json.loads(line)
            data.append(json_line)
    return data


def count_errores_json(path_input, path_output):
# Cargar datos del archivo CSV
    data = uploadResults(path_input)

    # Crear DataFrame
    df = pd.DataFrame(data)
    for i in range(len(df)):
        for error in df.loc[i, "Tipo Error JSON"]:
            if "Key '.deceasedDateTime' is present in FHIR" in error:
                df.loc[i,"Error JSON"] -=1   
                df.loc[i,"Tipo Error JSON"].remove(error)         

        if df.loc[i,"Error JSON"] == 0:
            df.loc[i,"Tipo Error JSON"] = "Sin errores"
            
    # Convertir la columna "Tipo Error JSON" a filas individuales
    df_exploded = df.explode("Tipo Error JSON")

    # Agregar el conteo de errores de "Error API" al DataFrame df_exploded
    count_error_api = df["Error API"].value_counts().reset_index()
    count_error_api.columns = ["Error", "count"]


    # Contar el número de ocurrencias de cada tipo de error
    error_counts = df_exploded["Tipo Error JSON"].value_counts().reset_index()

    error_counts.columns = ["Error", "count"]

    df_exploded = pd.concat([error_counts, count_error_api])


    # Añadimos los resultados de count_error_api al dataframe df_exploded
    df_exploded = pd.concat([error_counts, count_error_api])

    print(df_exploded)
    # Guardar resultados en un nuevo archivo CSV
    df_exploded.to_csv(path_output, index=False)

input_path = "GPT3/Encounter/resultados_iterative_encounter_json_final.ndjson"
output_path = "GPT3/Encounter/resultados_iterative_encounter_json_final_error_count.csv"
count_errores_json(input_path,output_path)