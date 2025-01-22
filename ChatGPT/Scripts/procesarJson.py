import json

# Función para convertir a NDJSON
def json_to_ndjson(json_data):
    ndjson_data = ""
    for item in json_data:
        ndjson_data += json.dumps(item) + "\n"
    return ndjson_data

# Ruta del archivo JSON de entrada
archivo_entrada = "serialized_encounter.ndjson"

with open(archivo_entrada,'r') as f:
    lines = f.readlines()
    for line in lines:
        json_data = json.load(line)
        ndjson_data = json_to_ndjson(json_data)
        with open('serialized_encounter_good.json', 'w') as fw:
            fw.write(ndjson_data)
    

