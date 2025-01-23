import json
import csv
import re

def parse_llama_response(llama_content):
    try:
        # Parsear el contenido JSON de la respuesta de Llama
        llama_json = json.loads(llama_content)
        print(llama_json)
        # Extraer los 'fhir_attributes' del mapeo
        mappings = llama_json.get('mappings', [])
        attribute_mapping = {}

        for mapping in mappings:
            table_attribute_name = mapping.get('table_attribute_name', '')
            fhir_attrs = mapping.get('fhir_attributes', {})
            if table_attribute_name:
                # Combinar los atributos FHIR en una cadena separada por punto y coma
                fhir_keys = ';'.join(fhir_attrs.keys())
                attribute_mapping[table_attribute_name] = fhir_keys

        return attribute_mapping
    except json.JSONDecodeError:
        return {}

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        # Escribir la cabecera
        csv_writer.writerow(['attribute', 'Llama'])

        attribute = ''
        llama_content = ''
        reading_llama = False

        lines = infile.readlines()
        lines.append('EOF')  # Añadir un marcador EOF para procesar el último atributo

        for line in lines:
            #print(line)
            line = line.strip()
            if line.startswith('Cluster:'):
                # Procesar el atributo anterior si hay datos recolectados
                if llama_content:
                    attribute_mapping = parse_llama_response(llama_content)
                    for attr, llama_result in attribute_mapping.items():
                        csv_writer.writerow([attr, llama_result])
                    attribute = ''
                    llama_content = ''
                reading_llama = False
            elif line.startswith('LLAMA Response:'):
                reading_llama = True
                llama_content = ''
            elif line == 'EOF':
                # Procesar el atributo anterior si hay datos recolectados
                if llama_content:
                    attribute_mapping = parse_llama_response(llama_content)
                    for attr, llama_result in attribute_mapping.items():
                        csv_writer.writerow([attr, llama_result])
                reading_llama = False
            else:
                if reading_llama:
                    llama_content += line

# Especifica las rutas de los archivos de entrada y salida
input_file_path = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/mapeo_clusters_v12_llama.json"  # Reemplaza con la ruta de tu archivo de entrada
output_file_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/formatted_output_clusters_v12_llama.csv'  # Reemplaza con la ruta deseada para el archivo de salida

# Procesa el archivo
process_file(input_file_path, output_file_path)
