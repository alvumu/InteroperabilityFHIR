import json
import csv
import re

def parse_llama_response(llama_content):
    try:
        # Parse the JSON content
        llama_json = json.loads(llama_content)
        # Extract the 'fhir_attributes' keys
        mappings = llama_json.get('mappings', [])
        fhir_attributes = []
        for mapping in mappings:
            fhir_attrs = mapping.get('fhir_attributes', {})
            fhir_attributes.extend(fhir_attrs.keys())
        # Join the attributes with semicolons
        return ';'.join(fhir_attributes)
    except json.JSONDecodeError:
        return ''

def parse_gpt_response(gpt_content):
    # Remove any triple backticks and 'json' specifiers
    gpt_content = gpt_content.strip('```').strip('json').strip()
    try:
        # Use regular expressions to fix invalid JSON (if necessary)
        # Replace multiple 'fhir_attribute_name' keys with a list
        gpt_content = re.sub(
            r'"fhir_attribute_name":/s*"([^"]+)",/s*"([^"]+)",/s*"([^"]+)"',
            r'"fhir_attribute_name": ["/1", "/2", "/3"]',
            gpt_content
        )
        # Parse the JSON content
        gpt_json = json.loads(gpt_content)
        # Extract the 'fhir_attribute_name' values
        fhir_attribute = gpt_json.get('fhir_attribute_name', [])
        if isinstance(fhir_attribute, list):
            return ';'.join(fhir_attribute)
        elif isinstance(fhir_attribute, str):
            return fhir_attribute
        else:
            return ''
    except json.JSONDecodeError:
        return ''

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile,\
         open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        # Write the header
        csv_writer.writerow(['attribute', 'llama', 'gpt'])

        attribute = ''
        llama_content = ''
        gpt_content = ''
        reading_llama = False
        reading_gpt = False

        lines = infile.readlines()
        lines.append('EOF')  # Add an EOF marker to process the last attribute

        for line in lines:
            line = line.strip()
            if line.startswith('Attribute:'):
                # Process the previous attribute if data is collected
                if attribute and llama_content and gpt_content:
                    # Parse the LLAMA response
                    #llama_result = parse_llama_response(llama_content)
                    # Parse the GPT response
                    gpt_result = parse_gpt_response(gpt_content)
                    # Write to CSV
                    csv_writer.writerow([attribute, llama_result, gpt_result])
                    # Reset variables for next attribute
                    attribute = ''
                    llama_content = ''
                    gpt_content = ''
                attribute = line.replace('Attribute:', '').strip()
                reading_llama = False
                reading_gpt = False
            elif line.startswith('LLAMA Response:'):
                reading_llama = True
                reading_gpt = False
                llama_content = ''
            elif line.startswith('GPT Response:'):
                reading_llama = False
                reading_gpt = True
                gpt_content = ''
            elif line.startswith('Cluster:') or line == 'EOF':
                # Process the previous attribute if data is collected
                if attribute and llama_content and gpt_content:
                    # Parse the LLAMA response
                    llama_result = parse_llama_response(llama_content)
                    # Parse the GPT response
                    gpt_result = parse_gpt_response(gpt_content)
                    # Write to CSV
                    csv_writer.writerow([attribute, llama_result, gpt_result])
                    # Reset variables for next attribute
                    attribute = ''
                    llama_content = ''
                    gpt_content = ''
            else:
                if reading_llama:
                    llama_content += line
                elif reading_gpt:
                    gpt_content += line

        # Handle the last attribute in case the file doesn't end with a marker
        if attribute and llama_content and gpt_content:
            #llama_result = parse_llama_response(llama_content)
            gpt_result = parse_gpt_response(gpt_content)
            csv_writer.writerow([attribute, llama_result, gpt_result])

# Especifica las rutas de los archivos de entrada y salida
#input_file_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/clusters_v4.json'  # Reemplaza con la ruta de tu archivo de entrada




input_file_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/mapeo_clusters_v10.json'  # Reemplaza con la ruta deseada para el archivo de entrada

output_file_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/formatted_output_clusters_v10.csv'  # Reemplaza con la ruta deseada para el archivo de salida

# Procesa el archivo
process_file(input_file_path, output_file_path)
