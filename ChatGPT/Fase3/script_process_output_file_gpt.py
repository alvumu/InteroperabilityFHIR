import json
import csv
import re

def fix_invalid_json(gpt_content):
    # Remove triple backticks and 'json' specifiers
    gpt_content = gpt_content.strip('```').strip('json').strip()
    
    # Replace multiple 'fhir_attribute_name' entries with a list
    pattern = r'"fhir_attribute_name":\s*"([^"]+)"(?:,\s*"([^"]+)")*'
    def repl(m):
        all_values = m.group(0).split('",')
        values = [v.split(':')[-1].strip(' "').strip() for v in all_values]
        return '"fhir_attribute_name": {}'.format(json.dumps(values))
    
    fixed_content = re.sub(pattern, repl, gpt_content)
    return fixed_content

def parse_gpt_response(gpt_content):
    # Fix the invalid JSON
    fixed_content = fix_invalid_json(gpt_content)
    try:
        # Parse the JSON content
        print(fixed_content)
        gpt_json = json.loads(fixed_content)
        # Ensure gpt_json is a list of dictionaries
        results = []
        for item in gpt_json:
            table_attribute_name = item.get('table_attribute_name', '')
            fhir_attribute = item.get('fhir_attribute_name', [])
            if isinstance(fhir_attribute, list):
                fhir_attribute_str = ';'.join(fhir_attribute)
            elif isinstance(fhir_attribute, str):
                fhir_attribute_str = fhir_attribute
            else:
                fhir_attribute_str = ''
            results.append((table_attribute_name, fhir_attribute_str))
        return results
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return []

def process_file(input_file_path, output_file_path):
    attribute_mapping = {}
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        cluster_name = ''
        gpt_content = ''
        reading_gpt = False

        lines = infile.readlines()
        lines.append('EOF')  # Add an EOF marker to process the last GPT content

        for line in lines:
            line = line.strip()
            if line.startswith('Cluster:'):
                # Process the previous GPT content if any
                if gpt_content:
                    # Parse the GPT response
                    gpt_results = parse_gpt_response(gpt_content)
                    # Collect mappings
                    for table_attribute_name, fhir_attribute_str in gpt_results:
                        if table_attribute_name in attribute_mapping:
                            # Merge the fhir_attribute_str
                            existing_fhir_attribute = attribute_mapping[table_attribute_name]
                            # Merge and remove duplicates
                            merged_attributes = set(existing_fhir_attribute.split(';') + fhir_attribute_str.split(';'))
                            attribute_mapping[table_attribute_name] = ';'.join(merged_attributes)
                        else:
                            attribute_mapping[table_attribute_name] = fhir_attribute_str
                    # Reset variables for next cluster
                    gpt_content = ''
                # Set the new cluster name
                cluster_name = line.replace('Cluster:', '').strip()
                reading_gpt = False
            elif line.startswith('GPT Response:'):
                reading_gpt = True
                gpt_content = ''
            elif line == '```':
                # End of GPT Response
                reading_gpt = False
            elif line == 'EOF':
                # Process the last GPT content if any
                if gpt_content:
                    gpt_results = parse_gpt_response(gpt_content)
                    for table_attribute_name, fhir_attribute_str in gpt_results:
                        if table_attribute_name in attribute_mapping:
                            existing_fhir_attribute = attribute_mapping[table_attribute_name]
                            merged_attributes = set(existing_fhir_attribute.split(';') + fhir_attribute_str.split(';'))
                            attribute_mapping[table_attribute_name] = ';'.join(merged_attributes)
                        else:
                            attribute_mapping[table_attribute_name] = fhir_attribute_str
                    gpt_content = ''
            else:
                if reading_gpt:
                    gpt_content += line + '\n'
    # Write to CSV
    with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        # Write the header
        csv_writer.writerow(['attribute', 'GPT'])
        for attribute, gpt in attribute_mapping.items():
            csv_writer.writerow([attribute, gpt])

# Especifica las rutas de los archivos de entrada y salida
input_file_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/formatted_CoTParallel_results.json'  # Reemplaza con la ruta de tu archivo de entrada
output_file_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase4/clustering/formatted_output_clusters_v12.csv'  # Reemplaza con la ruta deseada para el archivo de salida

# Process the file
process_file(input_file_path, output_file_path)
