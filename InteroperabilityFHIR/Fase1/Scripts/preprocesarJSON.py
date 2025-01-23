from datetime import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import regex as re
import pandas as pd
import json
from datetime import datetime

def process_file(input_file, output_file):
       with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
          if "Patient Info:" in line:
            # Procesar cada línea del archivo de entrada
            line = line.strip()  # Eliminar espacios en blanco al inicio y al final de la línea
            # Identificar la raza del paciente
            race,ethnicity = identify_race_ethnicity(line)
            estimated_birthdate = estimate_birthdate(extract_anchor_age(line,"txt"),extract_intime(line,"txt"))
            # Insertar la información de fecha de nacimiento antes de "dod"
            index_dod = line.find("dod")  # Encontrar la posición de "dod"
            line_preprocessing = f"{line[:index_dod]} The birthdate of the patient is {estimated_birthdate}. The race of the patient is {race}. The ethnicity of the patient is {ethnicity}. {line[index_dod:]}"

            
            # Escribir la línea modificada en el archivo de salida
            f_out.write(line_preprocessing+"\n")


def process_patient_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        lines = f_in.readlines()
        for line in lines:
            print(line)
            data = json.loads(line)
            print(data)
            anchor_age = data["anchor_age"]
            intime = data["intime"]
            print(intime)
            anchor_age = extract_anchor_age(anchor_age, 'json')
            intime = extract_intime(intime, 'json')
            estimated_birthdate = estimate_birthdate(anchor_age, intime)
            for admission in data["admissions"]:
                # Identificar la raza del paciente
                race, ethnicity = identify_race_ethnicity(admission["ethnicity"])
                marital_status = admission["marital_status"]
                language = admission["language"]
                

            selected_fields = {
            "subject_id": data["subject_id"],
            "gender": data["gender"],
            "birthdate": estimated_birthdate,
            "ethnicity": ethnicity,
            "race":race,
            "marital_status":marital_status,
            "language":language,
            "dod":data["dod"]

                }

            # Convertir el resultado de vuelta a JSON
            f_out.write(json.dumps(selected_fields, indent=4))



def extract_anchor_age(text, serializacion):
    if serializacion == 'json':
        anchor_age = text
        return int(anchor_age)
    else:
        anchor_age = re.findall(r'(?<=\s)\d{2}(?=\.)', text)
        return int(anchor_age[0])

def extract_intime(text, serializacion):
    if serializacion == 'json':
        intime = text
        return dt.strptime(intime, '%Y-%m-%d %H:%M:%S')
    else:
        intime = re.findall(r'(?<=The intime of the patient is )\d{4}-\d{2}-\d{2}(?=\s)', text)
        return(dt.strptime(intime[0], '%Y-%m-%d'))
    

def hcpevents2dict():
        
# making dataframe  
    df = pd.read_csv("RawData/hcpcsevents.csv")

    subdf = df[["hadm_id", "hcpcs_cd", "short_description", "seq_num"]].sort_values(by=["hadm_id", "seq_num"])

    # Creación del diccionario
    result_dict = {}
    for index, row in subdf.iterrows():
        hadm_id = row['hadm_id']
        hcpcs_cd = row['hcpcs_cd']
        short_description = row['short_description']
        
        if hadm_id not in result_dict:
            result_dict[hadm_id] = []
        result_dict[hadm_id].append((hcpcs_cd, short_description))

    return result_dict        

def get_hcp_codes(hadm_id, dict):
    print(hadm_id)
    result_list = []
    if hadm_id in dict:
        for hcp_code, short_description in dict[hadm_id]:
            result_list.append([hcp_code,short_description])
        return result_list
    else:
        hcp_code = "308335008"
        short_description = "Patient encounter procedure"
        result_list.append([hcp_code,short_description])
        return result_list
def process_json_encounter(input_file, output_file):
    hcp_codes = hcpevents2dict()
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        lines = f_in.readlines()
        expresion_regular = r'\{(?:[^{}]|(?R))*\}'
        for line in lines:
            cadenaFinal = ""
            encounters = re.findall(expresion_regular, line)
            # Cargar los datos JSON en una lista de diccionarios
            records = [json.loads(record) for record in encounters]
            # Filtrar registros con outtime vacío
            no_outtime_record = [record for record in records if record['outtime'] == ""]
            no_outtime_record = no_outtime_record[0]
            # Filtrar registros con outtime no vacío
            valid_records = [record for record in records if record['outtime'] != ""]
            
            # Ordenar la lista por el campo 'intime'
            valid_records.sort(key=lambda x: datetime.strptime(x['intime'], '%Y-%m-%d %H:%M:%S'))
            admission_type = no_outtime_record["admission_type"]
            encounter_class = get_encounter_class(admission_type)
            encounter_priority = get_encounter_priority(admission_type)
            hcpCodes_list = get_hcp_codes(no_outtime_record["hadm_id"],hcp_codes)
            if no_outtime_record["discharge_location"] != "":
                
                
                selected_fields = {
                "subject_id": no_outtime_record["subject_id"],
                "hadm_id": no_outtime_record["hadm_id"],
                "hcp_code, short_description" : hcpCodes_list,
                "admittime": no_outtime_record["admittime"]+"-"+get_timezone_offset(no_outtime_record["admittime"]),
                "dischtime": no_outtime_record["dischtime"]+"-"+get_timezone_offset(no_outtime_record["dischtime"]),
                "admision_location": no_outtime_record["admission_location"],
                "discharge_location": no_outtime_record["discharge_location"],
                "class": encounter_class,
                "priority": encounter_priority,
                "transfer_id": no_outtime_record["transfer_id"],
                "intime": no_outtime_record["intime"]+"-"+get_timezone_offset(no_outtime_record["intime"]),
                "outtime": no_outtime_record["outtime"],
                "curr_service": no_outtime_record["curr_service"]
                }
            else:
                selected_fields = {
                "subject_id": no_outtime_record["subject_id"],
                "hadm_id": no_outtime_record["hadm_id"],
                "hcp_code, short_description" : hcpCodes_list,
                "admittime": no_outtime_record["admittime"]+"-"+get_timezone_offset(no_outtime_record["admittime"]),
                "dischtime": no_outtime_record["dischtime"]+"-"+get_timezone_offset(no_outtime_record["dischtime"]),
                "admision_location": no_outtime_record["admission_location"],
                "class": encounter_class,
                "priority": encounter_priority,
                "transfer_id": no_outtime_record["transfer_id"],
                "intime": no_outtime_record["intime"]+"-"+get_timezone_offset(no_outtime_record["intime"]),
                "outtime": no_outtime_record["outtime"],
                "curr_service": no_outtime_record["curr_service"]
                }

            # Convertir el resultado de vuelta a JSON
            cadenaFinal+= json.dumps(selected_fields) 
            for encounter in valid_records:
                    selected_fields = {
                        "subject_id": encounter["subject_id"],
                        "hadm_id": encounter["hadm_id"],
                        "transfer_id": encounter["transfer_id"],
                        "careunit": encounter["careunit"],
                        "intime": encounter["intime"]+"-"+get_timezone_offset(encounter["intime"]),
                        "outtime": encounter["outtime"]+"-"+get_timezone_offset(encounter["outtime"]),
                    }
                    cadenaFinal+= json.dumps(selected_fields)
                
            cadenaFinal = cadenaFinal.replace('"\"','')
            
            f_out.write(json.dumps(cadenaFinal,ensure_ascii= False, indent=4)+'\n')

                    



def estimate_birthdate(anchor_age, intime_date):
    # Obtener los campos relevantes del texto

    print(anchor_age)
    print(intime_date)

    anchor_age_date = relativedelta(years=anchor_age)
    patient_birthdate = intime_date - anchor_age_date
    # Formatear la fecha de nacimiento estimada como una cadena
    patient_birthdate_str = patient_birthdate.strftime('%Y-%m-%d')

    return patient_birthdate_str

def identify_race_ethnicity(text):
    # Buscar la raza del paciente
    if "WHITE" in text: 
        return "White", "Not Hispanic or Latino"
    elif "HISPANIC/LATINO" in text:
        return "White","Hispanic or Latino"
    elif "ASIAN" in text:
        return "Asian","Not Hispanic or Latino"
    elif "BLACK/AFRICAN AMERICAN" in text:
        return "Black or African American","Not Hispanic or Latino"
    else:
        return "White", "Not Hispanic or Latino"
    

def get_encounter_class(text):
    if text == "EU OBSERVATION" or text == "DIRECT OBSERVATION" or text == "OBSERVATION ADMIT" in text:
        return "OBSENC"
    elif text == "URGENT" or text == "ELECTIVE" or text == "AMBULATORY OBSERVATION" in text:
        return "AMB"
    elif text == "DIRECT EMER." or text == "EW EMER." in text:
        return "EMER"
    elif text == "SURGICAL SAME DAY ADMISSION" in text:
        return "SS"
    
def get_encounter_priority(text):
    if text =="EU OBSERVATION" or text == "AMBULATORY OBSERVATION" or text =="SURGICAL SAME DAY ADMISSION" or text == "DIRECT OBSERVATION" or text == "OBSERVATION ADMIT" in text:
        return "R"
    elif text == "URGENT" in text:
        return "UR"
    elif text == "ELECTIVE" in text:
        return "EL"
    elif text == "EW EMER." or text == "DIRECT EMER." in text:
        return "EM"
    
def get_timezone_offset(date):
    fecha = datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
    if fecha.month > 3 and fecha.month < 11 or fecha.day > 10 and fecha.month == 3:
        return "04:00"
    else:
        return "05:00"

# Ejemplo de uso
input_file_encounter = 'RawData/EncounterData.ndjson'
output_file_encounter = 'SerializedData/serialized_encounter_json.json'

input_file_patient = 'SerializedData/serialized_good_json.json'
output_file_patient = 'SerializedData/serialized_patient_good_json_preprocessed.json'
process_json_encounter(input_file_encounter, output_file_encounter)
