import re
import json 

expresionUUID_pat = r'(?<=Patient\/)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
expresionUUID_enc = r'(?<=\/)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
expresionUUID_loc = r'(?<=Location\/)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'

expresionUUID_id = r'(?<=")[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'


expresionID = r'(?<=_)[0-9a-fA-F]{8}(?=")'
expresionLoc = r'"name":\s*"(.*?)"'


def extract_patient_dict(file_path, expresionUUID, expresionID):
    result_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        match_uuid = re.search(expresionUUID, line)
        match_id = re.search(expresionID, line)
        
        # Asegúrate de que ambos patrones coinciden antes de añadir a la lista
        if match_uuid and match_id:
            result_list.append((match_uuid.group(0), match_id.group(0)))

    print("Longitud de la lista:", len(result_list))
    
    # Crear el diccionario a partir de la lista de tuplas
    dict_id = dict(result_list)
    
    print("Longitud del diccionario:", len(dict_id))
    return dict_id

def extract_location_dict(file_path, expresionUUID, expresionID):
    result_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        match_uuid = re.findall(expresionUUID, line)[0]
        match_id = re.findall(expresionID, line)[0]
        
        # Asegúrate de que ambos patrones coinciden antes de añadir a la lista
        if match_uuid != "" and match_id != "":
            result_list.append((match_uuid, match_id))

    print("Longitud de la lista:", len(result_list))
    
    # Crear el diccionario a partir de la lista de tuplas
    dict_id = dict(result_list)
    
    print("Longitud del diccionario:", len(dict_id))
    return dict_id
    

def replaceUUID(filer, fileout, dict_id, expresionUUID):
    with open(filer, 'r') as filer, open(fileout, 'w') as fileout:
        lines = filer.readlines()
        for line in lines:
            uuids = re.findall(expresionUUID, line)
            for locId in uuids:
                if locId != "":
                    id_ = dict_id.get(locId)
                     # Usa uuid como valor por defecto si no está en el diccionario
                    line = line.replace(locId, id_)
            fileout.write(line)

dict_id = extract_location_dict("MimicLocation.ndjson",expresionUUID_id,expresionLoc)
replaceUUID("MimicEncounter.ndjson","MimicEncounter2.ndjson",dict_id,expresionUUID_loc)
replaceUUID("MimicEncounter2.ndjson","MimicEncounter3.ndjson",dict_id,expresionUUID_loc)
print("----------------------------------------------------")


dict_id = extract_patient_dict("MimicPatient.ndjson",expresionUUID_id,expresionID)
replaceUUID("MimicEncounter3.ndjson","MimicEncounter4.ndjson",dict_id,expresionUUID_pat)