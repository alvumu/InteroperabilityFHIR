import json 
import re 


def serializar2text(resource):
    if (resource == "Encounter"):
        with open('SerializedData/serialized_encounter_json.json') as file:
            lines = file.readlines()
            for line in lines: 
                lista_jsons = re.findall(r'(?={).+?}', line)
                str = '"Patient Info:'
                for encounter in lista_jsons:
                    print(encounter)
                    data = json.loads(encounter)
                    for key, value in data.items():
                        str+= ' The {key} of the patient is {value}.'.format(key=key, value=value)
                with open('SerializedData/serialized_encounter_txt.json', 'a') as filew:
                    filew.write(str+'"'+"\n")
    elif (resource == "Patient"):
         with open('SerializedData/serialized_good_json-preprocessed.json') as file:
            lines = file.readlines()
            for line in lines: 
                str = '"Patient Info:'
                data = json.loads(line)
                for key, value in data.items():
                    str+= ' The {key} of the patient is {value}.'.format(key=key, value=value)
                with open('SerializedData/serialized_good_json-preprocessed_txt.json', 'a') as filew:
                    filew.write(str+'"'+"\n")


serializar2text("Encounter")