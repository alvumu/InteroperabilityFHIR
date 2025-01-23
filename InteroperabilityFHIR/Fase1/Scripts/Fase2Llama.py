import os
from together import Together
import requests
import os 
import json
import pandas as pd
import regex as re
# Se carga la libreria de OPENAI para poder hacer uso de la API
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


# Se define la funcion que se encargara de generar el texto de salida del LLM
def get_completion(prompt, model="meta-llama/Llama-3-8b-chat-hf"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's 
    )
    return response.choices[0].message.content

url = "http://localhost:4567/validate"



def add_profile_to_API(profile):
    response = requests.post("http://localhost:4567/profiles", profile)
    if (response.status_code == 200):
        print("Perfil añadido correctamente")

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

def remove_id(json_data):
    if isinstance(json_data, dict):
        if "id" in json_data:
            del json_data["id"]
        for key, value in json_data.items():
            remove_id(value)
    elif isinstance(json_data, list):
        for item in json_data:
            remove_id(item)


def compare_jsons(json1, json2, parent_key='', errors=None):
    """
    Recursively compare two JSON objects and return a list of differences.
    """
    remove_id(json1)
    remove_id(json2)
    if errors is None:
        errors = []

    if isinstance(json1, dict) and isinstance(json2, dict):
        # Compare keys
        keys1 = set(json1.keys())
        keys2 = set(json2.keys())

        # Keys present in json1 but not in json2
        keys_removed = keys1 - keys2
        for key in keys_removed:
            errors.append(f"Key '{parent_key}.{key}' is present in FHIR generated but not in FHIR example")

        # Keys present in json2 but not in json1
        keys_added = keys2 - keys1
        for key in keys_added:
            errors.append(f"Key '{parent_key}.{key}' is present in FHIR example but not in FHIR generated")

        # Keys present in both json1 and json2, recursively compare values
        common_keys = keys1.intersection(keys2)
        for key in common_keys:
            new_key = f"{parent_key}.{key}" if parent_key else key
            compare_jsons(json1[key], json2[key], parent_key=new_key, errors=errors)
            
    elif isinstance(json1, list) and isinstance(json2, list):
        # Compare lengths of lists
        len1 = len(json1)
        len2 = len(json2)

        if len1 != len2:
            errors.append(f"The length of list '{parent_key}' in FHIR example differs")
        else:
            # Recursively compare elements of lists
            for i in range(len(json1)):
                new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                compare_jsons(json1[i], json2[i], parent_key=new_key, errors=errors)
    else:
        # Compare values
        #if re.match(r'(?:location\[\d+\]\.)?(period\.start)', parent_key) or re.match(r'p(?:location\[\d+\]\.)?(period\.end)', parent_key):
        match_json1 = re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',json1)
        match_json2 = re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',json2)
          
        # if match_json1 and match_json2:
        #     if match_json1.group(0) != match_json2.group(0):
        #         print("--------------------")
        #         print(match_json1.group(0))
        #         print(match_json2.group(0))
        #         print("--------------------")
        #         errors.append(f"'{parent_key}': FHIR Generated value is wrong") 
        if json1.strip() != json2.strip():
              errors.append(f"'{parent_key}': FHIR Generated value is wrong")


    return errors


def uploadFHIRData(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
patients = uploadFHIRData("MIMIC/MimicEncounter4.ndjson")

def get_patient_resource(identifier, patients):
    for patient in patients:
        for id_entry in patient.get("identifier", []):
            if id_entry.get("value") == identifier:
                return patient
            
def get_response_api(apiResponse):
  if re.search(r"'severity':\s*'(error)'", apiResponse):
      return "Error"
  elif re.search(r"'severity':\s*'(warning)'", apiResponse):
      return "Warning"
  elif re.search(r"'severity':\s*'(success)'", apiResponse):
      return "Success"
  
fhir_profile =""" {
  "resourceType": "StructureDefinition",
  "id": "mimic-encounter",
  "url": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter",
  "name": "MimicEncounter",
  "title": "MIMIC Encounter",
  "status": "draft",
  "description": "A MIMIC encounter profile based on FHIR R4 Encounter.",
  "fhirVersion": "4.0.1",
  "kind": "resource",
  "abstract": false,
  "type": "Encounter",
  "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Encounter",
  "derivation": "constraint",
  "differential": {
    "element": [
      {
        "id": "Encounter.identifier",
        "path": "Encounter.identifier",
        "slicing": {
          "discriminator": [
            {
              "type": "value",
              "path": "system"
            }
          ],
          "rules": "open",
          "description": "Patient identifier.system slicing"
        },
        "min": 1
      },
      {
        "id": "Encounter.identifier:HOSP_ID",
        "path": "Encounter.identifier",
        "sliceName": "HOSP_ID",
        "min": 0,
        "max": "1"
      },
      {
        "id": "Encounter.identifier:HOSP_ID.system",
        "path": "Encounter.identifier.system",
        "min": 1,
        "patternUri": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp"
      },
      {
        "id": "Encounter.identifier:HOSP_ID.value",
        "path": "Encounter.identifier.value",
        "short": "Hospital encounter identifier",
        "min": 1,
        "constraint": [
          {
            "key": "mimic-encounter-id",
            "severity": "error",
            "human": "Identifier must be a 8 digit numeric value",
            "expression": "value.matches('^[0-9]{8}$')",
            "source": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"
          }
        ]
      },
      {
        "id": "Encounter.identifier:ED_ID",
        "path": "Encounter.identifier",
        "sliceName": "ED_ID",
        "min": 0,
        "max": "1"
      },
      {
        "id": "Encounter.identifier:ED_ID.system",
        "path": "Encounter.identifier.system",
        "min": 1,
        "patternUri": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-ed"
      },
      {
        "id": "Encounter.identifier:ED_ID.value",
        "path": "Encounter.identifier.value",
        "short": "ED encounter identifier",
        "min": 1,
        "constraint": [
          {
            "key": "mimic-encounter-id",
            "severity": "error",
            "human": "Identifier must be a 8 digit numeric value",
            "expression": "value.matches('^[0-9]{8}$')",
            "source": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"
          }
        ]
      },
      {
        "id": "Encounter.identifier:ICU_ID",
        "path": "Encounter.identifier",
        "sliceName": "ICU_ID",
        "min": 0,
        "max": "1"
      },
      {
        "id": "Encounter.identifier:ICU_ID.system",
        "path": "Encounter.identifier.system",
        "min": 1,
        "patternUri": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-icu"
      },
      {
        "id": "Encounter.identifier:ICU_ID.value",
        "path": "Encounter.identifier.value",
        "short": "ICU encounter identifier",
        "min": 1,
        "constraint": [
          {
            "key": "mimic-encounter-id",
            "severity": "error",
            "human": "Identifier must be a 8 digit numeric value",
            "expression": "value.matches('^[0-9]{8}$')",
            "source": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"
          }
        ]
      },
      {
        "id": "Encounter.status",
        "path": "Encounter.status",
        "patternCode": "finished"
      },
      {
        "id": "Encounter.type",
        "path": "Encounter.type",
        "min": 1,
        "binding": {
          "strength": "required",
          "valueSet": "http://mimic.mit.edu/fhir/mimic/ValueSet/mimic-encounter-type"
        }
      },
      {
        "id": "Encounter.serviceType",
        "path": "Encounter.serviceType",
        "binding": {
          "strength": "required",
          "valueSet": "http://mimic.mit.edu/fhir/mimic/ValueSet/mimic-services"
        }
      },
      {
        "id": "Encounter.subject",
        "path": "Encounter.subject",
        "min": 1,
        "type": [
          {
            "code": "Reference",
            "targetProfile": [
              "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"
            ]
          }
        ]
      },
      {
        "id": "Encounter.period",
        "path": "Encounter.period",
        "min": 1
      },
      {
        "id": "Encounter.period.start",
        "path": "Encounter.period.start",
        "min": 1
      },
      {
        "id": "Encounter.period.end",
        "path": "Encounter.period.end",
        "min": 1
      },
      {
        "id": "Encounter.hospitalization.admitSource",
        "path": "Encounter.hospitalization.admitSource",
        "binding": {
          "strength": "required",
          "valueSet": "http://mimic.mit.edu/fhir/mimic/ValueSet/mimic-admit-source"
        }
      },
      {
        "id": "Encounter.hospitalization.dischargeDisposition",
        "path": "Encounter.hospitalization.dischargeDisposition",
        "binding": {
          "strength": "required",
          "valueSet": "http://mimic.mit.edu/fhir/mimic/ValueSet/mimic-discharge-disposition"
        }
      },
      {
        "id": "Encounter.serviceProvider",
        "path": "Encounter.serviceProvider",
        "type": [
          {
            "code": "Reference",
            "targetProfile": [
              "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-organization"
            ]
          }
        ]
      },
      {
        "id": "Encounter.partOf",
        "path": "Encounter.partOf",
        "type": [
          {
            "code": "Reference",
            "targetProfile": [
              "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"
            ],
            "extension": [
              {
                "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-hierarchy",
                "valueBoolean": true
              }
            ]
          }
        ]
      }
    ]
  }
}
"""

patient_1_desc = """"Patient Info: The subject_id of the patient is 10002428. The hadm_id of the patient is 26549334. The hcp_code, short_description of the patient is [['G0378', 'Hospital observation per hr']]. The admittime of the patient is 2160-07-15 23:37:00. The dischtime of the patient is 2160-07-16 18:49:00. The admision_location of the patient is EMERGENCY ROOM. The class of the patient is OBSENC. The priority of the patient is R. The transfer_id of the patient is 33428301. The intime of the patient is 2160-07-16 19:15:04. The outtime of the patient is . The curr_service of the patient is MED. The subject_id of the patient is 10002428. The hadm_id of the patient is 26549334. The transfer_id of the patient is 38430233. The careunit of the patient is Emergency Department. The intime of the patient is 2160-07-15 17:34:00. The outtime of the patient is 2160-07-16 18:49:00. The subject_id of the patient is 10002428. The hadm_id of the patient is 26549334. The transfer_id of the patient is 33162098. The careunit of the patient is Emergency Department Observation. The intime of the patient is 2160-07-16 18:49:00. The outtime of the patient is 2160-07-16 19:15:04."
 """ 
patient_1_desc_json ="""{"subject_id": 10002428, "hadm_id": 26549334, "hcp_code, short_description": [["G0378", "Hospital observation per hr"]], "admittime": "2160-07-15 23:37:00-04:00", "dischtime": "2160-07-16 18:49:00-04:00", "admision_location": "EMERGENCY ROOM", "class": "OBSENC", "priority": "R", "transfer_id": 33428301, "intime": "2160-07-16 19:15:04-04:00", "outtime": , "curr_service": "MED"}{"subject_id": 10002428, "hadm_id": 26549334, "transfer_id": 38430233, "careunit": "Emergency Department", "intime": "2160-07-15 17:34:00-04:00", "outtime": "2160-07-16 18:49:00-04:00"}{"subject_id": 10002428, "hadm_id": 26549334, "transfer_id": 33162098, "careunit": "Emergency Department Observation", "intime": "2160-07-16 18:49:00-04:00", "outtime": "2160-07-16 19:15:04-04:00"}
"""
patient_1_fhir = """ {"id": "00d55770-bb49-5d07-b395-43967c41a0b6", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"]}, "type": [{"coding": [{"code": "G0378", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-hcpcs-cd", "display": "Hospital observation per hr"}]}], "class": {"code": "OBSENC", "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "display": "observation encounter"}, "period": {"end": "2160-07-16T18:49:00-04:00", "start": "2160-07-15T23:37:00-04:00"}, "status": "finished", "subject": {"reference": "Patient/10002428"}, "location": [{"period": {"end": "2160-07-16T18:49:00-04:00", "start": "2160-07-15T17:34:00-04:00"}, "location": {"reference": "Location/Emergency Department"}}, {"period": {"end": "2160-07-16T19:15:04-04:00", "start": "2160-07-16T18:49:00-04:00"}, "location": {"reference": "Location/Emergency Department Observation"}}], "priority": {"coding": [{"code": "R", "system": "http://terminology.hl7.org/CodeSystem/v3-ActPriority", "display": "routine"}]}, "identifier": [{"use": "usual", "value": "26549334", "system": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp", "assigner": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}], "serviceType": {"coding": [{"code": "MED", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-services"}]}, "resourceType": "Encounter", "hospitalization": {"admitSource": {"coding": [{"code": "EMERGENCY ROOM", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-admit-source"}]}}, "serviceProvider": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}"""


patient_2_desc = """ "Patient Info: The subject_id of the patient is 10002528. The hadm_id of the patient is 28605730. The hcp_code, short_description of the patient is [['99220', 'Hospital observation services']]. The admittime of the patient is 2170-03-13 03:13:00. The dischtime of the patient is 2170-03-18 14:00:00. The admision_location of the patient is PHYSICIAN REFERRAL. The class of the patient is OBSENC. The priority of the patient is R. The transfer_id of the patient is 31748219. The intime of the patient is 2170-03-18 14:53:05. The outtime of the patient is . The curr_service of the patient is MED. The subject_id of the patient is 10002528. The hadm_id of the patient is 28605730. The transfer_id of the patient is 37170067. The careunit of the patient is Emergency Department. The intime of the patient is 2170-03-12 22:02:00. The outtime of the patient is 2170-03-13 17:10:00. The subject_id of the patient is 10002528. The hadm_id of the patient is 28605730. The transfer_id of the patient is 33038583. The careunit of the patient is Emergency Department Observation. The intime of the patient is 2170-03-13 17:10:00. The outtime of the patient is 2170-03-13 17:10:01. The subject_id of the patient is 10002528. The hadm_id of the patient is 28605730. The transfer_id of the patient is 34460753. The careunit of the patient is Medicine. The intime of the patient is 2170-03-13 17:10:01. The outtime of the patient is 2170-03-13 18:36:49. The subject_id of the patient is 10002528. The hadm_id of the patient is 28605730. The transfer_id of the patient is 33057346. The careunit of the patient is Medicine. The intime of the patient is 2170-03-13 18:36:49. The outtime of the patient is 2170-03-14 06:12:17. The subject_id of the patient is 10002528. The hadm_id of the patient is 28605730. The transfer_id of the patient is 33125435. The careunit of the patient is . The intime of the patient is 2170-03-14 06:12:17. The outtime of the patient is 2170-03-14 06:12:53. The subject_id of the patient is 10002528. The hadm_id of the patient is 28605730. The transfer_id of the patient is 37767815. The careunit of the patient is Medicine. The intime of the patient is 2170-03-14 06:12:53. The outtime of the patient is 2170-03-18 14:53:05."
"""
patient_2_desc_json = """{"subject_id": 10002528, "hadm_id": 28605730, "hcp_code, short_description": [["99220", "Hospital observation services"]], "admittime": "2170-03-13 03:13:00-04:00", "dischtime": "2170-03-18 14:00:00-04:00", "admision_location": "PHYSICIAN REFERRAL", "class": "OBSENC", "priority": "R", "transfer_id": 31748219, "intime": "2170-03-18 14:53:05-04:00", "outtime": , "curr_service": "MED"}{"subject_id": 10002528, "hadm_id": 28605730, "transfer_id": 37170067, "careunit": "Emergency Department", "intime": "2170-03-12 22:02:00-04:00", "outtime": "2170-03-13 17:10:00-04:00"}{"subject_id": 10002528, "hadm_id": 28605730, "transfer_id": 33038583, "careunit": "Emergency Department Observation", "intime": "2170-03-13 17:10:00-04:00", "outtime": "2170-03-13 17:10:01-04:00"}{"subject_id": 10002528, "hadm_id": 28605730, "transfer_id": 34460753, "careunit": "Medicine", "intime": "2170-03-13 17:10:01-04:00", "outtime": "2170-03-13 18:36:49-04:00"}{"subject_id": 10002528, "hadm_id": 28605730, "transfer_id": 33057346, "careunit": "Medicine", "intime": "2170-03-13 18:36:49-04:00", "outtime": "2170-03-14 06:12:17-04:00"}{"subject_id": 10002528, "hadm_id": 28605730, "transfer_id": 33125435, "careunit": , "intime": "2170-03-14 06:12:17-04:00", "outtime": "2170-03-14 06:12:53-04:00"}{"subject_id": 10002528, "hadm_id": 28605730, "transfer_id": 37767815, "careunit": "Medicine", "intime": "2170-03-14 06:12:53-04:00", "outtime": "2170-03-18 14:53:05-04:00"}
"""
patient_2_fhir = """ {"id": "9660d0f5-0f91-53b1-b6b5-bc5969ebb855", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"]}, "type": [{"coding": [{"code": "99220", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-hcpcs-cd", "display": "Hospital observation services"}]}], "class": {"code": "OBSENC", "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "display": "observation encounter"}, "period": {"end": "2170-03-18T14:00:00-04:00", "start": "2170-03-13T03:13:00-04:00"}, "status": "finished", "subject": {"reference": "Patient/10002528"}, "location": [{"period": {"end": "2170-03-13T17:10:00-04:00", "start": "2170-03-12T22:02:00-04:00"}, "location": {"reference": "Location/Emergency Department"}}, {"period": {"end": "2170-03-13T17:10:01-04:00", "start": "2170-03-13T17:10:00-04:00"}, "location": {"reference": "Location/Emergency Department Observation"}}, {"period": {"end": "2170-03-13T18:36:49-04:00", "start": "2170-03-13T17:10:01-04:00"}, "location": {"reference": "Location/Medicine"}}, {"period": {"end": "2170-03-14T06:12:17-04:00", "start": "2170-03-13T18:36:49-04:00"}, "location": {"reference": "Location/Medicine"}}, {"period": {"end": "2170-03-14T06:12:53-04:00", "start": "2170-03-14T06:12:17-04:00"}, "location": {"reference": "Location/Discharge Lounge"}}, {"period": {"end": "2170-03-18T14:53:05-04:00", "start": "2170-03-14T06:12:53-04:00"}, "location": {"reference": "Location/Medicine"}}], "priority": {"coding": [{"code": "R", "system": "http://terminology.hl7.org/CodeSystem/v3-ActPriority", "display": "routine"}]}, "identifier": [{"use": "usual", "value": "28605730", "system": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp", "assigner": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}], "serviceType": {"coding": [{"code": "MED", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-services"}]}, "resourceType": "Encounter", "hospitalization": {"admitSource": {"coding": [{"code": "PHYSICIAN REFERRAL", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-admit-source"}]}}, "serviceProvider": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}
 """ 

patient_3_desc = """"Patient Info: The subject_id of the patient is 10002430. The hadm_id of the patient is 26295318. The hcp_code, short_description of the patient is [['308335008', 'Patient encounter procedure']]. The admittime of the patient is 2129-06-13 00:00:00. The dischtime of the patient is 2129-06-24 16:01:00. The admision_location of the patient is TRANSFER FROM HOSPITAL. The discharge_location of the patient is SKILLED NURSING FACILITY. The class of the patient is AMB. The priority of the patient is UR. The transfer_id of the patient is 32453412. The intime of the patient is 2129-06-24 16:24:50. The outtime of the patient is . The curr_service of the patient is CMED. The subject_id of the patient is 10002430. The hadm_id of the patient is 26295318. The transfer_id of the patient is 35501806. The careunit of the patient is Coronary Care Unit (CCU). The intime of the patient is 2129-06-13 00:43:08. The outtime of the patient is 2129-06-15 22:51:40. The subject_id of the patient is 10002430. The hadm_id of the patient is 26295318. The transfer_id of the patient is 37409506. The careunit of the patient is Medicine/Cardiology Intermediate. The intime of the patient is 2129-06-15 22:51:40. The outtime of the patient is 2129-06-24 16:24:50."
""" 
patient_3_desc_json = """{"subject_id": 10002430, "hadm_id": 26295318, "hcp_code, short_description": [["308335008", "Patient encounter procedure"]], "admittime": "2129-06-13 00:00:00-04:00", "dischtime": "2129-06-24 16:01:00-04:00", "admision_location": "TRANSFER FROM HOSPITAL", "discharge_location": "SKILLED NURSING FACILITY", "class": "AMB", "priority": "UR", "transfer_id": 32453412, "intime": "2129-06-24 16:24:50-04:00", "outtime": , "curr_service": "CMED"}{"subject_id": 10002430, "hadm_id": 26295318, "transfer_id": 35501806, "careunit": "Coronary Care Unit (CCU)", "intime": "2129-06-13 00:43:08-04:00", "outtime": "2129-06-15 22:51:40-04:00"}{"subject_id": 10002430, "hadm_id": 26295318, "transfer_id": 37409506, "careunit": "Medicine/Cardiology Intermediate", "intime": "2129-06-15 22:51:40-04:00", "outtime": "2129-06-24 16:24:50-04:00"}
"""
patient_3_fhir = """{"id": "ef31194f-e1c4-59f2-8319-3316250fdd1f", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"]}, "type": [{"coding": [{"code": "308335008", "system": "http://snomed.info/sct", "display": "Patient encounter procedure"}]}], "class": {"code": "AMB", "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "display": "ambulatory"}, "period": {"end": "2129-06-24T16:01:00-04:00", "start": "2129-06-13T00:00:00-04:00"}, "status": "finished", "subject": {"reference": "Patient/10002430"}, "location": [{"period": {"end": "2129-06-15T22:51:40-04:00", "start": "2129-06-13T00:43:08-04:00"}, "location": {"reference": "Location/Coronary Care Unit (CCU)"}}, {"period": {"end": "2129-06-24T16:24:50-04:00", "start": "2129-06-15T22:51:40-04:00"}, "location": {"reference": "Location/Medicine/Cardiology Intermediate"}}], "priority": {"coding": [{"code": "UR", "system": "http://terminology.hl7.org/CodeSystem/v3-ActPriority", "display": "urgent"}]}, "identifier": [{"use": "usual", "value": "26295318", "system": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp", "assigner": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}], "serviceType": {"coding": [{"code": "CMED", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-services"}]}, "resourceType": "Encounter", "hospitalization": {"admitSource": {"coding": [{"code": "TRANSFER FROM HOSPITAL", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-admit-source"}]}, "dischargeDisposition": {"coding": [{"code": "SKILLED NURSING FACILITY", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-discharge-disposition"}]}}, "serviceProvider": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}
 """

patient_4_desc = """"Patient Info: The subject_id of the patient is 10000117. The hadm_id of the patient is 22927623. The hcp_code, short_description of the patient is [['43239', 'Digestive system'], ['G0378', 'Hospital observation per hr']]. The admittime of the patient is 2181-11-15 02:05:00. The dischtime of the patient is 2181-11-15 14:52:00. The admision_location of the patient is EMERGENCY ROOM. The class of the patient is OBSENC. The priority of the patient is R. The transfer_id of the patient is 39048268. The intime of the patient is 2181-11-15 14:52:47. The outtime of the patient is . The curr_service of the patient is MED. The subject_id of the patient is 10000117. The hadm_id of the patient is 22927623. The transfer_id of the patient is 33270271. The careunit of the patient is Emergency Department. The intime of the patient is 2181-11-14 21:51:00. The outtime of the patient is 2181-11-15 02:06:42. The subject_id of the patient is 10000117. The hadm_id of the patient is 22927623. The transfer_id of the patient is 34010499. The careunit of the patient is Emergency Department Observation. The intime of the patient is 2181-11-15 02:06:42. The outtime of the patient is 2181-11-15 08:15:53. The subject_id of the patient is 10000117. The hadm_id of the patient is 22927623. The transfer_id of the patient is 34491152. The careunit of the patient is Med/Surg. The intime of the patient is 2181-11-15 08:15:53. The outtime of the patient is 2181-11-15 14:52:47."
""" 
patient_4_desc_json = """{"subject_id": 10000117, "hadm_id": 22927623, "hcp_code, short_description": [["43239", "Digestive system"], ["G0378", "Hospital observation per hr"]], "admittime": "2181-11-15 02:05:00-05:00", "dischtime": "2181-11-15 14:52:00-05:00", "admision_location": "EMERGENCY ROOM", "class": "OBSENC", "priority": "R", "transfer_id": 39048268, "intime": "2181-11-15 14:52:47-05:00", "outtime": , "curr_service": "MED"}{"subject_id": 10000117, "hadm_id": 22927623, "transfer_id": 33270271, "careunit": "Emergency Department", "intime": "2181-11-14 21:51:00-05:00", "outtime": "2181-11-15 02:06:42-05:00"}{"subject_id": 10000117, "hadm_id": 22927623, "transfer_id": 34010499, "careunit": "Emergency Department Observation", "intime": "2181-11-15 02:06:42-05:00", "outtime": "2181-11-15 08:15:53-05:00"}{"subject_id": 10000117, "hadm_id": 22927623, "transfer_id": 34491152, "careunit": "Med/Surg", "intime": "2181-11-15 08:15:53-05:00", "outtime": "2181-11-15 14:52:47-05:00"}"""
patient_4_fhir = """{"id": "69ebeec4-c33e-59e4-9cb9-4f010404230d", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"]}, "type": [{"coding": [{"code": "43239", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-hcpcs-cd", "display": "Digestive system"}]}, {"coding": [{"code": "G0378", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-hcpcs-cd", "display": "Hospital observation per hr"}]}], "class": {"code": "OBSENC", "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "display": "observation encounter"}, "period": {"end": "2181-11-15T14:52:00-05:00", "start": "2181-11-15T02:05:00-05:00"}, "status": "finished", "subject": {"reference": "Patient/10000117"}, "location": [{"period": {"end": "2181-11-15T02:06:42-05:00", "start": "2181-11-14T21:51:00-05:00"}, "location": {"reference": "Location/Emergency Department"}}, {"period": {"end": "2181-11-15T08:15:53-05:00", "start": "2181-11-15T02:06:42-05:00"}, "location": {"reference": "Location/Emergency Department Observation"}}, {"period": {"end": "2181-11-15T14:52:47-05:00", "start": "2181-11-15T08:15:53-05:00"}, "location": {"reference": "Location/Med/Surg"}}], "priority": {"coding": [{"code": "R", "system": "http://terminology.hl7.org/CodeSystem/v3-ActPriority", "display": "routine"}]}, "identifier": [{"use": "usual", "value": "22927623", "system": "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp", "assigner": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}], "serviceType": {"coding": [{"code": "MED", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-services"}]}, "resourceType": "Encounter", "hospitalization": {"admitSource": {"coding": [{"code": "EMERGENCY ROOM", "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-admit-source"}]}}, "serviceProvider": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}
 """ 

profile = "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-encounter"

add_profile_to_API(fhir_profile)

def process_plain_text(contador,prompt_type,profile,file_path, output_path, continue_line=None):
  with open(file_path, "r") as file:
        lines = file.readlines()
        if continue_line is not None:
            lines = lines[continue_line:]
        for line in lines:
          if "Patient Info:" in line:
            patient_to_generate = line.strip()
            if prompt_type == "few_shot_basic" :      
                prompt = f"""
                ##Task
                I will provide you a few examples of serialized data and the corresponding FHIR resource.
                Your task is to generate a FHIR resource based on the information about the patient .
                
                ##Examples

                Serialized data example : {patient_1_desc}
                FHIR resource example : {patient_1_fhir}

                Serialized data example : {patient_2_desc}
                FHIR resource example : {patient_2_fhir}

                Serialized data example : {patient_3_desc}
                FHIR resource example : {patient_3_fhir}

                Serialized data example : {patient_4_desc}
                FHIR resource example : {patient_4_fhir}

               
                ##Real problem :
                Serialized data : {patient_to_generate}
                FHIR resource to generate :

                ##Output: 
                -You must only answer the FHIR resource. 
                
                ##Indications: 
                -If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. Only generate FHIR resources.

                """

            elif prompt_type == "iterative_prompt" :
                     prompt = f'''##Task
                I will provide you a few examples of serialized data and the corresponding FHIR resource.
                Your task is to generate a FHIR resource based on the information about the patient.
                
                ##Examples

                Serialized data example : {patient_1_desc}
                FHIR resource example : {patient_1_fhir}

                Serialized data example : {patient_2_desc}
                FHIR resource example : {patient_2_fhir}

                Serialized data example : {patient_3_desc}
                FHIR resource example : {patient_3_fhir}

                Serialized data example : {patient_4_desc}
                FHIR resource example : {patient_4_fhir}

                ##Real problem :
                Serialized data : {patient_to_generate}
                FHIR resource to generate :

                ##Mapping instructions:                               
                - Each location is represented as an object within the location list.
                - Each location object has a period field to define the time period and a location field to specify the physical location referenced by its identifier.
                - The fields which map the location object are "careunit", "intime", and "outtime".
                - The resource field "type", should be an array of the value "coding" with the keys "code", "display" and "system". 
                
                ##Output: 
                -You must only generate the FHIR resource as plain text format. Do not show the input data.
                 
                ##Indications: 
                - Ensure to follow the mapping instruccions.
                - If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. 

                """                      ''' 
            elif prompt_type == "zero_shot" :
                    rules = """
1. Birth Date (birthDate):
   - Ensure it's a valid date in the format "YYYY-MM-DD". This information is found in the serialized data.
   - Estimate the birth date by subtracting the anchor_age from the intime. This information is found in the serialized data.

2. Extensions:
   - Include valid extensions as per the FHIR specification. This information is found in the serialized data.
   - Each extension must have a unique URL identifying its type. This information is found in the serialized data.
   - Race and ethnicity extensions must follow the specified URLs in the structure. This information is found in the serialized data.
   - The race extension must contain a valueCoding with keys "code", "display", and "system". This information is found in the serialized data.
   - The ethnicity extension must contain a valueCoding with keys "code", "display", and "system". This information is found in the serialized data.

3. Gender (gender):
   - Can be "male", "female", "other", or "unknown". This information is found in the serialized data.
   -  

4. Identifier (identifier):
   - Must contain at least one system and one value. This information is found in the serialized data.
   - The system must be a valid URL identifying the source of the identifier. This information is found in the serialized data.

5. Managing Organization (managingOrganization):
   - Should be a valid reference to an organization in the format "Organization/[id]". 
   - The [id] must be unique and correspond to an existing organization.
   - The id is the following : ee172322-118b-5716-abbc-18e4c5437e15

6. Marital Status (maritalStatus):
   - Should be a valid coding as per the specified system. This information is found in the serialized data.
   - The system must be a valid URL identifying the marital status coding system. 
   -  This information is found in the serialized data.
7. Meta Profile (meta.profile):
   - Must contain at least one valid URL identifying the patient profile. This information is found in the serialized data.

8. Name (name):
   - Should contain at least one last name (family) and specify the usage (use). This information is found in the serialized data.

9. Resource Type (resourceType):
   - Must be "Patient". This information is found in the serialized data.
"""
                    prompt = f"""Task : You Must generate a FHIR resource having the profile, the data serialized and following the giving rules.
                          Data Input :  '''{patient_to_generate}''' 
                          Profile : {profile}
                          Rules : {rules}
                          Output : You must only answer generate the FHIR resource about the the serialized data as Plain Text format simulating a JSON. Do not make any comments.

                          Recommendations:
                          - Estimate patient birthdate by subtracting the 'intime' field from the 'anchor_age'.
                          - You must set the "ethnicity" field as 'Hispanic or Latino' if the patient is "Hispanic/Latino" or "South American"; otherwise, set it to 'Not Hispanic or Latino'.
                          - Exclude any field with missing or "?" values from the resource.

                          Indications: If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. Only generate FHIR resources.

                        """       
            contador+=1
            fhir_generated = get_completion(prompt)
            print(fhir_generated)
            if '"id":' not in fhir_generated:
                fhir_generated = get_completion(prompt)
            expresion_regular = r'\{(?:[^{}]|(?R))*\}'
            fhir_generated = re.findall(expresion_regular, fhir_generated)[0]
            response = requests.post(url, fhir_generated, params={"profile": profile})
            validationAPI = get_response_api(str(response.json()['issue']))
            fhir_generated_dict = json.loads(fhir_generated) 
            # Acceder al campo identifier y luego al campo value
            print(contador)
            # Acceder al campo identifier y luego al campo value
            expresion_identifier = r'(?<=\:\s\")\d{8}(?=\"\,)'
            identifier_value = re.findall(expresion_identifier, fhir_generated)[0]
            print(identifier_value)
            json_to_compare = get_patient_resource(identifier_value, patients)
            if json_to_compare is None:
                numErrores = "No evaluable"
            else:
            # Se valida que el recurso generado sea igual al recurso esperado
                numErrores = compare_jsons(json_to_compare, fhir_generated_dict)
            fhir_generated = fhir_generated.replace("\\", "").replace("\n", "")
 # Añadir los datos al archivo NDJSON
            data = {
                'ID': identifier_value,
                'Recurso generado': fhir_generated,
                'Tipo Serializacion': "Plain Text",
                'Error API': validationAPI,
                'Error JSON': len(numErrores),
                'Tipo Error JSON': numErrores,
            }

    # Escribir los datos en el archivo NDJSON
            with open(output_path, 'a') as file:
              file.write(json.dumps(data) + '\n')


def process_json(contador,prompt_type,profile, file_path, output_path, continue_line=None):
    with open(file_path, "r") as file:
        lines = file.readlines()
        if continue_line is not None:
            lines = lines[continue_line:]
    for line in lines:
            patient_to_generate = line.strip()
            if prompt_type == "few_shot_basic" :      
                prompt = f"""
                ##Task
                I will provide you a few examples of serialized data and the corresponding FHIR resource.
                Your task is to generate a FHIR resource based on the information about the patient .
                
                ##Examples

                Serialized data example : {patient_1_desc_json}
                FHIR resource example : {patient_1_fhir}

                Serialized data example : {patient_2_desc_json}
                FHIR resource example : {patient_2_fhir}

                Serialized data example : {patient_3_desc_json}
                FHIR resource example : {patient_3_fhir}

                Serialized data example : {patient_4_desc_json}
                FHIR resource example : {patient_4_fhir}
               
                ##Real problem :
                Serialized data : {patient_to_generate}
                FHIR resource to generate :

                ##Output: 
                -You must only answer the FHIR resource generated as Plain Text simulating JSON format. 
                ##Indications: If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. Only generate FHIR resources.

                """
            elif prompt_type == "iterative_prompt" :
                     prompt = f'''I will provide you a few examples of serialized data and the corresponding FHIR resource. Your task is to generate a FHIR resource based on the information about the patient .     
                                  ##Examples
                                  Serialized data example : {patient_1_desc_json}
                                  FHIR resource example : {patient_1_fhir}

                                  Serialized data example : {patient_2_desc_json}
                                  FHIR resource example : {patient_2_fhir}

                                  Serialized data example : {patient_3_desc_json}
                                  FHIR resource example : {patient_3_fhir}

                                  Serialized data example : {patient_4_desc_json}
                                  FHIR resource example : {patient_4_fhir}
                                  
                                  ##Real problem :
                                  Serialized data : {patient_to_generate}
                                  FHIR resource to generate :

                                  ##Mapping instructions:                               
                                  - Each location is represented as an object within the location list.
                                  - Each location object has a period field to define the time period and a location field to specify the physical location referenced by its identifier.
                                  - The fields which map the location object are "careunit", "intime", and "outtime".
                                  - The resource field "type", should be an array of the value "coding" with the keys "code", "display" and "system". 
                                  
                                  ##Output: 
                                  -You must only generate the FHIR resource as plain text format. Do not show the input data.
                                   
                                  ##Indications: 
                                  - Ensure to follow the mapping instruccions.
                                  - If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. 
                      ''' 
            elif prompt_type == "zero_shot" :       
                    rules = """
1. Birth Date (birthDate):
   - Ensure it's a valid date in the format "YYYY-MM-DD". This information is found in the serialized data.
  
2. Extensions:
   - Include valid extensions as per the FHIR specification. This information is found in the serialized data.
   - Each extension must have a unique URL identifying its type. This information is found in the serialized data.
   - Race and ethnicity extensions must follow the specified URLs in the structure. This information is found in the serialized data.
   - The race extension must contain a valueCoding with keys "code", "display", and "system". This information is found in the serialized data.
   - The ethnicity extension must contain a valueCoding with keys "code", "display", and "system". This information is found in the serialized data.

3. Gender (gender):
   - Can be "male", "female", "other", or "unknown". This information is found in the serialized data.

4. Identifier (identifier):
   - Must contain at least one system and one value. This information is found in the serialized data.
   - The system must be a valid URL identifying the source of the identifier. This information is found in the serialized data.

5. Managing Organization (managingOrganization):
   - Should be a valid reference to an organization in the format "Organization/[id]". 
   - The [id] must be unique and correspond to an existing organization.
   - The id is the following : ee172322-118b-5716-abbc-18e4c5437e15


6. Marital Status (maritalStatus):
   - Should be a valid coding as per the specified system. This information is found in the serialized data.
   - The system must be a valid URL identifying the marital status coding system. 
   -  This information is found in the serialized data.
7. Meta Profile (meta.profile):
   - Must contain at least one valid URL identifying the patient profile. This information is found in the serialized data.

8. Name (name):
   - Should contain at least one last name (family) and specify the usage (use). This information is found in the serialized data.
   - The family field have the following format : "Patient_[subject_id]"
9. Resource Type (resourceType):
   - Must be "Patient". This information is found in the serialized data.
"""
                    prompt = f"""
                          ##Task : Generate the FHIR related to the profile and data serialized. You must follow the rules indicated in the prompt.
                          ##Input
                          Data Serialized :  '''{patient_to_generate}''' 
                          Profile : {profile}
                          Rules : : {rules}
                          ##Output: You must only answer generate the FHIR resource about the the serialized data as JSON format. Do not make any comments. You must answer strictly the answer following the JSON format, using all the delimiters and commas.
                          ##Indications: If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. Only generate FHIR resources.                                       
                        """             
            contador+=1
            fhir_generated = get_completion(prompt)
            print(fhir_generated)
            if '"id":' not in fhir_generated:
                fhir_generated = get_completion(prompt)
            expresion_regular = r'\{(?:[^{}]|(?R))*\}'
            fhir_generated = re.findall(expresion_regular, fhir_generated)[0]
            response = requests.post(url, fhir_generated, params={"profile": profile})
            validationAPI = get_response_api(str(response.json()['issue']))
            fhir_generated_dict = json.loads(fhir_generated) 
            # Acceder al campo identifier y luego al campo value
            expresion_identifier = r'(?<=\:\s\")\d{8}(?=\"\,)'
            identifier_value = re.findall(expresion_identifier, fhir_generated)[0]
            print(contador)
            print(identifier_value)
            json_to_compare = get_patient_resource(identifier_value, patients)
            if json_to_compare is None:
                numErrores = "No evaluable"
            else:
            # Se valida que el recurso generado sea igual al recurso esperado
                numErrores = compare_jsons(json_to_compare, fhir_generated_dict)
            fhir_generated = fhir_generated.replace("\\", "").replace("\n", "")
 # Añadir los datos al archivo NDJSON
            data = {
                'ID': identifier_value,
                'Recurso generado': fhir_generated,
                'Tipo Serializacion': "JSON",
                'Error API': validationAPI,
                'Error JSON': len(numErrores),
                'Tipo Error JSON': numErrores,
            }

    # Escribir los datos en el archivo NDJSON
            with open(output_path, 'a') as file:
              file.write(json.dumps(data) + '\n')
    
def generate_and_validate_fhir_resource(prompt_type, profile, type_serialization, file_path, output_path, continue_line=None):
    # Crear una lista para almacenar los datos que se añadirán al DataFrame
    if continue_line is not None:
      contador=continue_line
    else:
      contador=0
    if type_serialization == "Plain Text":
        process_plain_text(contador,prompt_type,profile,file_path, output_path, continue_line)
    elif type_serialization == "JSON":
        print("JSON")
        process_json(contador,prompt_type,profile, file_path, output_path, continue_line)

generate_and_validate_fhir_resource("iterative_prompt", profile, "Plain Text", "SerializedData/serialized_encounter_txt.json", "Llama3/Encounter/resultados_iterative_prompt_llama_enc_txt_3.ndjson",108)
generate_and_validate_fhir_resource("iterative_prompt", profile, "JSON", "SerializedData/serialized_encounter_json.json", "Llama3/Encounter/resultados_iterative_prompt_llama_enc_3.ndjson")
 