import os
import requests
import json
import regex as re
# Se carga la libreria de OPENAI para poder hacer uso de la API
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Se define la funcion que se encargara de generar el texto de salida del LLM
def get_completion(prompt, model="gpt-3.5-turbo"):
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
        if json1.strip() != json2.strip():
            errors.append(f"'{parent_key}': FHIR Generated value is wrong")

    return errors


def uploadFHIRData(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
patients = uploadFHIRData("MIMIC/MimicPatient.ndjson")
def get_patient_resource(identifier, patients):
    for patient in patients:
        for id_entry in patient.get("identifier", []):
            if id_entry.get("value") == identifier:
                return patient

fhir_profile =""" {
  "resourceType": "StructureDefinition",
  "id": "mimic-patient",
  "url": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient",
  "name": "MimicPatient",
  "title": "MIMIC Patient",
  "status": "draft",
  "description": "A MIMIC patient based on FHIR R4 Patient.",
  "fhirVersion": "4.0.1",
  "kind": "resource",
  "abstract": false,
  "type": "Patient",
  "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Patient",
  "derivation": "constraint",
  "differential": {
    "element": [
      {
        "id": "Patient.extension",
        "path": "Patient.extension",
        "slicing": {
          "discriminator": [
            {
              "type": "value",
              "path": "url"
            }
          ],
          "ordered": false,
          "rules": "open"
        }
      },
      {
        "id": "Patient.extension:race",
        "path": "Patient.extension",
        "sliceName": "race",
        "min": 0,
        "max": "1",
        "type": [
          {
            "code": "Extension",
            "profile": [
              "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
            ]
          }
        ]
      },
      {
        "id": "Patient.extension:race.extension",
        "path": "Patient.extension.extension",
        "min": 1
      },
      {
        "id": "Patient.extension:ethnicity",
        "path": "Patient.extension",
        "sliceName": "ethnicity",
        "min": 0,
        "max": "1",
        "type": [
          {
            "code": "Extension",
            "profile": [
              "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
            ]
          }
        ]
      },
      {
        "id": "Patient.extension:birthsex",
        "path": "Patient.extension",
        "sliceName": "birthsex",
        "min": 0,
        "max": "1",
        "type": [
          {
            "code": "Extension",
            "profile": [
              "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex"
            ]
          }
        ]
      },
      {
        "id": "Patient.identifier",
        "path": "Patient.identifier",
        "min": 1,
        "max": "1"
      },
      {
        "id": "Patient.identifier.system",
        "path": "Patient.identifier.system",
        "patternUri": "http://mimic.mit.edu/fhir/mimic/identifier/patient"
      },
      {
        "id": "Patient.name",
        "path": "Patient.name",
        "min": 1
      },
      {
        "id": "Patient.gender",
        "path": "Patient.gender",
        "min": 1
      },
      {
        "id": "Patient.deceased[x]",
        "path": "Patient.deceased[x]",
        "slicing": {
          "discriminator": [
            {
              "type": "type",
              "path": "$this"
            }
          ],
          "ordered": false,
          "rules": "open"
        }
      },
      {
        "id": "Patient.deceased[x]:deceasedDateTime",
        "path": "Patient.deceased[x]",
        "sliceName": "deceasedDateTime",
        "min": 0,
        "max": "1",
        "type": [
          {
            "code": "dateTime"
          }
        ]
      }
    ]
  }
}
"""
patient_1_desc = """The subject_id of the patient is 10039694.\nThe gender of the patient is F.\nThe anchor_age of the patient is 36.\nThe anchor_year of the patient is 2170.\nThe anchor_year_group of the patient is 2014 - 2016.\nThe dod of the patient is .\nThe intime of the patient is 2170-06-28 19:42:37.\nThe subject_id is 10039694. The hadm_id is 20374452. The admittime is 2170-06-28 19:41:00. The dischtime is 2170-07-02 16:41:00. The deathtime is . The admission_type is URGENT. The admission_location is TRANSFER FROM HOSPITAL. The discharge_location is HOME. The insurance is Medicare. The language is ENGLISH. The marital_status is SINGLE. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0."""

patient_1_desc_json ="""
    "subject_id": "10039694",
    "gender": "F",
    "anchor_age": "36",
    "anchor_year": "2170",
    "anchor_year_group": "2014 - 2016",
    "dod": "",
    "intime": "2170-06-28 19:42:37",
    "admissions": [
      {
        "subject_id": "10039694",
        "hadm_id": "20374452",
        "admittime": "2170-06-28 19:41:00",
        "dischtime": "2170-07-02 16:41:00",
        "deathtime": "",
        "admission_type": "URGENT",
        "admission_location": "TRANSFER FROM HOSPITAL",
        "discharge_location": "HOME",
        "insurance": "Medicare",
        "language": "ENGLISH",
        "marital_status": "SINGLE",
        "ethnicity": "WHITE",
        "edregtime": "",
        "edouttime": "",
        "hospital_expire_flag": "0"
      }

     ]
  """
patient_1_fhir = """ {"id": "745954f6-dfa2-5412-8c94-cf05931a8257", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_10039694"}], "gender": "female", "birthDate": "2134-06-28", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2106-3", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "White"}}, {"url": "text", "valueString": "White"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2186-5", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Not Hispanic or Latino"}}, {"url": "text", "valueString": "Not Hispanic or Latino"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}], "identifier": [{"value": "10039694", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "S", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}} """

patient_2_desc = """The subject_id of the patient is 10073847.\nThe gender of the patient is M.\nThe anchor_age of the patient is 53.\nThe anchor_year of the patient is 2134.\nThe anchor_year_group of the patient is 2011 - 2013.\nThe dod of the patient is .\nThe intime of the patient is 2134-02-24 09:47:50.\nThe subject_id is 10073847. The hadm_id is 22630133. The admittime is 2135-09-02 13:03:00. The dischtime is 2135-09-06 11:24:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is OTHER. The edregtime is . The edouttime is . The hospital_expire_flag is 0."""
patient_2_desc_json = """
    "subject_id": "10073847",
    "gender": "M",
    "anchor_age": "53",
    "anchor_year": "2134",
    "anchor_year_group": "2011 - 2013",
    "dod": "",
    "intime": "2134-02-24 09:47:50",
    "admissions": [
      {
        "subject_id": "10073847",
        "hadm_id": "22630133",
        "admittime": "2135-09-02 13:03:00",
        "dischtime": "2135-09-06 11:24:00",
        "deathtime": "",
        "admission_type": "DIRECT EMER.",
        "admission_location": "PHYSICIAN REFERRAL",
        "discharge_location": "HOME",
        "insurance": "Other",
        "language": "ENGLISH",
        "marital_status": "MARRIED",
        "ethnicity": "OTHER",
        "edregtime": "",
        "edouttime": "",
        "hospital_expire_flag": "0"
      }
    ]
  
  """
patient_2_fhir = """ {"id": "1b5bc42d-95ac-58d5-8912-97cae4636967", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_10073847"}], "gender": "male", "birthDate": "2081-02-24", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "UNK", "system": "http://terminology.hl7.org/CodeSystem/v3-NullFlavor", "display": "unknown"}}, {"url": "text", "valueString": "unknown"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "M"}], "identifier": [{"value": "10073847", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "M", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "deceasedDateTime": "2136-02-11", "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}} """

patient_3_desc = """Patient Info:\nThe subject_id of the patient is 16050270.\nThe gender of the patient is F.\nThe anchor_age of the patient is 30.\nThe anchor_year of the patient is 2161.\nThe anchor_year_group of the patient is 2014 - 2016.\nThe dod of the patient is .\nThe intime of the patient is 2161-07-01 21:14:50.\nThe subject_id is 16050270. The hadm_id is 23791681. The admittime is 2161-07-01 21:13:00. The dischtime is 2161-07-05 15:45:00. The deathtime is . The admission_type is URGENT. The admission_location is PROCEDURE SITE. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is UNKNOWN. The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0."""
patient_3_desc_json = """{"subject_id":"16050270","gender":"F","anchor_age":"30","anchor_year":"2161","anchor_year_group":"2014 - 2016","dod":"","intime":"2161-07-01 21:14:50","admissions":[{"subject_id":"16050270","hadm_id":"23791681","admittime":"2161-07-01 21:13:00","dischtime":"2161-07-05 15:45:00","deathtime":"","admission_type":"URGENT","admission_location":"PROCEDURE SITE","discharge_location":"HOME","insurance":"Other","language":"ENGLISH","marital_status": "UNKNOWN","ethnicity":"UNKNOWN","edregtime":"","edouttime":"","hospital_expire_flag":"0"}]}"""
patient_3_fhir = """{"id": "5c1ecd80-eb29-52cb-a931-8b359445231b", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_16050270"}], "gender": "female", "birthDate": "2131-07-01", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "UNK", "system": "http://terminology.hl7.org/CodeSystem/v3-NullFlavor", "display": "unknown"}}, {"url": "text", "valueString": "unknown"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}], "identifier": [{"value": "16050270", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "UNK", "system": "http://terminology.hl7.org/CodeSystem/v3-NullFlavor"}]}, "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}"""

patient_4_desc = """The subject_id of the patient is 10322234.\nThe gender of the patient is F.\nThe anchor_age of the patient is 53.\nThe anchor_year of the patient is 2122.\nThe anchor_year_group of the patient is 2008 - 2010.\nThe dod of the patient is .\nThe intime of the patient is 2122-02-24 21:30:00.\nThe subject_id is 10322234. The hadm_id is 29789116. The admittime is 2122-02-25 01:06:00. The dischtime is 2122-02-25 12:42:00. The deathtime is . The admission_type is EU OBSERVATION. The admission_location is EMERGENCY ROOM. The discharge_location is . The insurance is Other. The language is ?. The marital_status is MARRIED. The ethnicity is ASIAN. The edregtime is 2122-02-24 21:30:00. The edouttime is 2122-02-25 12:42:00. The hospital_expire_flag is 0."""
patient_4_desc_json = """"subject_id": "10322234",
    "gender": "F",
    "anchor_age": "53",
    "anchor_year": "2122",
    "anchor_year_group": "2008 - 2010",
    "dod": "",
    "intime": "2122-02-24 21:30:00",
    "admissions": [
      {
        "subject_id": "10322234",
        "hadm_id": "29789116",
        "admittime": "2122-02-25 01:06:00",
        "dischtime": "2122-02-25 12:42:00",
        "deathtime": "",
        "admission_type": "EU OBSERVATION",
        "admission_location": "EMERGENCY ROOM",
        "discharge_location": "",
        "insurance": "Other",
        "language": "?",
        "marital_status": "MARRIED",
        "ethnicity": "ASIAN",
        "edregtime": "2122-02-24 21:30:00",
        "edouttime": "2122-02-25 12:42:00",
        "hospital_expire_flag": "0"
      }
    ]
  """
patient_4_fhir = """{"id": "2059aa39-7cf1-541c-bd8d-60f362553060", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_10322234"}], "gender": "female", "birthDate": "2069-02-24", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2028-9", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Asian"}}, {"url": "text", "valueString": "Asian"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2186-5", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Not Hispanic or Latino"}}, {"url": "text", "valueString": "Not Hispanic or Latino"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}], "identifier": [{"value": "10322234", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "maritalStatus": {"coding": [{"code": "M", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}"""

patient_5_fhir = """{"id": "05eda4be-6ecd-5fcb-a8a9-d6c6b06e54f4", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_19499460"}], "gender": "female", "birthDate": "2061-04-21", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2054-5", "system": "urn:oid:2.16.840.1.113883.6.238", "display": " Black or African American"}}, {"url": "text", "valueString": " Black or African American"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2186-5", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Not Hispanic or Latino"}}, {"url": "text", "valueString": "Not Hispanic or Latino"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}], "identifier": [{"value": "19499460", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "W", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}} """ 
patient_5_desc_json = """ {
    "subject_id": "19499460",
    "gender": "F",
    "anchor_age": "65",
    "anchor_year": "2126",
    "anchor_year_group": "2011 - 2013",
    "dod": "",
    "intime": "2126-04-21 15:45:00",
    "admissions": [
      {
        "subject_id": "19499460",
        "hadm_id": "22582753",
        "admittime": "2131-08-10 17:39:00",
        "dischtime": "2131-08-11 14:15:00",
        "deathtime": "",
        "admission_type": "EU OBSERVATION",
        "admission_location": "EMERGENCY ROOM",
        "discharge_location": "",
        "insurance": "Other",
        "language": "ENGLISH",
        "marital_status": "WIDOWED",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "2131-08-10 14:28:00",
        "edouttime": "2131-08-10 21:23:00",
        "hospital_expire_flag": "0"
      },
      {
        "subject_id": "19499460",
        "hadm_id": "27369522",
        "admittime": "2131-12-15 23:56:00",
        "dischtime": "2131-12-16 16:54:00",
        "deathtime": "",
        "admission_type": "EU OBSERVATION",
        "admission_location": "EMERGENCY ROOM",
        "discharge_location": "",
        "insurance": "Other",
        "language": "ENGLISH",
        "marital_status": "WIDOWED",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "2131-12-15 20:56:00",
        "edouttime": "2131-12-16 16:54:00",
        "hospital_expire_flag": "0"
      },
      {
        "subject_id": "19499460",
        "hadm_id": "28219458",
        "admittime": "2132-11-01 21:59:00",
        "dischtime": "2132-11-06 17:50:00",
        "deathtime": "",
        "admission_type": "OBSERVATION ADMIT",
        "admission_location": "WALK-IN/SELF REFERRAL",
        "discharge_location": "HOME HEALTH CARE",
        "insurance": "Other",
        "language": "ENGLISH",
        "marital_status": "WIDOWED",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "2132-11-01 18:05:00",
        "edouttime": "2132-11-02 01:35:00",
        "hospital_expire_flag": "0"
      },
      {
        "subject_id": "19499460",
        "hadm_id": "23218130",
        "admittime": "2132-11-12 20:34:00",
        "dischtime": "2132-11-14 18:15:00",
        "deathtime": "",
        "admission_type": "OBSERVATION ADMIT",
        "admission_location": "PHYSICIAN REFERRAL",
        "discharge_location": "HOME HEALTH CARE",
        "insurance": "Other",
        "language": "ENGLISH",
        "marital_status": "WIDOWED",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "2132-11-12 12:37:00",
        "edouttime": "2132-11-12 22:45:00",
        "hospital_expire_flag": "0"
      },
      {
        "subject_id": "19499460",
        "hadm_id": "26064324",
        "admittime": "2131-07-18 01:17:00",
        "dischtime": "2131-07-23 17:40:00",
        "deathtime": "",
        "admission_type": "SURGICAL SAME DAY ADMISSION",
        "admission_location": "PHYSICIAN REFERRAL",
        "discharge_location": "HOME",
        "insurance": "Other",
        "language": "ENGLISH",
        "marital_status": "WIDOWED",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "",
        "edouttime": "",
        "hospital_expire_flag": "0"
      },
      {
        "subject_id": "19499460",
        "hadm_id": "23167346",
        "admittime": "2134-05-06 15:13:00",
        "dischtime": "2134-05-10 14:45:00",
        "deathtime": "",
        "admission_type": "OBSERVATION ADMIT",
        "admission_location": "PHYSICIAN REFERRAL",
        "discharge_location": "HOME",
        "insurance": "Other",
        "language": "ENGLISH",
        "marital_status": "WIDOWED",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "",
        "edouttime": "",
        "hospital_expire_flag": "0"
      }
    ]
  } """
patient_5_desc = """The subject_id of the patient is 19499460.\nThe gender of the patient is F.\nThe anchor_age of the patient is 65.\nThe anchor_year of the patient is 2126.\nThe anchor_year_group of the patient is 2011 - 2013.\nThe dod of the patient is .\nThe intime of the patient is 2126-04-21 15:45:00.\nThe subject_id is 19499460. The hadm_id is 23167346. The admittime is 2134-05-06 15:13:00. The dischtime is 2134-05-10 14:45:00. The deathtime is . The admission_type is OBSERVATION ADMIT. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is WIDOWED. The ethnicity is BLACK/AFRICAN AMERICAN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.",
"""







profile = "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"

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
                -You must only generate the FHIR resource as JSON format. Avoid to generate any additional information. Do not generate any code or comments.

                ##Indications: 
                - If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. Only generate FHIR resources.

                """
            elif prompt_type == "iterative_prompt" :
                     prompt = f'''I will provide you a few examples of serialized data and the corresponding FHIR resource. Your task is to generate a FHIR resource based on the information about the patient .     
                                  Data Input:
                                  - Patient Profile:
                                      - Profile: {fhir_profile}
                                  - Input Examples:
                                      - Patient 1: {patient_1_desc}
                                      - Patient 2: {patient_2_desc}
                                      - Patient 3: {patient_3_desc}
                                      - Patient 4: {patient_4_desc}
                                  - Output Examples:
                                      - Resource 1: {patient_1_fhir}
                                      - Resource 2: {patient_2_fhir}
                                      - Resource 3: {patient_3_fhir}
                                      - Resource 4: {patient_4_fhir}
                                  - Objective Data:
                                      - Patient to Generate: {patient_to_generate}
                                                        ##Mapping instructions:
                                                        - Ensure to generate the managingOrganization field.
                                                        - Ensure to generate the field meta
                                                        - Ensure to generate the field identifier
                                                        - The UNKNOWN values have associated the url "http://terminology.hl7.org/CodeSystem/v3-NullFlavor" . 

                                                        ##Output: 
                                                        -You must only generate the FHIR resource as plain text following the example resources. Avoid to generate any additional information. Do not generate any code or comments.
                                                        
                                                        ##Indications: 
                                                        -If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. 
                      ''' 
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
            fhir_generated = get_completion(prompt)
            if '"id":' not in fhir_generated:
                fhir_generated = get_completion(prompt)
            expresion_regular = r'\{(?:[^{}]|(?R))*\}'
            fhir_generated = re.findall(expresion_regular, fhir_generated)[0]
            print(fhir_generated)
            response = requests.post(url, fhir_generated, params={"profile": profile} )
            validationAPI = response.json()['issue'][0]['severity']
            fhir_generated_dict = json.loads(fhir_generated) 
            # Acceder al campo identifier y luego al campo value
            contador+=1
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

                ##Output
                -You must only generate the FHIR resource as JSON format. Avoid to generate any additional information. Do not generate any code or comments.

                ##Indications
                - If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. Only generate FHIR resources.

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
                                                        - The UNKNOWN values have associated the url "http://terminology.hl7.org/CodeSystem/v3-NullFlavor" . 
                                                        ##Output: 
                                                        -You must only generate the FHIR resource as plain text following the example resources. Avoid to generate any additional information. Do not generate any code or comments.
                                                        
                                                        ##Indications: 
                                                        -Ensure to forget the previous context and generate the FHIR resource based on the new serialized data.
                                                        -If the maximum token limit is reached before completing the resource, stop the task without providing an incomplete resource. 
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
            fhir_generated = get_completion(prompt)
            if '"id":' not in fhir_generated:
                fhir_generated = get_completion(prompt)
            expresion_regular = r'\{(?:[^{}]|(?R))*\}'
            fhir_generated = re.findall(expresion_regular, fhir_generated)[0]
            print(fhir_generated)
            response = requests.post(url, fhir_generated, params={"profile": profile} )
            validationAPI = response.json()['issue'][0]['severity']
            fhir_generated_dict = json.loads(fhir_generated) 
            # Acceder al campo identifier y luego al campo value
            contador+=1
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

#generate_and_validate_fhir_resource("iterative_prompt", profile, "Plain Text", "serialized_good-preprocessed.json","GPT3/Patient/resultados_iterative_prompt_gpt_good_txt_pre_v2.ndjson")
#generate_and_validate_fhir_resource("few_shot_basic", profile, "JSON", "SerializedData/serialized_good_json-preprocessed.json","GPT3/Good/resultados_few_shot_gpt_good_json_pre_nuevo_2.ndjson")

try: 
    output_file = "GPT3/Patient/resultados_iterative_prompt_gpt_good_txt_pre_nuevo_2.ndjson"
    generate_and_validate_fhir_resource("iterative_prompt", profile, "Plain Text", "SerializedData/serialized_good_txt-preprocessed.json",output_file,169)
except Exception as e:
    if re.match("Expecting ',' delimiter", str(e)): 
      output_file = "GPT3/Patient/resultados_iterative_prompt_gpt_good_txt_pre_nuevo_2.ndjson"
      with open(output_file, 'r', encoding='utf-8') as f_in:
          lines = f_in.readlines()
          len(lines)
          generate_and_validate_fhir_resource("iterative_prompt", profile, "Plain Text", "SerializedData/serialized_good_txt-preprocessed.json",output_file,len(lines))
    