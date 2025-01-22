import pandas as pd
import os  
import json
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

patient_1_desc = """The subject_id of the patient is 10039694.\nThe gender of the patient is F.\nThe anchor_age of the patient is 36.\nThe anchor_year of the patient is 2170.\nThe anchor_year_group of the patient is 2014 - 2016.\nThe dod of the patient is .\nThe intime of the patient is 2170-06-28 19:42:37.\nThe subject_id is 10039694. The hadm_id is 20374452. The admittime is 2170-06-28 19:41:00. The dischtime is 2170-07-02 16:41:00. The deathtime is . The admission_type is URGENT. The admission_location is TRANSFER FROM HOSPITAL. The discharge_location is HOME. The insurance is Medicare. The language is ENGLISH. The marital_status is SINGLE. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.",
"""
patient_1_fhir = """ {"id": "745954f6-dfa2-5412-8c94-cf05931a8257", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_10039694"}], "gender": "female", "birthDate": "2134-06-28", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2106-3", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "White"}}, {"url": "text", "valueString": "White"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2186-5", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Not Hispanic or Latino"}}, {"url": "text", "valueString": "Not Hispanic or Latino"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}], "identifier": [{"value": "10039694", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "S", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}} """

patient_2_desc = """The subject_id of the patient is 10073847.\nThe gender of the patient is M.\nThe anchor_age of the patient is 53.\nThe anchor_year of the patient is 2134.\nThe anchor_year_group of the patient is 2011 - 2013.\nThe dod of the patient is .\nThe intime of the patient is 2134-02-24 09:47:50.\nThe subject_id is 10073847. The hadm_id is 20508747. The admittime is 2134-04-08 11:42:00. The dischtime is 2134-04-13 11:15:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 22194617. The admittime is 2135-12-27 18:51:00. The dischtime is 2136-02-11 06:41:00. The deathtime is 2136-02-11 06:41:00. The admission_type is DIRECT EMER.. The admission_location is CLINIC REFERRAL. The discharge_location is DIED. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is OTHER. The edregtime is . The edouttime is . The hospital_expire_flag is 1.\nThe subject_id is 10073847. The hadm_id is 26420408. The admittime is 2135-02-16 15:00:00. The dischtime is 2135-02-20 21:13:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 21266820. The admittime is 2135-04-14 12:36:00. The dischtime is 2135-04-18 16:03:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 29425640. The admittime is 2135-06-23 13:23:00. The dischtime is 2135-06-27 20:22:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is CLINIC REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 28366015. The admittime is 2135-03-02 15:58:00. The dischtime is 2135-03-07 16:20:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 24257242. The admittime is 2135-10-13 17:43:00. The dischtime is 2135-10-16 09:45:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is OTHER. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 28332330. The admittime is 2134-04-29 10:56:00. The dischtime is 2134-05-04 02:00:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 24440597. The admittime is 2135-06-01 17:29:00. The dischtime is 2135-06-03 20:00:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is CLINIC REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 29545085. The admittime is 2135-04-04 16:48:00. The dischtime is 2135-04-08 19:20:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 21298087. The admittime is 2135-01-31 04:13:00. The dischtime is 2135-02-05 13:00:00. The deathtime is . The admission_type is EW EMER.. The admission_location is EMERGENCY ROOM. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is 2135-01-31 00:48:00. The edouttime is 2135-01-31 05:31:00. The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 26416929. The admittime is 2135-05-02 14:48:00. The dischtime is 2135-05-05 17:15:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 24633575. The admittime is 2134-02-24 09:47:00. The dischtime is 2134-03-02 14:15:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 20458748. The admittime is 2135-05-10 12:32:00. The dischtime is 2135-05-14 17:39:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 25505474. The admittime is 2135-03-21 14:16:00. The dischtime is 2135-03-27 11:40:00. The deathtime is . The admission_type is EW EMER.. The admission_location is EMERGENCY ROOM. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 25544534. The admittime is 2134-06-14 12:52:00. The dischtime is 2134-06-18 21:00:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 27496246. The admittime is 2135-01-17 22:31:00. The dischtime is 2135-01-23 12:00:00. The deathtime is . The admission_type is EW EMER.. The admission_location is EMERGENCY ROOM. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is 2135-01-17 18:41:00. The edouttime is 2135-01-18 00:04:00. The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 23586255. The admittime is 2134-03-18 10:57:00. The dischtime is 2134-03-22 19:15:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 27988643. The admittime is 2135-11-11 00:00:00. The dischtime is 2135-12-14 16:19:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME HEALTH CARE. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is OTHER. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 22776224. The admittime is 2134-05-20 14:25:00. The dischtime is 2134-05-24 22:00:00. The deathtime is . The admission_type is ELECTIVE. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 10073847. The hadm_id is 22630133. The admittime is 2135-09-02 13:03:00. The dischtime is 2135-09-06 11:24:00. The deathtime is . The admission_type is DIRECT EMER.. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is OTHER. The edregtime is . The edouttime is . The hospital_expire_flag is 0."""
patient_2_fhir = """ {"id": "1b5bc42d-95ac-58d5-8912-97cae4636967", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_10073847"}], "gender": "male", "birthDate": "2081-02-24", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "UNK", "system": "http://terminology.hl7.org/CodeSystem/v3-NullFlavor", "display": "unknown"}}, {"url": "text", "valueString": "unknown"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "M"}], "identifier": [{"value": "10073847", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "M", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "deceasedDateTime": "2136-02-11", "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}} """

patient_3_desc = """The subject_id of the patient is 14986776.\nThe gender of the patient is M.\nThe anchor_age of the patient is 33.\nThe anchor_year of the patient is 2156.\nThe anchor_year_group of the patient is 2011 - 2013.\nThe dod of the patient is .\nThe intime of the patient is 2156-09-16 17:45:00.\nThe subject_id is 14986776. The hadm_id is 23966394. The admittime is 2157-03-25 15:15:00. The dischtime is 2157-03-25 21:45:00. The deathtime is . The admission_type is AMBULATORY OBSERVATION. The admission_location is PROCEDURE SITE. The discharge_location is . The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is HISPANIC/LATINO. The edregtime is 2157-03-24 19:50:00. The edouttime is 2157-03-24 23:39:00. The hospital_expire_flag is 0"""
patient_3_fhir = """ {"id": "90872263-8852-580b-893b-2c66f8653f38", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_14986776"}], "gender": "male", "birthDate": "2123-09-16", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2106-3", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "White"}}, {"url": "text", "valueString": "White"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2135-2", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Hispanic or Latino"}}, {"url": "text", "valueString": "Hispanic or Latino"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "M"}], "identifier": [{"value": "14986776", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "M", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}} """

patient_4_desc = """The subject_id of the patient is 10322234.\nThe gender of the patient is F.\nThe anchor_age of the patient is 53.\nThe anchor_year of the patient is 2122.\nThe anchor_year_group of the patient is 2008 - 2010.\nThe dod of the patient is .\nThe intime of the patient is 2122-02-24 21:30:00.\nThe subject_id is 10322234. The hadm_id is 29789116. The admittime is 2122-02-25 01:06:00. The dischtime is 2122-02-25 12:42:00. The deathtime is . The admission_type is EU OBSERVATION. The admission_location is EMERGENCY ROOM. The discharge_location is . The insurance is Other. The language is ?. The marital_status is MARRIED. The ethnicity is ASIAN. The edregtime is 2122-02-24 21:30:00. The edouttime is 2122-02-25 12:42:00. The hospital_expire_flag is 0."""

patient_4_fhir = """{"id": "2059aa39-7cf1-541c-bd8d-60f362553060", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_10322234"}], "gender": "female", "birthDate": "2069-02-24", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2028-9", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Asian"}}, {"url": "text", "valueString": "Asian"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2186-5", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Not Hispanic or Latino"}}, {"url": "text", "valueString": "Not Hispanic or Latino"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}], "identifier": [{"value": "10322234", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "maritalStatus": {"coding": [{"code": "M", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}"""

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
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content
def uploadFHIRData(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
patients = uploadFHIRData("MimicPatient.ndjson")

def get_patient_resource(identifier, patients):
    for patient in patients:
        for id_entry in patient.get("identifier", []):
            if id_entry.get("value") == identifier:
                return patient



print(get_patient_resource("14029832",patients))
patient_to_generate = """   "Patient Info:\nThe subject_id of the patient is 12427812.\nThe gender of the patient is F.\nThe anchor_age of the patient is 35.\nThe anchor_year of the patient is 2184.\nThe anchor_year_group of the patient is 2017 - 2019.\nThe dod of the patient is .\nThe intime of the patient is 2184-01-06 11:52:47.\nThe subject_id is 12427812. The hadm_id is 21593330. The admittime is 2184-01-06 11:51:00. The dischtime is 2184-01-10 11:45:00. The deathtime is . The admission_type is URGENT. The admission_location is PHYSICIAN REFERRAL. The discharge_location is HOME. The insurance is Other. The language is ENGLISH. The marital_status is . The ethnicity is UNKNOWN. The edregtime is . The edouttime is . The hospital_expire_flag is 0.\nThe subject_id is 12427812. The hadm_id is 23948770. The admittime is 2185-01-20 00:08:00. The dischtime is 2185-01-21 11:45:00. The deathtime is . The admission_type is EU OBSERVATION. The admission_location is PHYSICIAN REFERRAL. The discharge_location is . The insurance is Other. The language is ENGLISH. The marital_status is MARRIED. The ethnicity is WHITE. The edregtime is 2185-01-19 18:58:00. The edouttime is 2185-01-20 01:34:00. The hospital_expire_flag is 0.",
""" 

# prompt = f"""
#                           Task : You must generate a FHIR resource having the profile of the resource and the data serialized.
#                           Data Input : Its described in the following list :
#                             -Patient Profile : {fhir_profile}
#                             -Data Serialized_P1 : {patient_1_desc}
#                             -Resource_P1 : {patient_1_fhir}
#                             -Data Serialized_P2 : {patient_2_desc}
#                             -Resource_P2 : {patient_2_fhir}  
#                             -Data Serialized_P3 : {patient_3_desc}
#                             -Resource_P3 : {patient_3_fhir}
#                             -Data Serialized_Objetive : {patient_to_generate}
#                           Output : You Must to answer only the FHIR resource generated in JSON format.
#                           Recommendations : Patient birthdate was estimated by taking the anchor_age and subtracting the intime field.
#                                             If a Patient have more than one hamd_id, you must to take the information about the latest admission to fill following fields : ombCategory, maritalStatus and language.      
#                                             You have always to add if the patient is Hispanic or Latino or if is Not Hispanic or Latino with the respective code.
#                            Indications : If you can´t answer the whole resource because of the maximum of tokens, stop the task. Never answer the resource at half.
#                           """

#fhir_generated = get_completion(prompt)
#print(fhir_generated)

# Ruta del archivo CSV
#ruta_archivo = 'archivo.csv'

# Cargar el archivo CSV en un DataFrame de pandas
#datos = pd.read_csv("resultados_fase_1.csv")

#print(datos["Error API"])