def compare_jsons(json1, json2, parent_key='', errors=None):
    """
    Recursively compare two JSON objects and return a list of differences.
    """
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
            errors.append(f"The length of list '{parent_key}' in JSON 1 is {len1}, while in JSON 2 is {len2}")
        else:
            # Recursively compare elements of lists
            for i in range(len(json1)):
                new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                compare_jsons(json1[i], json2[i], parent_key=new_key, errors=errors)
    else:
        # Compare values
        if json1 != json2:
            errors.append(f"'{parent_key}': FHIR Generated value is '{json1}', JSON FHIR Example value is '{json2}'")

    return errors

# Ejemplo de uso
import json

# JSONs de ejemplo
json_str1 = '''
{  \"id\": \"f5f6a5f4-4f6a-4f6a-4f6a-4f6a4f6a4f6a\",  \"meta\": {    \"profile\": [\"http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient\"]  },  \"name\": [    {      \"use\": \"official\",      \"family\": \"Patient_13831972\"    }  ],  \"gender\": \"female\",  \"birthDate\": \"2084-08-31\",  \"extension\": [    {      \"url\": \"http://hl7.org/fhir/us/core/StructureDefinition/us-core-race\",      \"extension\": [        {          \"url\": \"ombCategory\",          \"valueCoding\": {            \"code\": \"2106-3\",            \"system\": \"urn:oid:2.16.840.1.113883.6.238\",            \"display\": \"White\"          }        },        {          \"url\": \"text\",          \"valueString\": \"White\"        }      ]    },    {      \"url\": \"http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity\",      \"extension\": [        {          \"url\": \"ombCategory\",          \"valueCoding\": {            \"code\": \"2186-5\",            \"system\": \"urn:oid:2.16.840.1.113883.6.238\",            \"display\": \"Not Hispanic or Latino\"          }        },        {          \"url\": \"text\",          \"valueString\": \"Not Hispanic or Latino\"        }      ]    },    {      \"url\": \"http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex\",      \"valueCode\": \"F\"    }  ],  \"identifier\": [    {      \"value\": \"13831972\",      \"system\": \"http://mimic.mit.edu/fhir/mimic/identifier/patient\"    }  ],  \"resourceType\": \"Patient\",  \"communication\": [    {      \"language\": {        \"coding\": [          {            \"code\": \"en\",            \"system\": \"urn:ietf:bcp:47\"          }        ]      }    }  ],  \"maritalStatus\": {    \"coding\": [      {        \"code\": \"S\",        \"system\": \"http://terminology.hl7.org/CodeSystem/v3-MaritalStatus\"      }    ]  },  \"admissions\": [    {      \"subject_id\": \"13831972\",      \"hadm_id\": \"25708445\",      \"admittime\": \"2130-11-22 22:10:00\",      \"dischtime\": \"2130-11-23 17:55:00\",      \"deathtime\": \"\",      \"admission_type\": \"EW EMER.\",      \"admission_location\": \"EMERGENCY ROOM\",      \"discharge_location\": \"HOME\",      \"insurance\": \"Medicaid\",      \"language\": \"ENGLISH\",      \"marital_status\": \"SINGLE\",      \"ethnicity\": \"WHITE\",      \"edregtime\": \"2130-11-22 14:30:00\",      \"edouttime\": \"2130-11-22 23:16:00\",      \"hospital_expire_flag\": \"0\"    }  ],  \"managingOrganization\": {    \"reference\": \"Organization/ee172322-118b-5716-abbc-18e4c5437e15\"  }}'''

json_str2 = '''
{"id": "0c66d85e-1318-560b-9527-2471c13e0bf4", "meta": {"profile": ["http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"]}, "name": [{"use": "official", "family": "Patient_13831972"}], "gender": "female", "birthDate": "2083-08-31", "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2106-3", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "White"}}, {"url": "text", "valueString": "White"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", "extension": [{"url": "ombCategory", "valueCoding": {"code": "2186-5", "system": "urn:oid:2.16.840.1.113883.6.238", "display": "Not Hispanic or Latino"}}, {"url": "text", "valueString": "Not Hispanic or Latino"}]}, {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}], "identifier": [{"value": "13831972", "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"}], "resourceType": "Patient", "communication": [{"language": {"coding": [{"code": "en", "system": "urn:ietf:bcp:47"}]}}], "maritalStatus": {"coding": [{"code": "S", "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"}]}, "deceasedDateTime": "2135-10-21", "managingOrganization": {"reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"}}

'''

# Parsear los JSON
data1 = json.loads(json_str1)
data2 = json.loads(json_str2)

# Comparar los JSON
errores = compare_jsons(data1, data2)
print(errores)

