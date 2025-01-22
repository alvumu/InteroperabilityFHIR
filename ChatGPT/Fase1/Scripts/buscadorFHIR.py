import json


def uploadFHIRData(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

patients = uploadFHIRData("MimicPatient.ndjson")
# Función para obtener el recurso de un paciente por su identificador
def get_patient_resource(identifier):
    for patient in patients:
        for id_entry in patient.get("identifier", []):
            if id_entry.get("value") == identifier:
                return patient

# Ejemplo de uso:
identifier = "10176087"
fhir_generated = """ {
    "id": "c1a51556-8f1b-5ca1-a8b5-ece5e7bb8602",
    "meta": {
        "profile": [
            "http://mimic.mit.edu/fhir/mimic/StructureDefinition/mimic-patient"
        ]
    },
    "name": [
        {
            "use": "official",
            "family": "Patient_10176087"
        }
    ],
    "gender": "male",
    "birthDate": "2048-07-11",
    "extension": [
        {
            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
            "extension": [
                {
                    "url": "ombCategory",
                    "valueCoding": {
                        "code": "2106-3",
                        "system": "urn:oid:2.16.840.1.113883.6.238",
                        "display": "White"
                    }
                },
                {
                    "url": "text",
                    "valueString": "White"
                }
            ]
        },
        {
            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
            "extension": [
                {
                    "url": "ombCategory",
                    "valueCoding": {
                        "code": "2186-5",
                        "system": "urn:oid:2.16.840.1.113883.6.238",
                        "display": "Not Hispanic or Latino"
                    }
                },
                {
                    "url": "text",
                    "valueString": "Not Hispanic or Latino"
                }
            ]
        },
        {
            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex",
            "valueCode": "M"
        }
    ],
    "identifier": [
        {
            "value": "10176087",
            "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient"
        }
    ],
    "resourceType": "Patient",
    "communication": [
        {
            "language": {
                "coding": [
                    {
                        "code": "en",
                        "system": "urn:ietf:bcp:47"
                    }
                ]
            }
        }
    ],
    "maritalStatus": {
        "coding": [
            {
                "code": "M",
                "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus"
            }
        ]
    },
    "managingOrganization": {
        "reference": "Organization/ee172322-118b-5716-abbc-18e4c5437e15"
    }
}"""
fhir_generated_dict = json.loads(fhir_generated) 
# Acceder al campo identifier y luego al campo value
identifier_value = fhir_generated_dict['identifier'][0]['value']

patient_resource_1 = get_patient_resource(identifier_value)
print(json.dumps(patient_resource_1, indent=4))
#patient_resource = get_patient_resource(identifier)
#print(json.dumps(patient_resource, indent=4))