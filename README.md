# InteroperabilityFHIR

## Descripción

Este repositorio contiene las implementaciones y resultados de un proyecto orientado a la generación e identificación de recursos HL7 FHIR (Fast Healthcare Interoperability Resources) utilizando modelos de lenguaje proporcionados por OpenAI y Meta. El objetivo principal es explorar y automatizar la generación de datos FHIR y analizar la calidad de los recursos generados.


## Estructura del Proyecto

El proyecto está organizado en tres fases principales, cada una de las cuales contiene varias carpetas y ficheros relacionados con diferentes aspectos de la implementación.

### Fase 1 
En la primera fase, el objetivo principal es llevar a cabo la representación de distintos recursos HL7 FHIR conocidos de antemano por el LLM (_Patient_ y _Encounter_) a partir de datos en bruto estructurados.

En primer lugar, se aplica una serialización y un preprocesamiento a las tablas fuente para facilitar la comprensión de los datos al modelo. Seguidamente, se proporciona la información de las tablas seleccionadas al LLM, además del recurso al que tiene que hacer la transformación. Finalmente, tras realizar las distintas técnicas de prompting (0-Shot, 4-Shot e Iterative Prompting). Se analizan los resultados en términos de precisión y consistencia de las generaciones.

![image](https://github.com/user-attachments/assets/bd2497f4-2257-469b-b6a7-c6538d07f079)

### Fase 2
La Fase 2 se enfoca en una identificación de los atributos de distintos recursos HL7 FHIR. Inicialmente, se parte de un conjunto de datos tabulares (17 tablas), las correspondientes descripciones de las tablas y casos de uso. Estos datos tabulares pueden representar historias clínicas electrónicas, registros de pacientes y resultados de laboratorio, entre otros.

En primer lugar, se crea una estructura _RAG (Retrieval Augmented Generation)_ , en la cual se prueban y seleccionan distintos embeddings con el fin de aplicarlos a los datos de entrada y seleccionar el recurso HL7 FHIR correspondiente a cada una de las tabla de entrada.

Tras obtener el recurso HL7 FHIR más similar a cada una de las tablas de entrada, se procede a interactuar con el LLM. Para conseguir el desarrollo de esta tarea se estudia el comportamiento de distintas técnicas de prompting con el fin de seleccionar la que mejor rendimiento obtenga y la que mejor permita al modelo comprender y resolver la tarea de la mejor manera. 

Las respuestas devueltas por el modelo tras utilizar las distintas técnicas de prompting (_Self-Reflexive_ Mixture of Prompts (MoP), _5_Serial_) , son los distintos atributos del recurso HL7 FHIR más similar, a cada uno de los atributos de la tabla de información clínica de entrada, teniendo en cuenta su descripción y casos de uso.

![image](https://github.com/user-attachments/assets/2be725e1-dcaa-444e-9ce6-d31f442b3905)

### Fase 3 
La Fase 3 introduce un mayor nivel de complejidad al proceso de transformación. En este escenario, se ha pasado de tener 17 tablas con sus respectivas descripciones y casos de uso. A una única tabla de 67 atributos y situados de manera aleatoria, sin contexto sobre los mismos, es decir, carentes de la descripción de una tabla y también del caso de uso de la misma. Simulando situaciones en las que los datos provienen de múltiples fuentes o carecen de una estructura clara.

En primer lugar, se busca replicar los resultados obtenidos en la fase anterior en la que se tenía un mayor contexto de los datos iniciales. Para intentar aportar contexto sobre los datos se implementa la técnica de clustering, mediante la cual tras analizar y evaluar distintos algoritmos. Se selecciona el mejor algoritmo de clustering para el contexto del que partimos. Una vez obtenemos los distintos clusters y/o agrupaciones de los distintos atributos, se vuelve a implementar la técnica de RAG, la que, de nuevo es necesario llevar a cabo un estudio de los distintos tipos de embeddings para observar el comportamiento de estos con las distintas agrupaciones de datos. 
Finalmente tras el estudio y análisis, se seleccionan los embeddings que mejor encapsulan e identifican los distintos atributos de la tabla a los correspondientes recursos HL7 FHIR. Debido a que las agrupaciones pueden poseer atributos de diferentes recursos HL7 FHIR, se seleccionan los 5 recursos HL7 FHIR a cada cluster. Tras tener esta información, se procede a la interacción con el LLM, para la cual se emplea un prompt similar al que proporcionó el mejor rendimiento en la Fase 2. Esta fase pone a prueba la robustez y adaptabilidad del LLM en condiciones más cercanas a las situaciones reales en entornos clínicos.

![image](https://github.com/user-attachments/assets/c5473b16-1930-4ab7-bd27-88d6a63d8361)


## Fuente de Datos
 **MIMIC**(_Medical Information Mart for Intensive Care_) es una gran base de datos de libre acceso que contiene datos sanitarios no identificados de pacientes ingresados en las unidades de cuidados intensivos del _Beth Israel Deaconess Medical Center_. En este proyecto se ha utilizado concretamente la versión _MIMIC-IV_, este conjunto de datos contiene información en el tramo temporal de 2008 a 2019. Los datos se recopilaron a partir de monitores de cabecera _Metavision_.
Esta versión es una base de datos relacional que contiene estancias hospitalarias reales de pacientes ingresados en un centro médico académico terciario de \textit{Boston}, _MA (EE.UU._). _MIMIC-IV_ contiene información exhaustiva de cada paciente durante su estancia en el hospital: mediciones de laboratorio, medicamentos administrados, constantes vitales documentadas, etc. La base de datos está pensada para apoyar una amplia variedad de investigaciones en el ámbito de la asistencia sanitaria. _MIMIC-IV_ se basa en el éxito de _MIMIC-III_ e incorpora numerosas mejoras con respecto a otras versiones del conjunto de datos. 
**MIMIC-IV** se divide en "módulos" para reflejar la procedencia de los datos. 
Actualmente existen cinco módulos:
   - hosp: datos de nivel hospitalario para pacientes: laboratorios, micro y administración electrónica de medicación
   - icu: datos a nivel de UCI. Se trata de las tablas de eventos, y su estructura es idéntica a la de MIMIC-III (chartevents, etc.)
   - ed: datos del servicio de urgencias
   - cxr: tablas de búsqueda y metadatos de MIMIC-CXR, que permiten la vinculación con MIMIC-IV
   - note: notas clínicas de texto libre no identificadas

Para poder acceder a los datos es necesario seguir los pasos detallados en : https://mimic.mit.edu/docs/gettingstarted/
  

## Replicabilidad

Debido a la responsabilidad y privacidad de los datos originales no es posible añadirlos a este repositorio para convertirlo en replicable. Las políticas de responsabilidad de MIMIC no permiten la publicación de los conjuntos de datos de manera open-source. 

 
