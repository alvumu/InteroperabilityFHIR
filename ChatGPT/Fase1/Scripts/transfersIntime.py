import pandas as pd

# Dataframe original
# Cargar datos desde el archivo transfers.csv en el mismo directorio
df = pd.read_csv("transfers.csv")

# Convertir la columna 'intime' a formato datetime
df['intime'] = pd.to_datetime(df['intime'])

# Crear un nuevo dataframe con una instancia del subject_id y la fecha más pequeña de intime para cada paciente
new_df = df.groupby('subject_id')['intime'].min().reset_index()
# Guardar el dataframe en un archivo JSON llamado "transfers_ordered.json"
new_df['intime'] = new_df['intime'].dt.strftime('%Y-%m-%d %H:%M:%S')

json_data = new_df.to_json("transfers_ordered.json",orient='records')
