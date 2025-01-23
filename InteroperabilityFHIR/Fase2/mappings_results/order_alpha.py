import pandas as pd

def sort_csv_by_attribute(input_file, output_file):
    """
    Ordena un archivo CSV por la columna 'attribute' en orden alfabético y guarda el resultado en un nuevo archivo.

    :param input_file: Ruta del archivo CSV de entrada.
    :param output_file: Ruta del archivo CSV de salida.
    """
    try:
        # Leer el archivo CSV
        df = pd.read_csv(input_file)

        # Ordenar el DataFrame por la columna 'attribute'
        df_sorted = df.sort_values(by='attribute')

        # Guardar el DataFrame ordenado en un nuevo archivo CSV
        df_sorted.to_csv(output_file, index=False)
        print(f"Archivo ordenado guardado en: {output_file}")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

# Ruta de los archivos
input_file = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/mapeos/mapeosReflexiveSerial_NoSchema/mapeosReflexiveSerial_NoSchema_results.csv"  # Cambia esto por la ruta de tu archivo CSV original
output_file = "C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/mapeos/mapeosReflexiveSerial_NoSchema/mapeosReflexiveSerial_NoSchema_results_ordered.csv"  # Cambia esto por la ruta donde quieres guardar el archivo ordenado

# Ejecutar la función
sort_csv_by_attribute(input_file, output_file)
