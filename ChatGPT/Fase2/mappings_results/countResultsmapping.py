import pandas as pd
# Cargar el archivo CSV
file_path = 'C:/Users/Alvaro/ChatGPT/ChatGPT/Fase3/mapeos/mapeosReflexiveSerial_NoSchema/'
file_name = 'mapeosReflexiveSerial_NoSchema_results_ordered.csv'
df = pd.read_csv(file_path+file_name)

def count_results_gpt(df):
    content_gpt_yes = df['CorrectGPT'].value_counts().get('YES', 0)
    content_gpt_no = df['CorrectGPT'].value_counts().get('NO', 0)
    #content_gpt_no_mapea = df['GPT'].value_counts().get('No mapea', 0)
    total_gpt = content_gpt_yes + content_gpt_no
    acierto_gpt = (content_gpt_yes / total_gpt) * 100 if total_gpt > 0 else 0

        #Mostrar resultados
    print("######Resultados : "+file_name+"#######")
    print(f'ContentGPT - Métrica de acierto: {acierto_gpt:.2f}%')
    print(f'ContentGPT YES: {content_gpt_yes}')
    print(f'ContentGPT NO: {content_gpt_no}')
    #print(f'ContentGPT No direct mapping: {content_gpt_no_direct_mapping}')
    #print(f'ContentGPT No mapea: {content_gpt_no_mapea}')
    return 

def count_results_llama(df):
    content_llama_yes = df['CorrectLlama'].value_counts().get('YES', 0)
    content_llama_no = df['CorrectLlama'].value_counts().get('NO', 0)
    #content_llama_no_mapea = df['Llama'].value_counts().get('No mapea', 0)
    total_llama = content_llama_yes + content_llama_no
    acierto_llama = (content_llama_yes / total_llama) * 100 if total_llama > 0 else 0
    print("--------------------------")
    print(f'ContentLlama - Métrica de acierto: {acierto_llama:.2f}%')
    print(f'ContentLlama YES: {content_llama_yes}')
    print(f'ContentLlama NO: {content_llama_no}')
    #print(f'ContentLlama No direct mapping: {content_llama_no_direct_mapping}')
    #print(f'ContentLlama No mapea: {content_llama_no_mapea}')
    return 


# Contar resultados

count_results_gpt(df)
count_results_llama(df)




