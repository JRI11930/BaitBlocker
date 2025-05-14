"""
Proyecto de Detección de Clickbait - Módulo de Preprocesamiento y Normalización

Este script implementa diferentes técnicas de normalización para procesar el texto
de titulares potencialmente clickbait, preparándolo para análisis posteriores.

Técnicas implementadas:
1. Tokenización + eliminación de stopwords
2. Tokenización + stopwords + lematización
3. Todas las técnicas juntas (tokenización + limpieza + lematización + stopwords + stemming)

Fecha: 12/05/2025
"""

import os
import re
import sys
import pandas as pd
import nltk
import emoji
import logging
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nlp = spacy.load('es_core_news_sm')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Descargar recursos de NLTK necesarios
logger.info("Descargando recursos de NLTK...")
nltk.download('punkt')
nltk.download('stopwords')

# Inicializar recursos de NLTK
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))

OUTPUT_DIR = "../outputs/normalized/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def limpiar_texto_basico(texto):
    """
    Realiza limpieza básica del texto:
    - Elimina URLs
    - Elimina menciones (@usuario)
    - Elimina emojis
    - Elimina símbolos que no aportan al análisis
    
    Argumentos:
        texto (str): Texto a limpiar
        
    Salidas:
        str: Texto limpio
    """
    if not isinstance(texto, str):
        return ""
    
    # Eliminar URLs
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    
    # Eliminar menciones (@usuario)
    texto = re.sub(r'@\w+', '', texto)
    
    # Eliminar emojis
    texto = emoji.replace_emoji(texto, replace='')
    
    # Eliminar símbolos irrelevantes pero conservar signos importantes para el clickbait
    texto = re.sub(r'[^\w\s\?\!\.\,\:\;\…]', '', texto)
    
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def tokenizar_texto(texto):
    """
    Tokeniza el texto en palabras
    
    Argumentos:
        texto (str): Texto a tokenizar
        
    Salidas:
        list: Lista de tokens
    """
    if not isinstance(texto, str):
        return []
    
    return word_tokenize(texto)

def eliminar_stopwords(tokens):
    """
    Elimina stopwords de una lista de tokens
    
    Argumentos:
        tokens (list): Lista de tokens
        
    Salidas:
        list: Lista de tokens sin stopwords
    """
    return [token for token in tokens if token.lower() not in stop_words]

def lematizar_texto(texto):
    """
    Lematiza el texto usando spaCy
    
    Argumentos:
        texto (str): Texto a lematizar
        
    Salidas:
        list: Lista de lemas
    """
    if not isinstance(texto, str):
        return []
    
    doc = nlp(texto)
    return [token.lemma_ for token in doc]

def aplicar_stemming(tokens):
    """
    Aplica stemming a una lista de tokens
    
    Argumentos:
        tokens (list): Lista de tokens
        
    Salidas:
        list: Lista de stems
    """
    return [stemmer.stem(token) for token in tokens]

def normalizar_token_stop(texto):
    """
    Aplica tokenización y eliminación de stopwords
    
    Argumentos:
        texto (str): Texto a normalizar
        
    Salidas:
        str: Texto normalizado
    """
    tokens = tokenizar_texto(texto)
    tokens_sin_stop = eliminar_stopwords(tokens)
    return ' '.join(tokens_sin_stop)

def normalizar_token_stop_lemma(texto):
    """
    Aplica tokenización, eliminación de stopwords y lematización
    
    Argumentos:
        texto (str): Texto a normalizar
        
    Salidas:
        str: Texto normalizado
    """
    lemas = lematizar_texto(texto)
    lemas_sin_stop = [lema for lema in lemas if lema.lower() not in stop_words]
    return ' '.join(lemas_sin_stop)

def normalizar_all(texto):
    """
    Aplica todas las técnicas de normalización:
    - Limpieza básica
    - Tokenización
    - Lematización
    - Eliminación de stopwords
    - Stemming
    
    Argumentos:
        texto (str): Texto a normalizar
        
    Salidas:
        str: Texto normalizado
    """
    texto_limpio = limpiar_texto_basico(texto)
    lemas = lematizar_texto(texto_limpio)
    lemas_sin_stop = [lema for lema in lemas if lema.lower() not in stop_words]
    stems = aplicar_stemming(lemas_sin_stop)
    return ' '.join(stems)

def procesar_dataset(input_path):
    """
    Procesa el dataset aplicando diferentes técnicas de normalización
    
    Argumentos:
        input_path (str): Ruta al archivo CSV de entrada
        
    Salidas:
        dict: Diccionario con los DataFrames resultantes
    """
    logger.info(f"Cargando dataset desde {input_path}")
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Dataset cargado correctamente. Forma: {df.shape}")
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        return {}
    
    resultados = {}
    
    # Verificar si la columna existe
    if 'Teaser Text' not in df.columns:
        logger.error("La columna 'Teaser Text' no existe en el dataset")
        return {}
    
    # Aplicar tokenización + eliminación de stopwords
    logger.info("Aplicando tokenización + eliminación de stopwords")
    df_token_stop = df.copy()
    df_token_stop['Teaser Text'] = df_token_stop['Teaser Text'].apply(normalizar_token_stop)
    resultados['token_stop'] = df_token_stop
    
    # Aplicar tokenización + stopwords + lematización
    logger.info("Aplicando tokenización + stopwords + lematización")
    df_token_stop_lemma = df.copy()
    df_token_stop_lemma['Teaser Text'] = df_token_stop_lemma['Teaser Text'].apply(normalizar_token_stop_lemma)
    resultados['token_stop_lemma'] = df_token_stop_lemma
    
    # Aplicar todas las técnicas
    logger.info("Aplicando todas las técnicas")
    df_all = df.copy()
    df_all['Teaser Text'] = df_all['Teaser Text'].apply(normalizar_all)
    resultados['all'] = df_all
    
    return resultados

def guardar_resultados(resultados):
    """
    Guarda los DataFrames resultantes en archivos CSV
    
    Argumentos:
        resultados (dict): Diccionario con los DataFrames resultantes
    """
    for tecnica, df in resultados.items():
        output_path = os.path.join(OUTPUT_DIR, f"normalized_{tecnica}.csv")
        logger.info(f"Guardando resultado de la técnica '{tecnica}' en {output_path}")
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Archivo guardado correctamente")
        except Exception as e:
            logger.error(f"Error al guardar el archivo: {e}")

def main():
    """
    Función principal del script
    """
    logger.info("Iniciando procesamiento de dataset para detección de clickbait")
    
    input_path = "../data/TA1C_dataset_detection_train.csv"
    
    # Verificar si el archivo existe
    if not os.path.exists(input_path):
        logger.error(f"El archivo {input_path} no existe")
        logger.info("Por favor, asegúrate de que el archivo se encuentra en la ruta correcta")
        sys.exit(1)
    
    resultados = procesar_dataset(input_path)
    
    if resultados:
        guardar_resultados(resultados)
        logger.info("Procesamiento completado con éxito")
    else:
        logger.error("No se pudo procesar el dataset")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)