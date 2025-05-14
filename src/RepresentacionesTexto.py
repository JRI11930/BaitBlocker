"""
Proyecto de Detección de Clickbait - Módulo de generación de representaciones vectoriales

Este script implementa diferentes técnicas para generar las representaciones de texto de 
archivos normalizados previamente de titulares potencialmente clickbait.

Técnicas implementadas:
1. N-gramas (1, 2 y 3)
2. Binario, frecuencia y TF-IDF

También se implementó Truncated SVD

Fecha: 13/05/2025
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directorios de entrada y salida
INPUT_DIR = "../outputs/normalized/"
OUTPUT_DIR = "../outputs/text-reps/"

# Crear directorio de salida si no existe
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Configuración de parámetros
NGRAM_RANGES = [(1, 1), (2, 2), (3, 3)]
NGRAM_NAMES = ['uni', 'bi', 'tri']
VECTORIZER_MODES = ['binary', 'freq', 'tfidf']
SVD_COMPONENTS = 50
RANDOM_STATE = 0

def get_vectorizer(mode, ngram_range):
    """
    Devuelve el vectorizador apropiado según el modo y el rango de n-gramas
    
    Args:
        mode (str): Modo de vectorización ('binary', 'freq', 'tfidf')
        ngram_range (tuple): Rango de n-gramas como tupla (n, n)
        
    Returns:
        Vectorizador configurado
    """
    if mode == 'binary':
        return CountVectorizer(ngram_range=ngram_range, binary=True)
    elif mode == 'freq':
        return CountVectorizer(ngram_range=ngram_range)
    elif mode == 'tfidf':
        return TfidfVectorizer(ngram_range=ngram_range)
    else:
        raise ValueError(f"Modo de vectorización no válido: {mode}")

def process_file(file_path):
    """
    Procesa un archivo, generando todas las representaciones vectoriales y aplicando SVD
    
    Args:
        file_path (str): Ruta al archivo CSV de entrada
    """
    try:
        # Obtener nombre base del archivo
        base_name = os.path.basename(file_path).split('.')[0]
        logger.info(f"Procesando {base_name}")
        
        # Leer datos
        df = pd.read_csv(file_path)
        logger.info(f"Leído {len(df)} filas de {file_path}")
        
        # Verificar columnas
        if 'Teaser Text' not in df.columns or 'Tag Value' not in df.columns:
            logger.error(f"Columnas requeridas no encontradas en {file_path}")
            return
        
        # Separar características y etiquetas
        X_text = df['Teaser Text'].fillna('').values
        y = df['Tag Value'].values
        
        # Iterar sobre configuraciones de n-gramas y modos de vectorización
        for ngram_idx, ngram_range in enumerate(NGRAM_RANGES):
            ngram_name = NGRAM_NAMES[ngram_idx]
            
            for mode in VECTORIZER_MODES:
                logger.info(f"Generando representación {ngram_name}_{mode} para {base_name}")
                
                # Obtener vectorizador
                vectorizer = get_vectorizer(mode, ngram_range)
                
                # Generar matriz de características
                logger.info("Aplicando vectorización...")
                X_vectorized = vectorizer.fit_transform(X_text)
                logger.info(f"Matriz generada con forma: {X_vectorized.shape}")
                
                # Aplicar TruncatedSVD para reducción de dimensionalidad
                logger.info("Aplicando TruncatedSVD...")
                svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
                X_svd = svd.fit_transform(X_vectorized)
                logger.info(f"Matriz reducida con forma: {X_svd.shape}")
                
                # Calcular varianza explicada
                explained_variance = svd.explained_variance_ratio_.sum()
                logger.info(f"Varianza explicada: {explained_variance:.4f}")
                
                # Crear el objeto de resultado para guardar
                result = {
                    'X': X_svd,
                    'y': y,
                    'vectorizer': vectorizer,
                    'svd': svd,
                    'explained_variance': explained_variance,
                    'config': {
                        'ngram_range': ngram_range,
                        'mode': mode,
                        'svd_components': SVD_COMPONENTS
                    }
                }
                
                # Definir nombre de archivo de salida
                output_filename = f"{base_name}_{ngram_name}_{mode}_svd{SVD_COMPONENTS}.pkl"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # Guardar resultado
                logger.info(f"Guardando resultado en {output_path}")
                with open(output_path, 'wb') as f:
                    pickle.dump(result, f)
                
                logger.info(f"Representación {output_filename} guardada exitosamente")
    
    except Exception as e:
        logger.error(f"Error procesando {file_path}: {str(e)}")
        raise

def main():
    """Función principal que procesa todos los archivos en el directorio de entrada"""
    logger.info(f"Iniciando procesamiento de archivos en {INPUT_DIR}")
    
    # Verificar que exista el directorio de entrada
    if not os.path.exists(INPUT_DIR):
        logger.error(f"El directorio de entrada {INPUT_DIR} no existe")
        return
    
    # Enumerar archivos CSV en el directorio de entrada
    csv_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
                 if f.endswith('.csv') and f.startswith('normalized_')]
    
    if not csv_files:
        logger.error(f"No se encontraron archivos CSV en {INPUT_DIR}")
        return
    
    logger.info(f"Se procesarán {len(csv_files)} archivos: {', '.join(os.path.basename(f) for f in csv_files)}")
    
    # Procesar cada archivo
    for file_path in csv_files:
        process_file(file_path)
    
    logger.info("Procesamiento completado")

if __name__ == "__main__":
    main()