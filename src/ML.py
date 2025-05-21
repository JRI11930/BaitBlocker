"""
Proyecto de Detección de Clickbait - Módulo de Machine Learning

Este script implementa diferentes modelos de machine learning sobre los archivos .pkl
generados en RepresentacionesTexto.py.

Modelos implementados:
1. Regresión Logística
2. Bosques Aleatorios
3. XGB Classifier

Fecha: 14/05/2025
"""

import pickle
import os
import pandas as pd
import numpy as np
import mlflow
import time
import logging
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

# Configuración del sistema de logging
def setup_logger():
    # Crear logger
    logger = logging.getLogger('clickbait_ml')
    logger.setLevel(logging.INFO)
    
    # Crear handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Crear handler para archivo
    file_handler = logging.FileHandler('clickbait_ml.log')
    file_handler.setLevel(logging.INFO)
    
    # Formato simple pero informativo
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Aplicar formato a los handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Añadir handlers al logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Inicializar logger
logger = setup_logger()

# Configuración del experimento MLflow
EXPERIMENT_NAME = "Clickbait_ML_Experiments"
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info(f"MLflow configurado: {EXPERIMENT_NAME}")

# Directorio donde se encuentran los archivos pkl
DATA_DIR = "../outputs/text-reps/"
logger.info(f"Directorio de datos: {DATA_DIR}")

# Definir los modelos a utilizar
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier()
}
logger.info(f"Modelos configurados: {', '.join(models.keys())}")

# Definir las métricas a calcular
scoring = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'accuracy': make_scorer(accuracy_score),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro')
}
logger.info(f"Métricas configuradas: {', '.join(scoring.keys())}")

# Función para cargar los datos
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        
        # Extraer X e y del diccionario
        X = result['X']
        y = result['y']
        
        # Podemos extraer la configuración para registro en MLflow
        config = result['config']
        
        return X, y, config
    except Exception as e:
        logger.error(f"Error al cargar los datos: {str(e)}")
        raise

# Función para aplicar LabelEncoder a las etiquetas
def encode_labels(y):
    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        # Guardar el mapeo de clases para referencia
        class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        logger.info(f"Etiquetas codificadas. Clases encontradas: {len(class_mapping)}")
        return y_encoded, label_encoder, class_mapping
    except Exception as e:
        logger.error(f"Error al codificar etiquetas: {str(e)}")
        raise

# Función para aplicar oversampling
def apply_oversampling(X, y):
    try:
        oversample = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X, y)
        
        return X_resampled, y_resampled
    except Exception as e:
        logger.error(f"Error en oversampling: {str(e)}")
        raise

# Función para entrenar y evaluar modelos con validación cruzada
def train_and_evaluate(X, y, model_name, model, dataset_name, config, class_mapping):
    
    # Configurar la validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        # Iniciar MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run iniciado: {run_id}")
            
            # Registrar el nombre del modelo y del dataset
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_name", dataset_name)
            
            # Registrar la configuración del dataset
            mlflow.log_param("ngram_range", str(config['ngram_range']))
            mlflow.log_param("mode", config['mode'])
            mlflow.log_param("svd_components", config['svd_components'])
            
            # Registrar el mapeo de clases
            for original_class, encoded_value in class_mapping.items():
                mlflow.log_param(f"class_{encoded_value}", original_class)
            
            # Medir el tiempo de ejecución
            start_time = time.time()
            
            # Realizar validación cruzada con múltiples métricas
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
            
            # Calcular tiempo total
            exec_time = time.time() - start_time
            
            # Registrar métricas promedio
            metrics_log = {}
            for metric_name, metric_values in cv_results.items():
                if metric_name.startswith('test_'):
                    # Solo registrar métricas de evaluación (no tiempos)
                    clean_name = metric_name.replace('test_', '')
                    metric_avg = np.mean(metric_values)
                    mlflow.log_metric(clean_name, metric_avg)
                    metrics_log[clean_name] = f"{metric_avg:.4f}"
            
            logger.info(f"MÉTRICAS REGISTRADAS: {metrics_log}")
            
            # Registrar tiempo de ejecución
            mlflow.log_metric("execution_time", exec_time)
            
            # Devolver resultados para el dataframe final
            result = {
                'dataset': dataset_name,
                'MLFlowID': run_id,
                'model': model_name,
                'ngram_range': str(config['ngram_range']),
                'mode': config['mode'],
                'svd_components': config['svd_components'],
                'num_classes': len(class_mapping),
                'f1_macro': np.mean(cv_results['test_f1_macro']),
                'accuracy': np.mean(cv_results['test_accuracy']),
                'precision_macro': np.mean(cv_results['test_precision_macro']),
                'recall_macro': np.mean(cv_results['test_recall_macro']),
                'execution_time': exec_time
            }
            
            logger.info(f"Evaluación de {model_name} completada con F1-macro: {result['f1_macro']:.4f}")
            return result
            
    except Exception as e:
        logger.error(f"Error durante entrenamiento/evaluación de {model_name}: {str(e)}")
        raise

def main():
    logger.info("=== INICIO DE EXPERIMENTOS DE ML PARA CLICKBAIT ===")
    
    # Lista para almacenar todos los resultados
    all_results = []
    
    try:
        # Obtener lista de archivos pkl
        pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
        
        logger.info(f"{len(pkl_files)} archivos pkl fueron encontrados para procesar")
        
        # Procesar cada archivo
        for i, pkl_file in enumerate(pkl_files):
            logger.info(f"Procesando archivo {i+1}/{len(pkl_files)}: {pkl_file}")
            
            # Cargar datos
            X, y, config = load_data(os.path.join(DATA_DIR, pkl_file))
            
            # Aplicar LabelEncoder a las etiquetas
            y_encoded, label_encoder, class_mapping = encode_labels(y)
            
            # Aplicar oversampling
            X_resampled, y_resampled = apply_oversampling(X, y_encoded)
            
            # Entrenar y evaluar cada modelo
            for model_name, model in models.items():
                logger.info(f"Iniciando entrenamiento de {model_name} para {pkl_file}")
                
                # Entrenar y evaluar modelo
                result = train_and_evaluate(X_resampled, y_resampled, model_name, model, pkl_file, config, class_mapping)
                
                # Añadir resultados
                all_results.append(result)
        
        # Crear dataframe con todos los resultados
        results_df = pd.DataFrame(all_results)
        
        # Ordenar por f1_macro descendente
        results_df = results_df.sort_values('f1_macro', ascending=False)
        
        # Guardar resultados en CSV
        output_file = '../outputs/ml_experiment_results.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"RESULTADOS GUARDADOS EN: {output_file}")
        
        # Mostrar los 10 mejores resultados
        print("Top 10 mejores combinaciones (por F1-macro):")
        for idx, row in results_df.head(10).iterrows():
            print(f"  {row['model']} - {row['dataset']} - f1: {row['f1_macro']:.4f} - acc: {row['accuracy']:.4f}")
        
        # Mostrar análisis agrupado para tener visión general de rendimiento
        print("Rendimiento promedio por tipo de modelo:")
        model_perf = results_df.groupby('model')['f1_macro'].mean().sort_values(ascending=False)
        for model, score in model_perf.items():
            print(f"  {model}: {score:.4f}")
        
        print("Rendimiento promedio por n-gramas:")
        ngram_perf = results_df.groupby('ngram_range')['f1_macro'].mean().sort_values(ascending=False)
        for ngram, score in ngram_perf.items():
            print(f"  {ngram}: {score:.4f}")
        
        print("Rendimiento promedio por modo de vectorización:")
        mode_perf = results_df.groupby('mode')['f1_macro'].mean().sort_values(ascending=False)
        for mode, score in mode_perf.items():
            print(f"  {mode}: {score:.4f}")
        
        print("=== EXPERIMENTOS COMPLETADOS CON ÉXITO ===")
        
        return results_df
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    main()