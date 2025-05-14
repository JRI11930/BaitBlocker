# 🚀 BaitBlocker

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/) [![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-yellow)](https://pandas.pydata.org/) [![NLTK](https://img.shields.io/badge/NLTK-3.6%2B-green)](https://www.nltk.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-FF6F00)](https://www.tensorflow.org/) [![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-0194E2)](https://mlflow.org/)

## 📋 Descripción

Este proyecto implementa un modelo de aprendizaje automático que clasifica textos como "clickbait" o "no clickbait".

## 🎯 Objetivo

Crear un modelo preciso de aprendizaje automático que pueda detectar automáticamente si un texto es clickbait o no, ayudando a los usuarios a identificar contenido de mayor calidad informativa.

## 🔍 Conjunto de datos

El sistema utiliza los siguientes conjuntos de datos:
- `TA1C_dataset_detection_train.csv`: Conjunto de entrenamiento
- `TA1C_dataset_detection_dev.csv`: Conjunto de prueba

## 📌 Características

- **Normalización de texto**: Tokenización, limpieza, eliminación de stopwords y lematización
- **Representaciones de texto avanzadas**: Unigramas, bigramas, trigramas y sus combinaciones

## 🛠️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/JRI11930/BaitBlocker.git
cd BaitBlocker

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 📊 Flujo de trabajo

1. **Carga de datos**: Cargar el conjunto de datos y extraer características (textos) y etiquetas (clickbait/no clickbait)
2. **Preprocesamiento**: Normalizar los textos mediante diversas técnicas
3. **Extracción de características**: Crear diferentes representaciones de texto

## 📂 Estructura del proyecto

```
BaitBlocker/
├── data/                      # Conjuntos de datos
├── src/                       # Código fuente refinado
    ├── Normalizacion.py
    └── RepresentacionesTexto.py
├── notebooks/                 # Notebooks para análisis rápidos
├── outputs/
    ├── normalized/
    └── text-reps/
├── requirements.txt
├── LICENSE
└── README.md                  # 
```

## 👤 Autor

José **Armando** Ramírez **Islas**

## 📄 Licencia

Este proyecto está licenciado bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.