"""
APLICACIÓN STREAMLIT PARA DESPLIEGUE DEL MODELO DE PRECIOS DE VIVIENDAS
Esta aplicación permite a los usuarios interactuar con el modelo para predecir precios de viviendas
y visualizar el análisis exploratorio de datos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Datos de Viviendas",
    page_icon="🏠",
    layout="wide"
)

st.markdown("Aplicación interactiva para cargar datos, limpiar, analizar y modelar (clustering) viviendas.")

# 1. Recepción de datos ingresados por el usuario
st.sidebar.header("Carga de Datos 📥")
uploaded = st.sidebar.file_uploader("Sube un CSV de viviendas", type=["csv"])
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file)
    try:
        return pd.read_csv('housing_data.csv')
    except FileNotFoundError:
        st.error("No se encontró housing_data.csv en el directorio.")
        return None

df_raw = load_data(uploaded)

# 2. Limpieza de datos: función que aplica transformaciones y registra el log
def clean_data(df):
    log = []
    df_clean = df.copy()
    # Eliminación de duplicados
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    log.append(f"Eliminados {before - len(df_clean)} duplicados.")
    # Imputación de nulos numéricos con mediana
    num_cols = df_clean.select_dtypes(include=np.number).columns
    for col in num_cols:
        n_null = df_clean[col].isna().sum()
        if n_null:
            median = df_clean[col].median()
            df_clean[col].fillna(median, inplace=True)
            log.append(f"Imputados {n_null} nulos en '{col}' con mediana={median}.")
    # Codificación de categóricas
    cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
        log.append(f"Codificadas variables categóricas: {', '.join(cat_cols)}.")
    else:
        log.append("No se encontraron variables categóricas.")
    return df_clean, log

if df_raw is not None:
    df_clean, clean_log = clean_data(df_raw)
    # Mostrar log de limpieza
    st.sidebar.header("✅ Limpieza Aplicada")
    for entry in clean_log:
        st.sidebar.markdown(f"- {entry}")
else:
    df_clean = None

# Navegación de secciones
page = st.sidebar.radio("Sección", ["Inicio", "Limpieza", "Análisis", "Modelado", "Acerca de"])

# Inicio
if page == "Inicio":
    st.title("🏠 Explorador de Viviendas")
    st.write("Carga tus datos y navega por las diferentes etapas: limpieza, análisis y modelado.")

# Limpieza de datos
elif page == "Limpieza":
    st.header("🔧 Limpieza de Datos")
    if df_raw is None:
        st.error("No hay datos cargados.")
    else:
        st.subheader("Datos Originales")
        st.dataframe(df_raw.head())
        st.subheader("Datos Limpios")
        st.dataframe(df_clean.head())
        st.markdown("**Detalles de limpieza:**")
        for entry in clean_log:
            st.write(f"- {entry}")
        st.markdown("*Las decisiones buscan robustez (mediana vs outliers), evitar sesgo por duplicados y compatibilidad de datos.*")

# Análisis de los datos
elif page == "Análisis":
    st.header("📊 Análisis Exploratorio")
    if df_clean is None:
        st.error("No hay datos para analizar.")
    else:
        st.subheader("Estadísticas Descriptivas")
        st.write(df_clean.describe())
        st.subheader("Distribución de Precio")
        fig1, ax1 = plt.subplots()
        sns.histplot(df_clean['PRICE'], kde=True, ax=ax1)
        st.pyplot(fig1)
        st.subheader("Matriz de Correlación")
        corr = df_clean.corr()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', ax=ax2)
        st.pyplot(fig2)
        st.markdown("*Estas visualizaciones ayudan a identificar patrones y relaciones clave para comprender los datos.*")

# Modelado sobre los datos 1
elif page == "Modelado":
    st.header("🤖 Modelado: Clustering con KMeans")
    if df_clean is None:
        st.error("No hay datos para modelar.")
    else:
        # Selección de variables numéricas para clustering
        features = df_clean.select_dtypes(include=np.number).drop(columns=['PRICE'], errors='ignore')
        n_clusters = st.slider("Número de clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        features['Cluster'] = clusters
        st.subheader("Distribución de Clusters")
        st.write(features['Cluster'].value_counts())
        # PCA para visualización 2D
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features.drop(columns='Cluster'))
        fig3, ax3 = plt.subplots()
        scatter = ax3.scatter(coords[:,0], coords[:,1], c=clusters)
        st.pyplot(fig3)
        st.markdown(f"*Se aplicó KMeans con {n_clusters} clusters sobre variables numéricas. PCA permite visualizar agrupamientos.*")

# Acerca de
else:
    st.header("ℹ️ Acerca de esta App")
    st.markdown("Aplicación desarrollada en Streamlit para procesar datos de viviendas cargados por el usuario, realizar limpieza, análisis descriptivo y clustering con KMeans.")
