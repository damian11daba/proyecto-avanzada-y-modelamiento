"""
Módulo de Preprocesamiento de Texto y Construcción de la Matriz Término-Documento.
Asignatura: Álgebra Lineal Avanzada y Modelamiento (IMT2230) - UC.

Implementa los Pasos 3 y 4 de la guía del proyecto:
- Limpieza y normalización de texto.
- Tokenización y filtrado de stopwords en inglés y términos de baja/alta frecuencia.
- Ponderación TF-IDF sublineal para reducir el sesgo por longitud de documento.
- Construcción y transposición de la matriz término-documento A de tamaño (m x n).
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopwords de dominio académico general que no aportan discriminación semántica
DOMAIN_STOPWORDS = [
    'paper', 'propose', 'proposed', 'approach', 'method', 'methods', 'results',
    'based', 'using', 'study', 'presents', 'show', 'demonstrate', 'et', 'al',
    'also', 'new', 'different', 'provide', 'use', 'used', 'article', 'author',
    'authors', 'work', 'present', 'presents', 'shown', 'various', 'given'
]

def clean_text(text: str) -> str:
    """Limpia caracteres especiales, saltos de línea y normaliza a minúsculas."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Conservar únicamente letras y espacios (eliminar dígitos y puntuación suelta)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_term_document_matrix(
    data_path: str = "data/corpus.csv",
    min_df: int = 3,
    max_df: float = 0.60,
    sublinear_tf: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, TfidfVectorizer]:
    """
    Construye la matriz término-documento A de tamaño (m x n) a partir del corpus.
    
    Retorna:
    --------
    A : np.ndarray de tamaño (m, n)
        Matriz donde las filas representan términos (m) y las columnas documentos (n).
    vocab : np.ndarray de strings de tamaño (m,)
        Vocabulario de términos ordenado alfabéticamente.
    df : pd.DataFrame
        DataFrame con los metadatos de los n documentos.
    vectorizer : TfidfVectorizer
        Instancia ajustada de scikit-learn.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en {data_path}. Ejecuta download_corpus.py primero.")
        
    df = pd.read_csv(data_path)
    
    # Combinar stopwords estándar de inglés con stopwords de dominio
    base_stopwords = list(TfidfVectorizer(stop_words='english').get_stop_words())
    combined_stopwords = list(set(base_stopwords + DOMAIN_STOPWORDS))
    
    # Configuración de TF-IDF:
    # - sublinear_tf=True: aplica 1 + log(tf), suavizando términos muy repetidos
    # - token_pattern: palabras alfabéticas de longitud >= 3
    # - min_df: descarta términos que aparecen en menos de min_df documentos (reduce ruido)
    # - max_df: descarta términos que aparecen en más del 60% de documentos (reduce ubicuidad)
    vectorizer = TfidfVectorizer(
        stop_words=combined_stopwords,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
    )
    
    # scikit-learn genera X con shape (n_documentos, m_términos)
    X = vectorizer.fit_transform(df['abstract'])
    
    # Paso 4: Transponer para obtener A de tamaño (m_términos x n_documentos)
    A = X.T.toarray()
    vocab = np.array(vectorizer.get_feature_names_out())
    
    m, n = A.shape
    assert m == len(vocab), "El número de filas debe coincidir con el tamaño del vocabulario."
    assert n == len(df), "El número de columnas debe coincidir con el número de documentos."
    
    print(f"Preprocesamiento completado con éxito:")
    print(f" - Tamaño del vocabulario m (filas): {m} términos")
    print(f" - Cantidad de documentos n (columnas): {n} documentos")
    print(f" - Dimensiones de la matriz textual A: {A.shape}")
    print(f" - Densidad de la matriz: {100 * np.count_nonzero(A) / (m * n):.2f}% de elementos no nulos")
    
    return A, vocab, df, vectorizer

if __name__ == "__main__":
    A, vocab, df, vectorizer = build_term_document_matrix()
