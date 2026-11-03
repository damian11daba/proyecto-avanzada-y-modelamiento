"""
Módulo de Exploración Semántica en el Espacio Latente (LSA).
Asignatura: Álgebra Lineal Avanzada y Modelamiento (IMT2230) - UC.

Implementa el Paso 8 de la guía del proyecto:
- Representación reducida de rango k para términos y documentos.
- Similitud coseno en el espacio latente.
- Búsqueda de vecinos más cercanos para términos (sinonimia y co-ocurrencia latente).
- Búsqueda de documentos cercanos e interrelaciones temáticas.
- Extracción e interpretación de componentes temáticas (tópicos latentes de U y V).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

def get_latent_representations(
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye las coordenadas latentes en dimensión reducida k.
    
    Términos: T_k = U_k @ Sigma_k in R^{m x k}
    Documentos: D_k = V_k @ Sigma_k in R^{n x k} (donde V = Vt.T)
    """
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vk = Vt.T[:, :k]
    
    term_embeddings = Uk @ Sk
    doc_embeddings = Vk @ Sk
    
    return term_embeddings, doc_embeddings

def find_nearest_terms(
    query_term: str,
    vocab: np.ndarray,
    term_embeddings: np.ndarray,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Encuentra los términos más cercanos a una palabra de consulta usando similitud coseno
    en el espacio latente reducido de dimensión k.
    """
    vocab_list = list(vocab)
    if query_term not in vocab_list:
        raise ValueError(f"El término '{query_term}' no se encuentra en el vocabulario.")
        
    idx = vocab_list.index(query_term)
    query_vec = term_embeddings[idx]
    
    # Normalización L2
    norms = np.linalg.norm(term_embeddings, axis=1)
    norms[norms == 0] = 1e-12
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        q_norm = 1e-12
        
    cosine_sims = (term_embeddings @ query_vec) / (norms * q_norm)
    
    # Ordenar descendente (excluyendo el término mismo en índice 0)
    sorted_indices = np.argsort(cosine_sims)[::-1]
    
    results = []
    for i in sorted_indices:
        if vocab[i] != query_term:
            results.append((vocab[i], float(cosine_sims[i])))
        if len(results) >= top_n:
            break
            
    return results

def find_nearest_documents(
    doc_index: int,
    df: pd.DataFrame,
    doc_embeddings: np.ndarray,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Encuentra los documentos más cercanos a un documento de referencia usando similitud coseno.
    """
    if doc_index < 0 or doc_index >= len(df):
        raise IndexError(f"Índice de documento fuera de rango: {doc_index}")
        
    query_vec = doc_embeddings[doc_index]
    norms = np.linalg.norm(doc_embeddings, axis=1)
    norms[norms == 0] = 1e-12
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        q_norm = 1e-12
        
    cosine_sims = (doc_embeddings @ query_vec) / (norms * q_norm)
    sorted_indices = np.argsort(cosine_sims)[::-1]
    
    results = []
    for i in sorted_indices:
        if i != doc_index:
            row = df.iloc[i]
            results.append({
                'index': int(i),
                'doc_id': row['doc_id'],
                'category': row['category_name'],
                'title': row['title'],
                'similarity': float(cosine_sims[i])
            })
        if len(results) >= top_n:
            break
            
    return results

def extract_component_topics(
    U: np.ndarray,
    S: np.ndarray,
    vocab: np.ndarray,
    top_k_components: int = 4,
    words_per_pole: int = 8
) -> List[Dict[str, Any]]:
    """
    Extrae la interpretación semántica de los vectores singulares izquierdos u_k,
    analizando las palabras con mayores cargas positivas y negativas.
    """
    topics = []
    for c in range(top_k_components):
        u_col = U[:, c]
        sigma = float(S[c])
        
        # Polo positivo
        top_pos_idx = np.argsort(u_col)[-words_per_pole:][::-1]
        top_pos = [(vocab[i], float(u_col[i])) for i in top_pos_idx]
        
        # Polo negativo
        top_neg_idx = np.argsort(u_col)[:words_per_pole]
        top_neg = [(vocab[i], float(u_col[i])) for i in top_neg_idx]
        
        topics.append({
            'component': c + 1,
            'singular_value': sigma,
            'positive_pole': top_pos,
            'negative_pole': top_neg
        })
    return topics

if __name__ == "__main__":
    from preprocessing import build_term_document_matrix
    from svd_analysis import compute_svd
    
    A, vocab, df, _ = build_term_document_matrix()
    U, S, Vt = compute_svd(A)
    term_emb, doc_emb = get_latent_representations(U, S, Vt, k=20)
    
    print("\nEjemplos de Búsqueda Semántica:")
    for word in ['neural', 'galaxy', 'gene', 'portfolio', 'quantum']:
        neighbors = find_nearest_terms(word, vocab, term_emb, top_n=4)
        print(f"\nPalabras más cercanas a '{word}':")
        for w, sim in neighbors:
            print(f"  - {w:15s} (sim: {sim:.3f})")
