"""
Módulo de Descomposición en Valores Singulares (SVD) y Proyecciones en Baja Dimensión.
Asignatura: Álgebra Lineal Avanzada y Modelamiento (IMT2230) - UC.

Implementa los Pasos 5, 6 y 7 de la guía del proyecto:
- Cálculo exacto de la SVD: A = U Sigma V^T.
- Análisis del espectro de valores singulares y decaimiento.
- Teorema de Eckart-Young-Mirsky y selección rigurosa de k.
- Proyección de términos en el plano Span(v_a, v_b) por la derecha.
- Proyección de documentos en el plano Span(u_a, u_b) por la izquierda.
- Verificación de la equivalencia algebraica de las proyecciones.
"""

import numpy as np
from typing import Tuple, Dict, Any

def compute_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la descomposición SVD compacta / reducida de la matriz A (m x n).
    
    A = U @ diag(S) @ Vt
    
    Parámetros:
    -----------
    A : np.ndarray de tamaño (m, n)
        Matriz término-documento (m términos, n documentos, m >= n).
        
    Retorna:
    --------
    U : np.ndarray de tamaño (m, n)
        Vectores singulares izquierdos (columnas ortonormales en R^m).
    S : np.ndarray de tamaño (n,)
        Valores singulares ordenados de mayor a menor (sigma_1 >= sigma_2 >= ... >= sigma_n >= 0).
    Vt : np.ndarray de tamaño (n, n)
        Vectores singulares derechos traspuestos (filas ortonormales en R^n).
        Nota: V = Vt.T tiene dimensiones (n, n), con columnas ortonormales v_1, ..., v_n.
    """
    m, n = A.shape
    # Usamos full_matrices=False para obtener la descomposición reducida (tamaño n en lugar de m)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    print("\nDescomposición SVD calculada:")
    print(f" - Matriz A: ({m}, {n})")
    print(f" - Matriz U (vectores singulares por la izquierda): {U.shape}")
    print(f" - Vector S (valores singulares sigma_r): {S.shape}")
    print(f" - Matriz V^T (vectores singulares por la derecha traspuestos): {Vt.shape}")
    print(f" - Matriz V: {Vt.T.shape}")
    
    # Verificación de ortonormalidad numérica
    ortho_U_err = np.max(np.abs(U.T @ U - np.eye(n)))
    ortho_V_err = np.max(np.abs(Vt @ Vt.T - np.eye(n)))
    recon_err = np.linalg.norm(A - (U * S) @ Vt, ord='fro')
    
    print(f" - Error de ortonormalidad U^T U - I: {ortho_U_err:.2e}")
    print(f" - Error de ortonormalidad V^T V - I: {ortho_V_err:.2e}")
    print(f" - Error de reconstrucción Frobenius ||A - U Sigma V^T||_F: {recon_err:.2e}")
    
    return U, S, Vt

def compute_singular_spectrum_metrics(S: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calcula métricas clave del espectro de valores singulares:
    - Varianza (energía) explicada individual: sigma_i^2 / sum(sigma_j^2)
    - Varianza acumulada: sum_{j=1}^k sigma_j^2 / sum(sigma_j^2)
    - Error teórico de Eckart-Young: sqrt(sum_{j=k+1}^n sigma_j^2)
    """
    total_energy = np.sum(S ** 2)
    var_individual = (S ** 2) / total_energy
    var_cumulative = np.cumsum(var_individual)
    
    # Error residual de rango k bajo norma Frobenius según Eckart-Young
    n = len(S)
    frobenius_residual = np.zeros(n)
    for k in range(1, n + 1):
        frobenius_residual[k - 1] = np.sqrt(np.sum(S[k:] ** 2))
        
    return {
        'singular_values': S,
        'var_individual': var_individual,
        'var_cumulative': var_cumulative,
        'frobenius_residual': frobenius_residual
    }

def project_terms_2d(
    A: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    comp_a: int = 1,
    comp_b: int = 2
) -> Tuple[np.ndarray, float]:
    """
    Proyecta las filas de A (términos) en el subespacio Span(v_a, v_b) generado
    por los vectores singulares derechos v_a y v_b (Paso 7 de la pauta).
    
    Fundamento algebraico:
    ----------------------
    Cada fila i de A es r_i^T = e_i^T A en R^{1 x n}.
    Proyectada sobre la base ortonormal {v_a, v_b}:
      coord_x(i) = <r_i, v_a> = (e_i^T A) v_a = e_i^T (A v_a) = e_i^T (sigma_a u_a) = sigma_a * u_a(i)
      coord_y(i) = <r_i, v_b> = (e_i^T A) v_b = e_i^T (A v_b) = e_i^T (sigma_b u_b) = sigma_b * u_b(i)
    
    En forma matricial:
      Coords = A @ [v_a, v_b] == [sigma_a * u_a, sigma_b * u_b]
      
    Parámetros:
    -----------
    comp_a, comp_b : int (1-indexed, ej. 1 para componente 1, 2 para componente 2)
    
    Retorna:
    --------
    coords : np.ndarray de tamaño (m, 2)
        Coordenadas 2D de los m términos en la base (v_a, v_b).
    discrepancy : float
        Diferencia absoluta máxima entre proyección directa (A @ v) y producto singular (sigma * u).
    """
    idx_a, idx_b = comp_a - 1, comp_b - 1
    V = Vt.T
    v_a, v_b = V[:, idx_a], V[:, idx_b]
    
    # 1. Proyección explícita por producto interno con la base de Span(v_a, v_b)
    coords_direct = np.column_stack([A @ v_a, A @ v_b])
    
    # 2. Equivalente analítico directo derivado de SVD
    coords_analytical = np.column_stack([S[idx_a] * U[:, idx_a], S[idx_b] * U[:, idx_b]])
    
    discrepancy = float(np.max(np.abs(coords_direct - coords_analytical)))
    return coords_direct, discrepancy

def project_documents_2d(
    A: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    comp_a: int = 1,
    comp_b: int = 2
) -> Tuple[np.ndarray, float]:
    """
    Proyecta las columnas de A (documentos) en el subespacio Span(u_a, u_b) generado
    por los vectores singulares izquierdos u_a y u_b (Paso 7 de la pauta).
    
    Fundamento algebraico:
    ----------------------
    Cada columna j de A es c_j = A e_j en R^{m x 1}.
    Proyectada sobre la base ortonormal {u_a, u_b}:
      coord_x(j) = <u_a, c_j> = u_a^T (A e_j) = (u_a^T A) e_j = (sigma_a v_a^T) e_j = sigma_a * v_a(j)
      coord_y(j) = <u_b, c_j> = u_b^T (A e_j) = (u_b^T A) e_j = (sigma_b v_b^T) e_j = sigma_b * v_b(j)
    
    En forma matricial:
      Coords = [u_a, u_b]^T @ A == [sigma_a * v_a^T; sigma_b * v_b^T]
      (al transponer, cada fila j es la coordenada 2D del documento j).
      
    Parámetros:
    -----------
    comp_a, comp_b : int (1-indexed)
    
    Retorna:
    --------
    coords : np.ndarray de tamaño (n, 2)
        Coordenadas 2D de los n documentos en la base (u_a, u_b).
    discrepancy : float
        Diferencia absoluta máxima entre proyección directa (u^T @ A) y producto singular (sigma * v).
    """
    idx_a, idx_b = comp_a - 1, comp_b - 1
    u_a, u_b = U[:, idx_a], U[:, idx_b]
    V = Vt.T
    
    # 1. Proyección explícita por producto interno con la base de Span(u_a, u_b)
    coords_direct = np.vstack([u_a @ A, u_b @ A]).T
    
    # 2. Equivalente analítico directo derivado de SVD
    coords_analytical = np.column_stack([S[idx_a] * V[:, idx_a], S[idx_b] * V[:, idx_b]])
    
    discrepancy = float(np.max(np.abs(coords_direct - coords_analytical)))
    return coords_direct, discrepancy

if __name__ == "__main__":
    from preprocessing import build_term_document_matrix
    A, vocab, df, _ = build_term_document_matrix()
    U, S, Vt = compute_svd(A)
    term_coords, t_err = project_terms_2d(A, U, S, Vt, comp_a=2, comp_b=3)
    doc_coords, d_err = project_documents_2d(A, U, S, Vt, comp_a=2, comp_b=3)
    print(f"Verificación de proyección de términos: error max = {t_err:.2e}")
    print(f"Verificación de proyección de documentos: error max = {d_err:.2e}")
