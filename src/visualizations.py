"""
Módulo de Visualizaciones Científicas y Generación de Gráficos.
Asignatura: Álgebra Lineal Avanzada y Modelamiento (IMT2230) - UC.

Genera las figuras requeridas en los Pasos 6, 7 y 8 de la pauta:
- Espectro de valores singulares y varianza acumulada.
- Scree plot con selección y justificación del hiperparámetro k.
- Proyección 2D de documentos en Span(u_a, u_b).
- Proyección 2D de términos en Span(v_a, v_b).
- Biplot semántico dual (documentos y términos en el mismo plano latente).
- Cargas temáticas (loadings) de los vectores singulares.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

# Estilo visual académico sobrio y limpio
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

CATEGORY_COLORS = {
    'Inteligencia Artificial': '#2b5c8f',             # Azul acero
    'Astrofísica y Cosmología': '#7b2cbf',            # Púrpura cósmico
    'Genómica y Biología Computacional': '#2a9d8f',   # Verde azulado
    'Finanzas Cuantitativas': '#e76f51'               # Coral terracota
}

CATEGORY_MARKERS = {
    'Inteligencia Artificial': 'o',
    'Astrofísica y Cosmología': 's',
    'Genómica y Biología Computacional': '^',
    'Finanzas Cuantitativas': 'D'
}

def plot_singular_values_and_variance(
    S: np.ndarray,
    output_path: str = "figures/01_singular_values_decay.png",
    k_selected: int = 20
):
    """Genera gráfico del espectro singular (escala normal y logarítmica) y varianza acumulada."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(S)
    indices = np.arange(1, n + 1)
    
    total_energy = np.sum(S ** 2)
    cum_var = (np.cumsum(S ** 2) / total_energy) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Espectro singular
    ax1.plot(indices, S, color='#1f77b4', lw=2.2, label=r'Valor singular $\sigma_i$')
    ax1.axvline(x=k_selected, color='#d62728', ls='--', lw=1.8, label=f'Corte seleccionado $k={k_selected}$')
    ax1.set_title('Decaimiento del Espectro de Valores Singulares')
    ax1.set_xlabel(r'Índice de componente singular $i$')
    ax1.set_ylabel(r'Valor singular $\sigma_i$')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Inset en escala logarítmica para observar la ley de potencias
    ax1_inset = ax1.inset_axes([0.48, 0.45, 0.48, 0.48])
    ax1_inset.semilogy(indices, S, color='#2ca02c', lw=1.8)
    ax1_inset.axvline(x=k_selected, color='#d62728', ls='--', lw=1.4)
    ax1_inset.set_title('Escala semi-log (decaimiento)', fontsize=9)
    ax1_inset.set_xlabel('i', fontsize=8)
    ax1_inset.set_ylabel(r'$\log(\sigma_i)$', fontsize=8)
    ax1_inset.tick_params(labelsize=8)
    ax1_inset.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower left')
    
    # 2. Varianza acumulada
    ax2.plot(indices, cum_var, color='#ff7f0e', lw=2.2, label='Varianza acumulada (%)')
    ax2.axvline(x=k_selected, color='#d62728', ls='--', lw=1.8, label=f'$k={k_selected}$ ({cum_var[k_selected-1]:.1f}%)')
    
    # Puntos de referencia
    for k_ref in [4, 10, 20, 50]:
        ax2.scatter(k_ref, cum_var[k_ref-1], color='#333333', zorder=5, s=30)
        ax2.annotate(
            f'k={k_ref}: {cum_var[k_ref-1]:.1f}%',
            xy=(k_ref, cum_var[k_ref-1]),
            xytext=(k_ref + 3, cum_var[k_ref-1] - 4),
            fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray', lw=0.6),
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
        )
        
    ax2.set_title('Energía y Varianza Explicada Acumulada')
    ax2.set_xlabel(r'Número de componentes conservadas $k$')
    ax2.set_ylabel(r'Varianza explicada acumulada (%)')
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_path}")

def plot_scree_and_error(
    S: np.ndarray,
    output_path: str = "figures/02_scree_plot_k_selection.png",
    k_selected: int = 20
):
    """Genera gráfico de diferencias sucesivas de valores singulares y error residual de Frobenius."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(S)
    diffs = -np.diff(S)  # delta sigma_k
    
    frobenius_residual = np.zeros(n)
    for k in range(1, n + 1):
        frobenius_residual[k - 1] = np.sqrt(np.sum(S[k:] ** 2))
    norm_A = np.sqrt(np.sum(S ** 2))
    rel_error = (frobenius_residual / norm_A) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # 1. Tasa de cambio de valores singulares (método del codo)
    ax1.plot(np.arange(1, 41), diffs[:40], marker='o', color='#8c564b', lw=1.8, ms=5)
    ax1.axvline(x=4, color='#7f7f7f', ls=':', lw=1.5, label='k=4 (Macro-categorías)')
    ax1.axvline(x=k_selected, color='#d62728', ls='--', lw=1.8, label=f'k={k_selected} (Rango de estabilidad semántica)')
    ax1.set_title(r'Diferencias Sucesivas $\Delta \sigma_k = \sigma_k - \sigma_{k+1}$ (Primeras 40 comp.)')
    ax1.set_xlabel('k')
    ax1.set_ylabel(r'$\Delta \sigma_k$')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # 2. Error relativo de reconstrucción ||A - A_k||_F / ||A||_F
    ax2.plot(np.arange(1, n + 1), rel_error, color='#9467bd', lw=2.2)
    ax2.axvline(x=k_selected, color='#d62728', ls='--', lw=1.8, label=f'k={k_selected} (Error: {rel_error[k_selected-1]:.1f}%)')
    ax2.set_title(r'Error Relativo de Aproximación $\|A - A_k\|_F / \|A\|_F$')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Error relativo de Frobenius (%)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_path}")

def plot_documents_2d(
    doc_coords_12: np.ndarray,
    doc_coords_23: np.ndarray,
    df: pd.DataFrame,
    output_path: str = "figures/03_documents_2d_projection.png"
):
    """
    Visualiza la proyección 2D de documentos en Span(u_1, u_2) y Span(u_2, u_3).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    categories = df['category_name'].unique()
    
    # 1. Proyección en Span(u_1, u_2)
    for cat in categories:
        mask = df['category_name'] == cat
        ax1.scatter(
            doc_coords_12[mask, 0],
            doc_coords_12[mask, 1],
            color=CATEGORY_COLORS.get(cat, '#333'),
            marker=CATEGORY_MARKERS.get(cat, 'o'),
            s=45,
            alpha=0.85,
            edgecolors='white',
            linewidths=0.7,
            label=cat
        )
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_title('Proyección de Documentos en Span(u_1, u_2)')
    ax1.set_xlabel(r'Componente 1 ($u_1$ - Densidad / Escala Común)')
    ax1.set_ylabel(r'Componente 2 ($u_2$ - Astrofísica vs Resto)')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    # 2. Proyección en Span(u_2, u_3)
    for cat in categories:
        mask = df['category_name'] == cat
        ax2.scatter(
            doc_coords_23[mask, 0],
            doc_coords_23[mask, 1],
            color=CATEGORY_COLORS.get(cat, '#333'),
            marker=CATEGORY_MARKERS.get(cat, 'o'),
            s=50,
            alpha=0.85,
            edgecolors='white',
            linewidths=0.7,
            label=cat
        )
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.set_title('Proyección de Documentos en Span(u_2, u_3) [Ejes Semánticos Principales]')
    ax2.set_xlabel(r'Componente 2 ($u_2$ - Astrofísica y Cosmología)')
    ax2.set_ylabel(r'Componente 3 ($u_3$ - Finanzas Cuantitativas)')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    # Anotar cuadrantes temáticos
    ax2.text(0.70, 0.05, 'Sector Astrofísica', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', color='#7b2cbf', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e8ff', alpha=0.7, edgecolor='none'))
    ax2.text(0.05, 0.90, 'Sector Finanzas', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', color='#e76f51', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe8e1', alpha=0.7, edgecolor='none'))
    ax2.text(0.05, 0.10, 'Sector IA / Genómica', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', color='#2a9d8f', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f7f5', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_path}")

def plot_terms_2d(
    term_coords: np.ndarray,
    vocab: np.ndarray,
    output_path: str = "figures/04_terms_2d_projection.png"
):
    """
    Visualiza la proyección 2D de términos en Span(v_2, v_3) con etiquetas para términos representativos.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 9))
    
    x = term_coords[:, 0]
    y = term_coords[:, 1]
    
    # Nube general de términos en gris suave
    ax.scatter(x, y, color='#b0bec5', alpha=0.45, s=20, edgecolors='none', label='Otros términos del vocabulario')
    
    # Términos clave a resaltar y etiquetar
    highlight_terms = {
        'Astrofísica': (
            ['dark', 'galaxy', 'galaxies', 'mass', 'matter', 'cosmological', 'redshift', 'telescope', 'halo', 'stellar'],
            '#7b2cbf'
        ),
        'Finanzas': (
            ['portfolio', 'risk', 'market', 'investor', 'assets', 'return', 'trading', 'volatility', 'wealth', 'pricing'],
            '#e76f51'
        ),
        'Genómica': (
            ['gene', 'genes', 'cell', 'cells', 'expression', 'dna', 'rna', 'genomic', 'sequencing', 'biological'],
            '#2a9d8f'
        ),
        'Inteligencia Artificial': (
            ['agent', 'agents', 'learning', 'training', 'neural', 'task', 'tasks', 'reinforcement', 'deep', 'llm'],
            '#2b5c8f'
        ),
        'Puente / Transversal': (
            ['model', 'models', 'data', 'distribution', 'parameter', 'parameters', 'optimization', 'dynamics'],
            '#e63946'
        )
    }
    
    vocab_map = {word: i for i, word in enumerate(vocab)}
    
    for group_name, (words, color) in highlight_terms.items():
        gx, gy, glabels = [], [], []
        for w in words:
            if w in vocab_map:
                idx = vocab_map[w]
                gx.append(x[idx])
                gy.append(y[idx])
                glabels.append(w)
                
        ax.scatter(gx, gy, color=color, s=75, alpha=0.95, edgecolors='black', linewidths=0.6, label=group_name)
        
        for w, wx, wy in zip(glabels, gx, gy):
            offset_x = 0.015 if wx >= 0 else -0.015
            offset_y = 0.012 if wy >= 0 else -0.012
            ax.annotate(
                w,
                xy=(wx, wy),
                xytext=(wx + offset_x, wy + offset_y),
                fontsize=9.5,
                fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor=color, lw=0.7)
            )
            
    ax.axhline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.set_title('Proyección 2D de Términos en Span(v_2, v_3) [Plano Semántico Latente]')
    ax.set_xlabel(r'Coordenada en $v_2$ (Asociación con Astrofísica / Cosmología)')
    ax.set_ylabel(r'Coordenada en $v_3$ (Asociación con Finanzas Cuantitativas)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_path}")

def plot_semantic_biplot(
    doc_coords: np.ndarray,
    df: pd.DataFrame,
    term_coords: np.ndarray,
    vocab: np.ndarray,
    output_path: str = "figures/05_semantic_biplot.png"
):
    """
    Biplot conjunto: muestra documentos (puntos con forma/color según categoría)
    y vectores de términos directrices proyectados en el mismo espacio 2D.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 9.5))
    
    categories = df['category_name'].unique()
    for cat in categories:
        mask = df['category_name'] == cat
        ax.scatter(
            doc_coords[mask, 0],
            doc_coords[mask, 1],
            color=CATEGORY_COLORS.get(cat, '#333'),
            marker=CATEGORY_MARKERS.get(cat, 'o'),
            s=40,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.6,
            label=f"Docs: {cat}"
        )
        
    # Términos ancla para trazar flechas
    anchor_words = [
        ('galaxy', '#7b2cbf'), ('dark', '#7b2cbf'), ('mass', '#7b2cbf'),
        ('portfolio', '#e76f51'), ('risk', '#e76f51'), ('market', '#e76f51'),
        ('gene', '#2a9d8f'), ('cell', '#2a9d8f'), ('expression', '#2a9d8f'),
        ('agent', '#2b5c8f'), ('learning', '#2b5c8f'), ('training', '#2b5c8f'),
        ('model', '#e63946'), ('data', '#e63946')
    ]
    
    vocab_map = {word: i for i, word in enumerate(vocab)}
    for word, color in anchor_words:
        if word in vocab_map:
            idx = vocab_map[word]
            tx, ty = term_coords[idx, 0], term_coords[idx, 1]
            # Escalar ligeramente para visualización armoniosa con los documentos
            ax.annotate(
                '', xy=(tx, ty), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6, alpha=0.85)
            )
            ax.text(
                tx * 1.05, ty * 1.05, word,
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.9, edgecolor=color, lw=0.6)
            )
            
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title('Biplot Dual LSA: Correspondencia Documento-Término en Span(Componente 2, 3)')
    ax.set_xlabel('Eje Semántico 2 (Astrofísica)')
    ax.set_ylabel('Eje Semántico 3 (Finanzas Cuantitativas)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_path}")

def plot_topic_loadings(
    U: np.ndarray,
    S: np.ndarray,
    vocab: np.ndarray,
    output_path: str = "figures/06_topic_loadings.png"
):
    """
    Grafica las cargas (loadings) de los términos en las componentes 2, 3 y 4.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    
    components = [2, 3, 4]
    titles = [
        'Componente 2: Astrofísica (+) vs Resto (-)',
        'Componente 3: Finanzas (+) vs Bio/IA (-)',
        'Componente 4: IA (+) vs Genómica (-)'
    ]
    
    for ax, comp_num, title in zip(axes, components, titles):
        c_idx = comp_num - 1
        u_col = U[:, c_idx]
        
        # 6 mayores positivos y 6 mayores negativos
        top_pos_idx = np.argsort(u_col)[-6:]
        top_neg_idx = np.argsort(u_col)[:6]
        selected_idx = np.concatenate([top_neg_idx, top_pos_idx])
        
        words = vocab[selected_idx]
        weights = u_col[selected_idx]
        colors = ['#d9534f' if w < 0 else '#428bca' for w in weights]
        
        y_pos = np.arange(len(words))
        ax.barh(y_pos, weights, color=colors, edgecolor='black', linewidth=0.6, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10.5, fontweight='bold')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(title, fontsize=11.5, pad=10)
        ax.set_xlabel(r'Carga en el vector singular $u_{%d}$' % comp_num, fontsize=10.5)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {output_path}")

if __name__ == "__main__":
    from preprocessing import build_term_document_matrix
    from svd_analysis import compute_svd, project_documents_2d, project_terms_2d
    
    A, vocab, df, _ = build_term_document_matrix()
    U, S, Vt = compute_svd(A)
    
    # Métricas y figuras
    plot_singular_values_and_variance(S, k_selected=20)
    plot_scree_and_error(S, k_selected=20)
    
    doc_12, _ = project_documents_2d(A, U, S, Vt, comp_a=1, comp_b=2)
    doc_23, _ = project_documents_2d(A, U, S, Vt, comp_a=2, comp_b=3)
    plot_documents_2d(doc_12, doc_23, df)
    
    term_23, _ = project_terms_2d(A, U, S, Vt, comp_a=2, comp_b=3)
    plot_terms_2d(term_23, vocab)
    
    plot_semantic_biplot(doc_23, df, term_23, vocab)
    plot_topic_loadings(U, S, vocab)
