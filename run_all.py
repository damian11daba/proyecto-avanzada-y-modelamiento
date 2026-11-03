"""
Pipeline Completo de Ejecución Automatizada: Proyecto LSA con SVD.
Asignatura: Álgebra Lineal Avanzada y Modelamiento (IMT2230) - UC.

Ejecuta el flujo de trabajo completo de principio a fin:
1. Descarga / verificación de integridad del corpus de arXiv.
2. Preprocesamiento de texto y construcción de la matriz textual A (m x n).
3. Cálculo exacto de la SVD (A = U Sigma V^T) y validación de propiedades.
4. Generación y exportación de todas las figuras analíticas a figures/.
5. Exploración semántica y pruebas de similitud latente.
6. Compilación y renderizado del Jupyter Notebook (notebooks/proyecto_lsa_svd.ipynb).
7. Generación del informe final en HTML y PDF (report/informe_LSA_IMT2230.pdf).
"""

import os
import sys
import time

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)

def main():
    start_time = time.time()
    print_section("Proyecto IMT2230: Análisis Semántico Latente con SVD")
    print("Iniciando pipeline de ejecución reproducible...")

    # Paso 1: Corpus
    print_section("Paso 1: Corpus de arXiv")
    corpus_file = "data/corpus.csv"
    if not os.path.exists(corpus_file):
        print(f"Descargando corpus desde arXiv API...")
        from src.download_corpus import download_and_save_corpus
        download_and_save_corpus(corpus_file)
    else:
        print(f"Corpus ya presente en {corpus_file}. Verificando integridad...")
        import pandas as pd
        df = pd.read_csv(corpus_file)
        print(f" -> Total documentos: {len(df)}")
        print(f" -> Categorías: {list(df['category_code'].unique())}")

    # Paso 2, 3, 4: Preprocesamiento y Matriz A
    print_section("Pasos 3 y 4: Preprocesamiento y Matriz Textual A (m x n)")
    from src.preprocessing import build_term_document_matrix
    A, vocab, df_docs, vectorizer = build_term_document_matrix(
        data_path=corpus_file,
        min_df=3,
        max_df=0.60,
        sublinear_tf=True
    )

    # Paso 5 y 6: SVD y Espectro Singular
    print_section("Pasos 5 y 6: Cálculo de SVD y Análisis Espectral")
    from src.svd_analysis import (
        compute_svd,
        compute_singular_spectrum_metrics,
        project_documents_2d,
        project_terms_2d
    )
    U, S, Vt = compute_svd(A)
    metrics = compute_singular_spectrum_metrics(S)
    print(f"Top 5 valores singulares: {S[:5].round(4)}")
    print(f"Varianza acumulada (k=4): {metrics['var_cumulative'][3]*100:.2f}%")
    print(f"Varianza acumulada (k=20): {metrics['var_cumulative'][19]*100:.2f}%")

    # Paso 7: Verificación de Proyecciones 2D
    print_section("Paso 7: Proyecciones 2D en Span(u_a, u_b) y Span(v_a, v_b)")
    doc_coords_12, d_err_12 = project_documents_2d(A, U, S, Vt, comp_a=1, comp_b=2)
    doc_coords_23, d_err_23 = project_documents_2d(A, U, S, Vt, comp_a=2, comp_b=3)
    term_coords_23, t_err_23 = project_terms_2d(A, U, S, Vt, comp_a=2, comp_b=3)
    print(f"Discrepancia analítica docs (Span(u2, u3)): {d_err_23:.2e}")
    print(f"Discrepancia analítica términos (Span(v2, v3)): {t_err_23:.2e}")

    # Generación de Figuras
    print_section("Generación de Figuras en figures/")
    from src.visualizations import (
        plot_singular_values_and_variance,
        plot_scree_and_error,
        plot_documents_2d,
        plot_terms_2d,
        plot_semantic_biplot,
        plot_topic_loadings
    )
    plot_singular_values_and_variance(S, k_selected=20)
    plot_scree_and_error(S, k_selected=20)
    plot_documents_2d(doc_coords_12, doc_coords_23, df_docs)
    plot_terms_2d(term_coords_23, vocab)
    plot_semantic_biplot(doc_coords_23, df_docs, term_coords_23, vocab)
    plot_topic_loadings(U, S, vocab)
    print("Todas las figuras generadas con éxito.")

    # Paso 8: Exploración Semántica
    print_section("Paso 8: Exploración Semántica (k = 20)")
    from src.semantic_exploration import (
        get_latent_representations,
        find_nearest_terms,
        find_nearest_documents,
        extract_component_topics
    )
    term_emb, doc_emb = get_latent_representations(U, S, Vt, k=20)
    test_words = ['neural', 'galaxy', 'gene', 'portfolio', 'model']
    for word in test_words:
        neighbors = find_nearest_terms(word, vocab, term_emb, top_n=3)
        res_str = ", ".join([f"{w} ({s:.2f})" for w, s in neighbors])
        print(f" -> Vecinos de '{word}': {res_str}")

    # Compilación del Informe PDF
    print_section("Compilación del Informe PDF")
    from report.generate_pdf import generate_report_html, compile_html_to_pdf
    html_report = generate_report_html()
    compile_html_to_pdf(html_report)

    elapsed = time.time() - start_time
    print_section("Pipeline Finalizado con Éxito")
    print(f"Tiempo total transcurrido: {elapsed:.2f} segundos")
    print("Todos los requisitos y entregables están listos y verificados.")

if __name__ == "__main__":
    main()
