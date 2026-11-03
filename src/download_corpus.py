"""
Script para descarga reproducible del corpus de arXiv.
Asignatura: Álgebra Lineal Avanzada y Modelamiento (IMT2230) - UC.

Descarga 200 abstracts científicos de arXiv distribuidos en 4 categorías:
1. cs.AI (Inteligencia Artificial)
2. astro-ph.CO (Astrofísica y Cosmología)
3. q-bio.GN (Genómica y Biología Computacional)
4. q-fin.PM (Finanzas Cuantitativas)
"""

import os
import time
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

CATEGORIES = {
    'cs.AI': {
        'name': 'Inteligencia Artificial',
        'code': 'cs.AI',
        'desc': 'Modelos de machine learning, redes neuronales y razonamiento artificial'
    },
    'astro-ph.CO': {
        'name': 'Astrofísica y Cosmología',
        'code': 'astro-ph.CO',
        'desc': 'Evolución cósmica, materia oscura, galaxias y radiación de fondo'
    },
    'q-bio.GN': {
        'name': 'Genómica y Biología Computacional',
        'code': 'q-bio.GN',
        'desc': 'Secuenciación genética, expresión génica, mutaciones y bioinformática'
    },
    'q-fin.PM': {
        'name': 'Finanzas Cuantitativas',
        'code': 'q-fin.PM',
        'desc': 'Gestión de portafolios, fijación de precios de activos y gestión de riesgo'
    }
}

def fetch_arxiv_category(category_code: str, max_results: int = 50) -> list:
    """Descarga metadatos y abstracts de arXiv para una categoría dada."""
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=cat:{category_code}&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    headers = {'User-Agent': 'IMT2230-LSA-Project/1.0 (academic research; mailto:damin.bravo@estudiante.uc.cl)'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_content = resp.read()
        
    root = ET.fromstring(xml_content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)
    
    records = []
    for entry in entries:
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        id_url = entry.find('atom:id', ns).text.strip()
        published = entry.find('atom:published', ns).text.strip()
        authors = [a.find('atom:name', ns).text.strip() for a in entry.findall('atom:author', ns)]
        
        # Limpieza básica de espacios múltiples
        title = ' '.join(title.split())
        summary = ' '.join(summary.split())
        
        records.append({
            'doc_id': f"{category_code}_{len(records)+1:03d}",
            'arxiv_id': id_url.split('/abs/')[-1],
            'category_code': category_code,
            'category_name': CATEGORIES[category_code]['name'],
            'title': title,
            'abstract': summary,
            'authors': '; '.join(authors[:3]) + (' et al.' if len(authors) > 3 else ''),
            'published': published[:10],
            'word_count': len(summary.split())
        })
    return records

def download_and_save_corpus(output_path: str = "data/corpus.csv", docs_per_cat: int = 50):
    """Descarga todos los documentos de las categorías especificadas y guarda un CSV estructurado."""
    all_records = []
    print("Iniciando descarga del corpus desde arXiv...")
    
    for cat_code, info in CATEGORIES.items():
        print(f" -> Descargando {docs_per_cat} artículos de {info['name']} ({cat_code})...")
        cat_records = fetch_arxiv_category(cat_code, max_results=docs_per_cat)
        all_records.extend(cat_records)
        print(f"    Completado ({len(cat_records)} documentos obtenidos).")
        time.sleep(1.5)  # Respetar rate limits de arXiv API
        
    df = pd.DataFrame(all_records)
    
    # Crear directorio data si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nCorpus guardado exitosamente en: {output_path}")
    print(f"Total de documentos descargados: {len(df)}")
    
    # Resumen descriptivo
    summary_path = os.path.join(os.path.dirname(output_path), "corpus_summary.csv")
    summary = df.groupby(['category_code', 'category_name']).agg(
        num_documentos=('doc_id', 'count'),
        palabras_promedio=('word_count', 'mean'),
        palabras_min=('word_count', 'min'),
        palabras_max=('word_count', 'max'),
        palabras_std=('word_count', 'std')
    ).round(2).reset_index()
    summary.to_csv(summary_path, index=False, encoding='utf-8')
    print(f"Resumen del corpus guardado en: {summary_path}")
    print("\nTabla Resumen del Corpus:")
    print(summary.to_string(index=False))
    return df

if __name__ == "__main__":
    download_and_save_corpus()
