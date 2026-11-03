# Proyecto Computacional: Análisis Semántico Latente (LSA) usando SVD

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Course: IMT2230](https://img.shields.io/badge/Curso-IMT2230%20UC-red.svg)](https://matematica.uc.cl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repositorio oficial del **Proyecto Computacional** para el curso **Álgebra Lineal Avanzada y Modelamiento (IMT2230)** de la **Pontificia Universidad Católica de Chile**.

* **Estudiantes:** Damián Bravo, Felipe López
* **Profesor:** Alexander Kozachinskiy (`alexander.kozachinskyi@cenia.cl`)
* **Ayudante:** Victoria Otárola (`victoriaotarola@estudiante.uc.cl`)
* **Repositorio GitHub:** [https://github.com/damian11daba/proyecto-avanzada-y-modelamiento.git](https://github.com/damian11daba/proyecto-avanzada-y-modelamiento.git)
* **Fecha de Entrega:** Noviembre 2026

---

## 📌 Descripción del Proyecto

Este proyecto aplica las herramientas teóricas y computacionales del **Álgebra Lineal Matricial**, en particular la **Descomposición en Valores Singulares (SVD)** y el **Teorema de Eckart-Young-Mirsky**, para descubrir la estructura semántica latente (*Latent Semantic Analysis*, LSA) en una colección de $n = 200$ resúmenes de artículos científicos de acceso abierto obtenidos desde la API de **arXiv**.

Los artículos corresponden a cuatro áreas contrastantes del modelamiento cuantitativo:
1. **Inteligencia Artificial (`cs.AI`)**
2. **Astrofísica y Cosmología (`astro-ph.CO`)**
3. **Genómica y Biología Computacional (`q-bio.GN`)**
4. **Finanzas Cuantitativas (`q-fin.PM`)**

A través de la construcción de una matriz término-documento $A \in \mathbb{R}^{m \times n}$ con ponderación TF-IDF sublineal, derivamos y comprobamos numéricamente las proyecciones ortogonales de términos y documentos en subespacios de baja dimensión, analizamos el decaimiento espectral y validamos experimentalmente la hipótesis semántica formulada previamente.

---

## 🚀 Instrucciones de Instalación y Ejecución

### 1. Clonar el Repositorio
```bash
git clone https://github.com/damian11daba/proyecto-avanzada-y-modelamiento.git
cd proyecto-avanzada-y-modelamiento
```

### 2. Crear y Activar Entorno Virtual (Opcional pero recomendado)
```bash
# En Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# En Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecución del Pipeline Completo (1 solo comando)
Para ejecutar todo el flujo (validación de datos, preprocesamiento, cálculo SVD, generación de figuras 300 DPI, exploración semántica y compilación del informe PDF):
```bash
python run_all.py
```

### 5. Exploración Interactiva en Jupyter Notebook
Para inspeccionar el análisis paso a paso con gráficos interactivos y código pre-renderizado:
```bash
jupyter notebook notebooks/proyecto_lsa_svd.ipynb
```

---

## 📂 Estructura del Repositorio

```text
proyecto-AAM/
│
├── README.md                          # Documentación e instrucciones de ejecución
├── requirements.txt                   # Lista de dependencias del proyecto
├── run_all.py                         # Pipeline maestro de ejecución completa
├── .gitignore                         # Configuración de exclusión para Git
│
├── data/
│   ├── corpus.csv                     # Dataset completo de 200 abstracts científicos de arXiv
│   └── corpus_summary.csv             # Resumen estadístico por categoría
│
├── notebooks/
│   ├── proyecto_lsa_svd.ipynb         # Jupyter Notebook interactivo con los 9 pasos de la pauta
│   └── build_notebook.py              # Script generador del notebook con nbformat
│
├── src/
│   ├── __init__.py
│   ├── download_corpus.py             # Script reproducible para descargar datos vía arXiv API
│   ├── preprocessing.py               # Limpieza, stopwords, TF-IDF y construcción de matriz A (m x n)
│   ├── svd_analysis.py                # Cálculo SVD, proyecciones 2D y verificación algebraica
│   ├── semantic_exploration.py        # Similitud coseno, búsqueda de vecinos y tópicos latentes
│   └── visualizations.py              # Generación de figuras estéticas en alta resolución (300 DPI)
│
├── figures/
│   ├── 01_singular_values_decay.png   # Espectro de valores singulares y varianza acumulada
│   ├── 02_scree_plot_k_selection.png  # Scree plot, método del codo y error relativo de Frobenius
│   ├── 03_documents_2d_projection.png # Proyecciones 2D de documentos en Span(u1, u2) y Span(u2, u3)
│   ├── 04_terms_2d_projection.png     # Proyección 2D de términos en Span(v2, v3) con etiquetas
│   ├── 05_semantic_biplot.png         # Biplot dual LSA (documentos y términos directrices)
│   └── 06_topic_loadings.png          # Cargas extremas de términos en vectores singulares u2, u3, u4
│
└── report/
    ├── informe_LSA_IMT2230.tex        # Código fuente LaTeX completo (listo para Overleaf)
    ├── informe_LSA_IMT2230.pdf        # Informe final compilado en PDF con formato académico UC
    ├── informe_LSA_IMT2230.html       # Versión web del informe con KaTeX
    └── generate_pdf.py                # Script de compilación automática a PDF
```

---

## 📐 Fundamentos Teóricos y Matemáticos

### 1. Construcción de la Matriz Término-Documento $A$
La matriz $A \in \mathbb{R}^{m \times n}$ ($m = 1907$, $n = 200$) tiene en sus filas los términos del vocabulario y en sus columnas los documentos del corpus:
$$A = X^T$$
donde $X \in \mathbb{R}^{n \times m}$ es la matriz obtenida mediante ponderación TF-IDF sublineal con normalización euclídea por fila.

### 2. Descomposición en Valores Singulares (SVD)
$$A = U \Sigma V^T = \sum_{r=1}^n \sigma_r u_r v_r^T$$
* $U \in \mathbb{R}^{m \times n}$: Matriz ortonormal con los vectores singulares izquierdos (base del espacio columna).
* $\Sigma \in \mathbb{R}^{n \times n}$: Matriz diagonal con los valores singulares ordenados $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n \ge 0$.
* $V \in \mathbb{R}^{n \times n}$: Matriz ortonormal con los vectores singulares derechos (base del espacio fila).

### 3. Teorema de Eckart-Young-Mirsky
La mejor aproximación de rango $k$ para $A$ bajo la norma de Frobenius $\|\cdot\|_F$ es $A_k = U_k \Sigma_k V_k^T$, con error residual exacto:
$$\|A - A_k\|_F = \sqrt{\sum_{j=k+1}^n \sigma_j^2}$$
Para este corpus, se determinó $k = 20$ como el umbral óptimo de estabilidad semántica.

### 4. Proyecciones en Baja Dimensión (Paso 7)
* **Proyección de Términos (Filas de $A$):** Cada término $i$ es la fila $r_i^T = e_i^T A$. Su proyección en la base ortonormal $\{v_a, v_b\}$ de $\text{Span}(v_a, v_b)$ es:
  $$\langle r_i, v_a \rangle = (e_i^T A) v_a = e_i^T (\sigma_a u_a) = \sigma_a u_a(i)$$
  $$\text{Coords\_Términos} = A [v_a \mid v_b] = [\sigma_a u_a \mid \sigma_b u_b] \in \mathbb{R}^{m \times 2}$$
* **Proyección de Documentos (Columnas de $A$):** Cada documento $j$ es la columna $c_j = A e_j$. Su proyección en $\{u_a, u_b\}$ de $\text{Span}(u_a, u_b)$ es:
  $$\langle u_a, c_j \rangle = u_a^T (A e_j) = (\sigma_a v_a^T) e_j = \sigma_a v_a(j)$$
  $$\text{Coords\_Documentos} = [u_a \mid u_b]^T A = [\sigma_a v_a \mid \sigma_b v_b]^T \in \mathbb{R}^{n \times 2}$$

En el código se verifica numéricamente que la discrepancia entre la proyección explícita por producto interno y el producto singular directo es menor a $10^{-14}$ (precisión de punto flotante).

---

## 📊 Resultados Principales

| Componente | Valor Singular $\sigma_i$ | Interpretación Semántica Principal |
| :---: | :---: | :--- |
| **$u_1$** | $3.099$ | **Masa Léxica Común:** Longitud y vocabulario científico compartido (*model, data, learning, risk*). |
| **$u_2$** | $2.000$ | **Astrofísica vs Resto:** Polaridad positiva en *dark, galaxy, mass, matter, cosmological*. |
| **$u_3$** | $1.833$ | **Finanzas vs Resto:** Polaridad positiva en *portfolio, risk, optimal, market, investor*. |
| **$u_4$** | $1.579$ | **IA vs Genómica:** Polaridad positiva en *agent, llm, reinforcement* vs negativa en *cell, gene, rna*. |

---

## ✅ Checklist Mínimo de Entrega

- [x] Corpus descrito y fuente citada (200 abstracts de arXiv en 4 categorías: `cs.AI`, `astro-ph.CO`, `q-bio.GN`, `q-fin.PM`).
- [x] Hipótesis inicial formulada explícitamente antes del análisis.
- [x] Matriz término-documento construida ($m = 1907 \times n = 200$, términos en filas, documentos en columnas).
- [x] Descomposición SVD calculada con parámetros, dimensiones y verificación de ortonormalidad.
- [x] Gráfico de decaimiento del espectro singular y varianza explicada acumulada.
- [x] Visualización 2D de documentos en $\text{Span}(u_1, u_2)$ y $\text{Span}(u_2, u_3)$.
- [x] Visualización 2D de términos en $\text{Span}(v_2, v_3)$ con términos clave y de modelamiento destacados.
- [x] Ejemplos de exploración semántica (sinonimia por similitud coseno, documentos más cercanos y cargas de tópicos).
- [x] Discusión final contrastando con la hipótesis inicial y analizando límites del método LSA.
- [x] Código reproducible disponible públicamente en GitHub con instrucciones claras.
- [x] Informe final en formato PDF (`report/informe_LSA_IMT2230.pdf`) y código fuente LaTeX (`report/informe_LSA_IMT2230.tex`).

---

## 📄 Licencia y Créditos
Desarrollado para el curso **Álgebra Lineal Avanzada y Modelamiento (IMT2230)**, Pontificia Universidad Católica de Chile.
Código bajo licencia MIT.
