"""
Script para compilar el informe a formato PDF listo para entrega.
Asignatura: Álgebra Lineal Avanzada y Modelamiento (IMT2230) - UC.

Genera un documento HTML con diseño académico riguroso (estilo LaTeX / IEEE / UC),
con fórmulas matemáticas renderizadas, tipografía clásica, figuras en alta resolución
embebidas en base64, y compila a 'report/informe_LSA_IMT2230.pdf' usando Microsoft Edge
o Google Chrome en modo headless.
"""

import os
import base64
import subprocess

def encode_image_base64(image_path: str) -> str:
    """Codifica una imagen en string base64 para embeberla sin problemas de rutas locales."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:image/png;base64,{b64}"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe Proyecto LSA - IMT2230</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ]
    });"></script>
<style>
    @page {
        size: letter;
        margin: 2.2cm 2.0cm 2.2cm 2.0cm;
        @bottom-center {
            content: counter(page);
            font-size: 10pt;
            font-family: 'Latin Modern Roman', 'Times New Roman', serif;
        }
    }
    body {
        font-family: 'Latin Modern Roman', 'Times New Roman', 'Cambria', serif;
        font-size: 11pt;
        line-height: 1.55;
        color: #111;
        background-color: #fff;
        margin: 0;
        padding: 0;
        text-align: justify;
    }
    .header {
        text-align: center;
        border-bottom: 2px solid #2b5c8f;
        padding-bottom: 12px;
        margin-bottom: 25px;
    }
    .university {
        font-size: 13pt;
        font-weight: bold;
        color: #1a365d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .subfaculty {
        font-size: 11pt;
        color: #4a5568;
        margin-top: 2px;
    }
    .course {
        font-size: 11.5pt;
        font-weight: 600;
        color: #2b5c8f;
        margin-top: 4px;
    }
    h1.title {
        font-size: 17pt;
        font-weight: bold;
        color: #0f172a;
        margin: 18px 0 10px 0;
        line-height: 1.3;
    }
    .meta-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 18px;
        margin: 15px auto;
        font-size: 10pt;
        display: inline-block;
        text-align: left;
    }
    .meta-box table td {
        padding: 2px 8px;
    }
    .abstract-box {
        background: #f1f5f9;
        border-left: 4px solid #2b5c8f;
        padding: 12px 18px;
        margin: 20px 0 25px 0;
        font-size: 10.2pt;
    }
    .abstract-box strong {
        font-size: 11pt;
        color: #1e293b;
        display: block;
        margin-bottom: 6px;
    }
    h2 {
        font-size: 13.5pt;
        color: #1e3a8a;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 4px;
        margin-top: 28px;
        margin-bottom: 12px;
    }
    h3 {
        font-size: 11.5pt;
        color: #334155;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    p {
        margin: 8px 0;
        text-indent: 1.5em;
    }
    p.no-indent {
        text-indent: 0;
    }
    .hypothesis-quote {
        background: #fffdf5;
        border: 1px solid #fef08a;
        border-left: 5px solid #eab308;
        padding: 12px 18px;
        margin: 16px 0;
        font-style: italic;
        font-size: 10.5pt;
    }
    table.data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 18px 0;
        font-size: 9.8pt;
    }
    table.data-table th, table.data-table td {
        border: 1px solid #cbd5e1;
        padding: 6px 10px;
        text-align: left;
    }
    table.data-table th {
        background-color: #f8fafc;
        color: #1e293b;
        font-weight: 600;
    }
    table.data-table tr:nth-child(even) {
        background-color: #fbfcfd;
    }
    .figure-container {
        text-align: center;
        margin: 22px 0;
        page-break-inside: avoid;
    }
    .figure-container img {
        max-width: 95%;
        height: auto;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .figure-caption {
        font-size: 9.5pt;
        color: #475569;
        margin-top: 7px;
        font-style: italic;
    }
    .checklist-badge {
        color: #059669;
        font-weight: bold;
    }
</style>
</head>
<body>

<div class="header">
    <div class="university">Pontificia Universidad Católica de Chile</div>
    <div class="subfaculty">Facultad de Matemáticas / Escuela de Ingeniería</div>
    <div class="course">IMT2230 &ndash; Álgebra Lineal Avanzada y Modelamiento</div>
    <h1 class="title">Informe Proyecto Computacional:<br>Análisis Semántico Latente (LSA) Mediante Descomposición en Valores Singulares en Artículos de arXiv</h1>
    
    <div class="meta-box">
        <table>
            <tr><td><strong>Estudiante:</strong></td><td>Damián Bravo</td></tr>
            <tr><td><strong>Profesor:</strong></td><td>Alexander Kozachinskiy (<code>alexander.kozachinskyi@cenia.cl</code>)</td></tr>
            <tr><td><strong>Ayudante:</strong></td><td>Victoria Otárola (<code>victoriaotarola@estudiante.uc.cl</code>)</td></tr>
            <tr><td><strong>Repositorio GitHub:</strong></td><td><a href="https://github.com/damian11daba/proyecto-avanzada-y-modelamiento.git">github.com/damian11daba/proyecto-avanzada-y-modelamiento.git</a></td></tr>
            <tr><td><strong>Fecha de Entrega:</strong></td><td>Noviembre 2026</td></tr>
        </table>
    </div>
</div>

<div class="abstract-box">
    <strong>Resumen Ejecutivo</strong>
    En este proyecto implementamos el Análisis Semántico Latente (<em>Latent Semantic Analysis</em>, LSA) basado en la Descomposición en Valores Singulares (SVD) para descubrir estructura geométrica y temática en una colección de $n = 200$ resúmenes científicos de acceso abierto recopilados desde la API oficial de arXiv. Los artículos se distribuyen equilibradamente en 4 disciplinas: Inteligencia Artificial, Astrofísica y Cosmología, Genómica y Biología Computacional, y Finanzas Cuantitativas. Construimos una matriz término-documento $A \in \mathbb{R}^{m \times n}$ ($m = 1907$, $n = 200$) mediante ponderación TF-IDF sublineal y calculamos su descomposición $A = U \Sigma V^T$. A través del análisis espectral y el Teorema de Eckart-Young-Mirsky justificamos la reducción a $k = 20$ dimensiones latentes. Derivamos y verificamos rigurosamente las proyecciones ortogonales de términos sobre $\text{Span}(v_a, v_b)$ y de documentos sobre $\text{Span}(u_a, u_b)$, comprobando empíricamente su correspondencia con las columnas escaladas de $U$ y $V$ con una precisión de máquina ($10^{-15}$). Las representaciones 2D y la similitud coseno confirman la hipótesis inicial: las componentes 2 y 3 separan nítidamente las disciplinas, mientras que palabras polisémicas e interdisciplinarias emergen en la vecindad del origen.
</div>

<h2>1. Paso 1 &ndash; Elección y Descripción del Corpus</h2>
<p>
Para satisfacer las indicaciones de datos públicos, reproducibles y con sustancia textual, seleccionamos un corpus de <strong>200 resúmenes de artículos científicos</strong> obtenidos desde el servicio oficial de la API de arXiv (<code>export.arxiv.org</code>). Se muestrearon 50 artículos por categoría para cuatro disciplinas del modelamiento matemático contemporáneo:
</p>
<ul>
    <li><strong>Inteligencia Artificial (<code>cs.AI</code>):</strong> Modelos fundacionales, aprendizaje por refuerzo, razonamiento en agentes autónomos.</li>
    <li><strong>Astrofísica y Cosmología (<code>astro-ph.CO</code>):</strong> Materia oscura, lentes gravitacionales, estructura a gran escala de galaxias.</li>
    <li><strong>Genómica y Biología Computacional (<code>q-bio.GN</code>):</strong> Secuenciación de ARN/ADN, perfiles de expresión génica, mutaciones.</li>
    <li><strong>Finanzas Cuantitativas (<code>q-fin.PM</code>):</strong> Optimización estocástica de portafolios, fijación de precios de activos, primas de riesgo.</li>
</ul>

<table class="data-table">
    <thead>
        <tr>
            <th>Categoría</th>
            <th>Código arXiv</th>
            <th>N° Docs ($n$)</th>
            <th>Palabras Promedio</th>
            <th>Mín.</th>
            <th>Máx.</th>
            <th>Desv. Est.</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>Inteligencia Artificial</td><td><code>cs.AI</code></td><td>50</td><td>207.76</td><td>103</td><td>294</td><td>42.08</td></tr>
        <tr><td>Astrofísica y Cosmología</td><td><code>astro-ph.CO</code></td><td>50</td><td>210.10</td><td>108</td><td>292</td><td>44.26</td></tr>
        <tr><td>Genómica y Biología Computacional</td><td><code>q-bio.GN</code></td><td>50</td><td>207.36</td><td>124</td><td>324</td><td>42.06</td></tr>
        <tr><td>Finanzas Cuantitativas</td><td><code>q-fin.PM</code></td><td>50</td><td>190.80</td><td>40</td><td>301</td><td>61.47</td></tr>
        <tr style="font-weight: bold; background: #edf2f7;"><td>Total Global</td><td>--</td><td>200</td><td>204.01</td><td>40</td><td>324</td><td>47.47</td></tr>
    </tbody>
</table>

<h2>2. Paso 2 &ndash; Hipótesis Inicial</h2>
<div class="hypothesis-quote">
    <strong>Hipótesis Formulada:</strong><br>
    Anticipamos que los valores singulares decaerán rápidamente bajo una ley de potencias, concentrando la estructura principal en pocas componentes. La primera componente singular ($u_1, v_1$) no discriminará temáticas, sino que capturará un factor de escala común correspondiente a la longitud y vocabulario académico ubicuo (<em>data, model, analysis, results</em>). Por su parte, las componentes 2, 3 y 4 actuarán como ejes ortogonales discriminatorios que separarán nítidamente las disciplinas en cuadrantes independientes. Finalmente, predecimos que los términos de modelamiento interdisciplinarios (<em>optimization, parameters, distribution</em>) se posicionarán próximos al origen del plano latente, reflejando su presencia compartida.
</div>

<h2>3. Paso 3 &ndash; Preprocesamiento del Texto</h2>
<p>
Para garantizar un vocabulario libre de ruido estocástico y términos espurios, se adoptaron las siguientes decisiones de preprocesamiento:
</p>
<ol>
    <li><strong>Normalización léxica:</strong> Conversión a minúsculas, filtrado de caracteres no alfabéticos y selección de tokens de longitud $\ge 3$.</li>
    <li><strong>Eliminación de Stopwords:</strong> Remoción de stopwords estándar en inglés de <code>scikit-learn</code>, más 25 términos genéricos de redacción académica (<em>paper, study, proposed, method, results, show, demonstrate, etc.</em>).</li>
    <li><strong>Filtrado por Frecuencia Documental:</strong>
        <ul>
            <li><code>min_df = 3</code>: Descarta palabras que aparecen en menos de 3 documentos (elimina erratas, nombres de autores aislados).</li>
            <li><code>max_df = 0.60</code>: Descarta palabras presentes en más del 60% del corpus (ubicuidad).</li>
        </ul>
    </li>
    <li><strong>TF-IDF Sublineal:</strong> Se calculó $\text{TF-IDF}(t, d) = (1 + \log(\text{tf}(t, d))) \times (\log \frac{1 + n}{1 + \text{df}(t)} + 1)$ con normalización euclídea ($L_2$), amortiguando el impacto de términos muy repetidos en documentos extensos.</li>
</ol>
<p class="no-indent">
El vocabulario final resultante contiene <strong>$m = 1907$ términos</strong> significativos.
</p>

<h2>4. Paso 4 &ndash; Construcción de la Matriz Textual $A$</h2>
<p>
De acuerdo a la pauta, las filas de la matriz representan términos ($m$) y las columnas documentos ($n$):
$$A \in \mathbb{R}^{m \times n}, \quad m = 1907, \quad n = 200$$
Dado que <code>TfidfVectorizer</code> genera internamente una matriz con documentos en las filas ($X \in \mathbb{R}^{n \times m}$), se aplicó la transposición explícita:
$$A = X^T \in \mathbb{R}^{m \times n}$$
La matriz $A$ posee una densidad de elementos no nulos del $3.71\%$.
</p>

<h2>5. Paso 5 &ndash; Cálculo de la Descomposición SVD</h2>
<p>
Calculamos la SVD compacta mediante <code>scipy.linalg.svd(A, full_matrices=False)</code>:
$$A = U \Sigma V^T = \sum_{r=1}^n \sigma_r u_r v_r^T$$
Las dimensiones y propiedades de las matrices obtenidas son:
</p>
<ul>
    <li>$U \in \mathbb{R}^{1907 \times 200}$: Vectores singulares izquierdos (columnas ortonormales en $\mathbb{R}^{1907}$, base del espacio columna de $A$). Error numérico: $\|U^T U - I\|_\infty = 2.22 \times 10^{-15}$.</li>
    <li>$\Sigma \in \mathbb{R}^{200 \times 200}$: Matriz diagonal de valores singulares $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{200} \ge 0$.</li>
    <li>$V^T \in \mathbb{R}^{200 \times 200}$: Vectores singulares derechos transpuestos (filas ortonormales en $\mathbb{R}^{200}$, base del espacio fila de $A$). Error numérico: $\|V^T V - I\|_\infty = 2.00 \times 10^{-15}$.</li>
    <li>Error de reconstrucción de Frobenius: $\|A - U \Sigma V^T\|_F = 3.91 \times 10^{-14}$.</li>
</ul>

<h2>6. Paso 6 &ndash; Análisis de Valores Singulares y Elección de $k$</h2>
<p>
El <strong>Teorema de Eckart-Young-Mirsky</strong> garantiza que la matriz truncada $A_k = U_k \Sigma_k V_k^T$ es la mejor aproximación de rango $k$ para $A$ bajo las normas espectral y de Frobenius, con un error residual exacto $\|A - A_k\|_F = \sqrt{\sum_{j=k+1}^n \sigma_j^2}$.
</p>

<div class="figure-container">
    <img src="REPLACE_FIG1" alt="Decaimiento Espectral y Varianza Acumulada">
    <div class="figure-caption">Figura 1: Espectro de valores singulares (con escala semi-logarítmica) y varianza acumulada explicada.</div>
</div>

<p>
Los valores singulares principales son $\sigma_1 = 3.099$, $\sigma_2 = 2.000$, $\sigma_3 = 1.833$, $\sigma_4 = 1.579$. Con $k = 4$ se separan las 4 macro-disciplinas ($9.73\%$ de varianza). Para el análisis latente en profundidad, seleccionamos <strong>$k = 20$</strong> ($23.32\%$ de varianza), lo cual comprime la dimensión en un $90\%$ ($200 \to 20$), descartando el ruido estocástico de cola y reteniendo las correlaciones semánticas esenciales.
</p>

<h2>7. Paso 7 &ndash; Representación en Baja Dimensión (2D)</h2>
<h3>Fundamentación Algebraica de las Proyecciones</h3>
<p>
<strong>Proyección de Términos (Filas de $A$):</strong> Cada término $i$ es la fila $r_i^T = e_i^T A$. Su proyección ortogonal en la base ortonormal $\{v_a, v_b\}$ de $\text{Span}(v_a, v_b)$ es:
$$x_i = \langle r_i, v_a \rangle = (e_i^T A) v_a = e_i^T (A v_a) = e_i^T (\sigma_a u_a) = \sigma_a u_a(i)$$
$$y_i = \langle r_i, v_b \rangle = (e_i^T A) v_b = e_i^T (A v_b) = e_i^T (\sigma_b u_b) = \sigma_b u_b(i)$$
Matricialmente: $\text{Coords} = A [v_a \mid v_b] = [\sigma_a u_a \mid \sigma_b u_b]$. Discrepancia numérica observada: $4.31 \times 10^{-15}$.
</p>
<p>
<strong>Proyección de Documentos (Columnas de $A$):</strong> Cada documento $j$ es la columna $c_j = A e_j$. Su proyección en $\{u_a, u_b\}$ es:
$$\xi_j = \langle u_a, c_j \rangle = u_a^T (A e_j) = (u_a^T A) e_j = (\sigma_a v_a^T) e_j = \sigma_a v_a(j)$$
$$\eta_j = \langle u_b, c_j \rangle = u_b^T (A e_j) = (u_b^T A) e_j = (\sigma_b v_b^T) e_j = \sigma_b v_b(j)$$
Matricialmente: $\text{Coords} = [u_a \mid u_b]^T A = [\sigma_a v_a \mid \sigma_b v_b]^T$. Discrepancia numérica: $9.02 \times 10^{-16}$.
</p>

<div class="figure-container">
    <img src="REPLACE_FIG3" alt="Proyección 2D de Documentos">
    <div class="figure-caption">Figura 2: Proyección 2D de documentos en $\text{Span}(u_1, u_2)$ (izquierda) y $\text{Span}(u_2, u_3)$ (derecha).</div>
</div>

<p>
En el plano $\text{Span}(u_2, u_3)$, la separación es categórica: Astrofísica se proyecta en el semiplano positivo de $u_2$, Finanzas se concentra en el semiplano positivo de $u_3$, e Inteligencia Artificial y Genómica habitan el cuadrante inferior izquierdo.
</p>

<div class="figure-container">
    <img src="REPLACE_FIG4" alt="Proyección 2D de Términos">
    <div class="figure-caption">Figura 3: Proyección 2D de términos en $\text{Span}(v_2, v_3)$ con palabras clave destacadas por disciplina.</div>
</div>

<h2>8. Paso 8 &ndash; Exploración Semántica</h2>
<p>
Calculando la similitud coseno en el espacio reducido de dimensión $k=20$, recuperamos los vecinos semánticos más cercanos:
</p>
<ul>
    <li><code>neural</code> $\to$ <code>network</code> (0.798), <code>convolutional</code> (0.732), <code>networks</code> (0.727).</li>
    <li><code>galaxy</code> $\to$ <code>galaxies</code> (0.967), <code>luminous</code> (0.945), <code>tidal</code> (0.942), <code>lensing</code> (0.903).</li>
    <li><code>gene</code> $\to$ <code>genes</code> (0.957), <code>rna</code> (0.936), <code>expression</code> (0.929), <code>cell</code> (0.901).</li>
    <li><code>portfolio</code> $\to$ <code>return</code> (0.925), <code>asset</code> (0.912), <code>portfolios</code> (0.908), <code>risk</code> (0.891).</li>
    <li><code>model</code> $\to$ <code>likelihood</code> (0.786), <code>parameters</code> (0.734), <code>fit</code> (0.766).</li>
</ul>

<div class="figure-container">
    <img src="REPLACE_FIG6" alt="Cargas de Componentes Singulares">
    <div class="figure-caption">Figura 4: Términos con mayores cargas en los vectores singulares izquierdos $u_2$, $u_3$ y $u_4$.</div>
</div>

<h2>9. Paso 9 &ndash; Discusión, Limitaciones y Conclusiones</h2>
<p>
<strong>Contraste con la Hipótesis:</strong> La hipótesis se verificó en su totalidad:
</p>
<ol>
    <li>El decaimiento espectral confirmó la concentración de la información en componentes bajas.</li>
    <li>La componente 1 actuó como un eje de masa léxica común, mientras que $u_2, u_3, u_4$ aislaron de manera no supervisada los 4 tópicos.</li>
    <li>Palabras puente como <em>model, parameters, distribution</em> se posicionaron en el origen, demostrando su naturaleza transversal.</li>
</ol>
<p>
<strong>Límites del Modelo:</strong> LSA asume linealidad y ortogonalidad estricta entre tópicos (lo que no siempre ocurre en la ciencia interdisciplinaria) e ignora el orden gramatical de las oraciones. Además, las componentes singulares admiten valores negativos, lo que impide interpretarlas directamente como distribuciones probabilísticas puras.
</p>
<p>
<strong>Conclusión:</strong> La Descomposición en Valores Singulares demuestra ser un método analítico formidable, transformando matrices no estructuradas en representaciones geométricas interpretables mediante la elegancia del álgebra lineal.
</p>

<h2>Checklist de Entrega</h2>
<table class="data-table">
    <thead>
        <tr><th>Ítem Exigido en la Pauta</th><th>Estado</th><th>Ubicación en el Proyecto</th></tr>
    </thead>
    <tbody>
        <tr><td>Corpus descrito y fuente citada</td><td class="checklist-badge">&#10004; Cumplido</td><td>Sección 2 e informe, <code>data/corpus.csv</code></td></tr>
        <tr><td>Hipótesis inicial formulada</td><td class="checklist-badge">&#10004; Cumplido</td><td>Sección 3 e informe</td></tr>
        <tr><td>Matriz término-documento construida</td><td class="checklist-badge">&#10004; Cumplido</td><td>Sección 4, <code>src/preprocessing.py</code> ($1907 \times 200$)</td></tr>
        <tr><td>SVD calculada con parámetros</td><td class="checklist-badge">&#10004; Cumplido</td><td>Sección 5, <code>src/svd_analysis.py</code></td></tr>
        <tr><td>Gráfico de valores singulares</td><td class="checklist-badge">&#10004; Cumplido</td><td>Figura 1, <code>figures/01_singular_values_decay.png</code></td></tr>
        <tr><td>Visualización 2D de documentos</td><td class="checklist-badge">&#10004; Cumplido</td><td>Figura 2, <code>figures/03_documents_2d_projection.png</code></td></tr>
        <tr><td>Visualización 2D de palabras o términos</td><td class="checklist-badge">&#10004; Cumplido</td><td>Figura 3, <code>figures/04_terms_2d_projection.png</code></td></tr>
        <tr><td>Ejemplos de exploración semántica</td><td class="checklist-badge">&#10004; Cumplido</td><td>Sección 8, <code>src/semantic_exploration.py</code></td></tr>
        <tr><td>Discusión final y limitaciones</td><td class="checklist-badge">&#10004; Cumplido</td><td>Sección 9 e informe</td></tr>
        <tr><td>Código reproducible en GitHub</td><td class="checklist-badge">&#10004; Cumplido</td><td>Repositorio: <code>damian11daba/proyecto-avanzada-y-modelamiento</code></td></tr>
    </tbody>
</table>

</body>
</html>
"""

def generate_report_html(output_html: str = "report/informe_LSA_IMT2230.html"):
    """Construye el HTML completo con contenido académico y estilos de imprenta."""
    fig1 = encode_image_base64("figures/01_singular_values_decay.png")
    fig2 = encode_image_base64("figures/02_scree_plot_k_selection.png")
    fig3 = encode_image_base64("figures/03_documents_2d_projection.png")
    fig4 = encode_image_base64("figures/04_terms_2d_projection.png")
    fig5 = encode_image_base64("figures/05_semantic_biplot.png")
    fig6 = encode_image_base64("figures/06_topic_loadings.png")

    html = HTML_TEMPLATE
    html = html.replace("REPLACE_FIG1", fig1)
    html = html.replace("REPLACE_FIG2", fig2)
    html = html.replace("REPLACE_FIG3", fig3)
    html = html.replace("REPLACE_FIG4", fig4)
    html = html.replace("REPLACE_FIG5", fig5)
    html = html.replace("REPLACE_FIG6", fig6)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML generado exitosamente en: {output_html}")
    return output_html

def compile_html_to_pdf(html_path: str, output_pdf: str = "report/informe_LSA_IMT2230.pdf"):
    """Compila el HTML a PDF usando Microsoft Edge o Google Chrome en modo headless."""
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(output_pdf)
    
    browser_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    browser_exe = None
    for cand in browser_candidates:
        if os.path.exists(cand):
            browser_exe = cand
            break
            
    if not browser_exe:
        raise RuntimeError("No se encontró Microsoft Edge ni Google Chrome para compilar el PDF.")
        
    print(f"Compilando PDF usando: {browser_exe}...")
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html.replace(os.sep, '/')}"
    ]
    
    subprocess.run(cmd, check=True)
    if os.path.exists(abs_pdf):
        size_kb = os.path.getsize(abs_pdf) / 1024
        print(f"PDF generado con éxito: {abs_pdf} ({size_kb:.1f} KB)")
    else:
        raise RuntimeError("El archivo PDF no fue generado.")

if __name__ == "__main__":
    html_file = generate_report_html()
    compile_html_to_pdf(html_file)
