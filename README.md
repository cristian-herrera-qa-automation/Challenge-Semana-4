<div align="center">

# 🔍 Sistema RAG con FastAPI

**API de búsqueda semántica y generación de respuestas contextuales sobre documentos cargados por el usuario**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-FF6B6B?style=flat-square)](https://www.trychroma.com/)
[![Cohere](https://img.shields.io/badge/Cohere-command--a-39594D?style=flat-square)](https://cohere.com/)

_Challenge Semana 4 — Get Talent_

</div>

---

## 📑 Contenido

- [Instalación](#-instalación)
- [Endpoints](#-endpoints)
- [Arquitectura](#-arquitectura)
- [Decisiones de diseño](#-decisiones-de-diseño)
- [IA Responsable](#-ia-responsable)
- [Calibración del umbral](#-calibración-del-umbral-de-similitud)
- [Verificación](#-verificación)
- [Limitaciones](#-limitaciones-conocidas)

---

## ⚙️ Instalación

**1.** Instalar dependencias

```bash
pip install -r requirements.txt
```

**2.** Copiar `.env.example` como `.env` y completar con tu clave de Cohere

```bash
COHERE_API_KEY=tu_api_key_aca
```

**3.** Levantar la API

```bash
uvicorn main:app --reload
```

> 💡 Documentación interactiva disponible en **http://localhost:8000/docs**

---

## 🔌 Endpoints

|     | Método | Ruta                   | Descripción                                             |
| :-: | :----: | ---------------------- | ------------------------------------------------------- |
| 📤  | `POST` | `/upload`              | Carga un documento                                      |
| 🧬  | `POST` | `/generate-embeddings` | Genera embeddings de uno o de todos los documentos      |
| 🔎  | `POST` | `/search`              | Busca documentos relevantes para una consulta           |
| 💬  | `POST` | `/ask`                 | Responde una pregunta usando los documentos recuperados |

<details>
<summary><b>Ver ejemplos de uso con curl</b></summary>

<br>

**Cargar un documento**

```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"title": "Vacaciones", "content": "La politica de vacaciones otorga 14 dias habiles por anio."}'
```

**Generar embeddings de un documento**

```bash
curl -X POST http://localhost:8000/generate-embeddings \
  -H "Content-Type: application/json" \
  -d '{"document_id": "EL_ID_QUE_TE_DEVOLVIO_UPLOAD"}'
```

**Generar embeddings de todos los documentos**

```bash
curl -X POST http://localhost:8000/generate-embeddings \
  -H "Content-Type: application/json" -d '{}'
```

**Buscar**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "cuantos dias de vacaciones tengo", "top_k": 3}'
```

**Preguntar**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "cuantos dias de vacaciones tengo"}'
```

</details>

---

## 🏗️ Arquitectura

```
📁 proyecto/
├── 🐍 main.py         Endpoints. Recibe, valida, llama a rag.py y responde.
├── 🧠 rag.py          Lógica: chunking, embeddings, búsqueda, guardrails, generación.
├── 📋 schemas.py      Contratos de entrada y salida (validados por Pydantic).
├── 🔐 .env            Clave de Cohere (no versionado).
└── 📄 requirements.txt
```

**Flujo de una pregunta:**

```mermaid
flowchart LR
    A[/ask] --> B{¿Lenguaje<br/>inapropiado?}
    B -->|Sí| C[🚫 Bloquear]
    B -->|No| D[Embedding<br/>de la pregunta]
    D --> E[Búsqueda<br/>en Chroma]
    E --> F{¿Score ><br/>umbral?}
    F -->|No| G[⚠️ Sin información<br/>suficiente]
    F -->|Sí| H[🤖 LLM responde<br/>con el contexto]
```

---

## 🎯 Decisiones de diseño

<table>
<tr><td width="30%"><b>🧩 Chunking con solape</b></td>
<td>Los documentos se dividen en fragmentos de 500 caracteres con 50 de superposición. Un embedding de un documento entero promedia todos sus temas y pierde precisión; fragmentos chicos representan una idea concreta. El solape evita que una idea quede partida en el corte.</td></tr>

<tr><td><b>📐 Distancia coseno, no euclidiana</b></td>
<td>ChromaDB usa L2 por defecto, cuyas distancias no tienen techo y no se pueden mapear a un score interpretable. Con <code>hnsw:space: cosine</code> la distancia va de 0 a 2 y <code>similitud = 1 - distancia</code> da un valor real entre 0 y 1, como pide la consigna.</td></tr>

<tr><td><b>🔗 Un resultado por documento</b></td>
<td>Como un documento genera varios chunks, la búsqueda podría devolver el mismo documento repetido. Se conserva el mejor fragmento de cada documento.</td></tr>

<tr><td><b>🌐 Modelos externos</b></td>
<td>Embeddings con <code>embed-multilingual-v3.0</code> (multilingüe, necesario para un corpus en español) y generación con <code>command-a-03-2025</code>. Ambos nombres viven en constantes al inicio de <code>rag.py</code>.</td></tr>

<tr><td><b>💾 Persistencia en disco</b></td>
<td><code>PersistentClient</code> guarda la base vectorial en <code>chroma_data/</code>, así reiniciar el servidor no borra lo indexado.</td></tr>
</table>

---

## 🛡️ IA Responsable

|     | Requisito                 | Implementación                                                                                                                                                                                                                                                                                             |
| :-: | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ⚓  | **Grounding obligatorio** | Umbral de similitud de **0.45**, calibrado empíricamente. Por debajo del umbral no se llama al LLM y se devuelve el mensaje estándar. El prompt instruye explícitamente a no usar información externa. Además, si el propio modelo responde que no le alcanza el contexto, `grounded` se marca en `false`. |
| 🔒  | **Protección de datos**   | Los logs registran IDs, longitudes y scores — nunca contenido de documentos ni consultas. Los `except` loguean el tipo de excepción, no su mensaje, para no filtrar rutas ni claves.                                                                                                                       |
| 👁️  | **Transparencia**         | `/ask` siempre devuelve `context_used`, `similarity_score` y `grounded`.                                                                                                                                                                                                                                   |
| ⚠️  | **Errores controlados**   | Cualquier fallo de Cohere devuelve `503` con un mensaje genérico. Todos los errores usan el formato `{"error": "..."}`.                                                                                                                                                                                    |
| ⚖️  | **Equidad**               | Filtro de lenguaje inapropiado con límites de palabra (`\b`), para no dar falsos positivos con palabras que contienen a otras.                                                                                                                                                                             |
| 🔍  | **Control humano**        | Cada paso se loguea con su etiqueta (`PASO 1 CARGA` … `PASO 4 RESPUESTA`). Cada llamada a `/ask` lleva un ID de traza que permite reconstruir la decisión completa.                                                                                                                                        |

---

## 📊 Calibración del umbral de similitud

> El umbral que decide si hay contexto suficiente **no se eligió a ojo: se midió.**

Se cargó un documento narrativo en español (~2.200 caracteres, 5 chunks) y se consultó `/search` con preguntas de dos tipos:

| Consulta                              |  Relación  |   Score    |
| ------------------------------------- | :--------: | :--------: |
| _"quien era sol"_                     | ✅ Directa | **0.5248** |
| _"que paso durante la tormenta"_      | ✅ Directa | **> 0.50** |
| _"como era Luna"_                     | ✅ Directa | **> 0.50** |
| _"como se hace una tarta de manzana"_ | ❌ Ninguna | **0.3608** |
| _"cual es la capital de francia"_     | ❌ Ninguna | **0.3200** |

**Conclusión:** las consultas irrelevantes no bajan de ~0.32. Ese es el piso de ruido propio de los embeddings multilingües — dos textos en el mismo idioma comparten registro y estructura, así que nunca resultan totalmente disímiles. Un umbral bajo (0.30) dejaba pasar consultas sin relación y el grounding no cortaba.

```
   0.0 ──────────── 0.32 ─── 0.36 ──── 0.45 ──── 0.52 ──────────── 1.0
                    └── ruido ──┘        ▲       └── señal ──┘
                                      UMBRAL
```

Se fijó **`UMBRAL_SIMILITUD = 0.45`**, con unos 9 puntos de margen sobre el ruido más alto y 7 por debajo de la señal más baja.

---

## ✅ Verificación

El sistema fue probado end-to-end contra la API de Cohere, cargando un documento narrativo en español y ejecutando los cuatro endpoints desde `/docs`.

| Caso                                      | Resultado                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| 💬 Pregunta con contexto en el documento  | Responde con datos del documento · `grounded: true` · score **0.5250**         |
| 🚫 Pregunta sin relación con el documento | Mensaje estándar · `grounded: false` · `context_used` vacío · score **0.3198** |
| ⛔ Pregunta con lenguaje inapropiado      | Mensaje de bloqueo sin llegar a buscar · score **0.0**                         |
| ❌ `document_id` inexistente              | `404` con `{"error": "Documento no encontrado"}`                               |
| 🧩 Documento de ~2.200 caracteres         | Se divide en **5 chunks**                                                      |

En los tres casos de `/ask` el log de la aplicación registra la decisión tomada con su identificador de traza, permitiendo reconstruir por qué el sistema respondió, se abstuvo o bloqueó la consulta.

---

## 📌 Limitaciones conocidas

- **Filtro de lenguaje** — usa una lista fija de palabras. En producción correspondería un modelo de moderación, que detecta intención y no solo términos literales.
- **Almacenamiento en memoria** — los documentos originales se guardan en un diccionario (la consigna lo permite). Los embeddings sí persisten en disco.
- **Umbral dependiente del corpus** — fue calibrado sobre narrativa en español. Con documentos técnicos o de otro dominio habría que repetir la medición: el piso de similitud varía según el tipo de texto.
- **Formato de entrada** — `/upload` recibe el contenido como texto plano en el cuerpo JSON, según la especificación de la consigna (`content`: cadena de texto). Soportar archivos (PDF, DOCX, TXT) requeriría un endpoint `multipart/form-data` con extracción previa del texto — está fuera del alcance de esta versión, pero el resto del pipeline funcionaría sin cambios.
- **Deprecación de modelos** — Cohere retira modelos periódicamente: `command-r-plus` fue removido en septiembre de 2025 durante el desarrollo de este proyecto. Por eso los nombres están centralizados en constantes al inicio de `rag.py`, y cambiarlos implica editar una sola línea.
