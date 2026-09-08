# Sistema RAG con FastAPI

API de búsqueda semántica y generación de respuestas contextuales sobre documentos cargados por el usuario.

**Challenge Semana 4 — Get Talent**

---

## Instalación

```bash
pip install -r requirements.txt
```

Copiar `.env.example` como `.env` y completar con tu clave de Cohere:

```
COHERE_API_KEY=tu_api_key_aca
```

## Levantar la API

```bash
uvicorn main:app --reload
```

Documentación interactiva: http://localhost:8000/docs

---

## Endpoints

| Método | Ruta                   | Descripción                                             |
| ------ | ---------------------- | ------------------------------------------------------- |
| POST   | `/upload`              | Carga un documento                                      |
| POST   | `/generate-embeddings` | Genera embeddings de uno o de todos los documentos      |
| POST   | `/search`              | Busca documentos relevantes para una consulta           |
| POST   | `/ask`                 | Responde una pregunta usando los documentos recuperados |

### Ejemplos

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

---

## Arquitectura

```
main.py       Endpoints. Recibe, valida, llama a rag.py y responde.
rag.py        Lógica: chunking, embeddings, búsqueda, guardrails, generación.
schemas.py    Contratos de entrada y salida (validados por Pydantic).
```

**Flujo de una pregunta:**

```
/ask → guardrail de lenguaje → embedding de la pregunta
     → búsqueda en Chroma → ¿supera el umbral de similitud?
     → sí: LLM responde con el contexto | no: "No cuento con información suficiente"
```

---

## Decisiones de diseño

**Chunking con solape.** Los documentos se dividen en fragmentos de 500 caracteres con 50 de superposición. Un embedding de un documento entero promedia todos sus temas y pierde precisión; fragmentos chicos representan una idea concreta. El solape evita que una idea quede partida en el corte.

**Distancia coseno, no euclidiana.** ChromaDB usa L2 por defecto, cuyas distancias no tienen techo y no se pueden mapear a un score interpretable. Con `hnsw:space: cosine` la distancia va de 0 a 2 y `similitud = 1 - distancia` da un valor real entre 0 y 1, como pide la consigna.

**Un resultado por documento.** Como un documento genera varios chunks, la búsqueda podría devolver el mismo documento repetido. Se conserva el mejor fragmento de cada documento.

**Modelos externos.** Embeddings con `embed-multilingual-v3.0` (multilingüe, necesario para un corpus en español) y generación con `command-a-03-2025`. Ambos nombres viven en constantes al inicio de `rag.py`.

**Persistencia en disco.** `PersistentClient` guarda la base vectorial en `chroma_data/`, así reiniciar el servidor no borra lo indexado.

---

## Prácticas de IA Responsable

| Requisito                 | Implementación                                                                                                                                                                                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Grounding obligatorio** | Umbral de similitud de 0.45, calibrado empíricamente (ver más abajo). Por debajo del umbral no se llama al LLM y se devuelve el mensaje estándar. El prompt instruye explícitamente a no usar información externa. Además, si el propio modelo responde que no le alcanza el contexto, `grounded` se marca en `false`. |
| **Protección de datos**   | Los logs registran IDs, longitudes y scores — nunca contenido de documentos ni consultas. Los `except` loguean el tipo de excepción, no su mensaje, para no filtrar rutas ni claves.                                                                                                                                   |
| **Transparencia**         | `/ask` siempre devuelve `context_used`, `similarity_score` y `grounded`.                                                                                                                                                                                                                                               |
| **Errores controlados**   | Cualquier fallo de Cohere devuelve 503 con un mensaje genérico. Todos los errores usan el formato `{"error": "..."}`.                                                                                                                                                                                                  |
| **Equidad**               | Filtro de lenguaje inapropiado con límites de palabra (`\b`), para no dar falsos positivos con palabras que contienen a otras.                                                                                                                                                                                         |
| **Control humano**        | Cada paso se loguea con su etiqueta (`PASO 1 CARGA` … `PASO 4 RESPUESTA`). Cada llamada a `/ask` lleva un ID de traza que permite reconstruir la decisión completa.                                                                                                                                                    |

---

## Calibración del umbral de similitud

El umbral que decide si hay contexto suficiente no se eligió a ojo: se midió.

Se cargó un documento narrativo en español (~2.200 caracteres, 5 chunks) y se consultó `/search` con preguntas de dos tipos:

| Consulta                            | Relación con el documento | Score  |
| ----------------------------------- | ------------------------- | ------ |
| "quien era sol"                     | Directa                   | 0.5248 |
| "que paso durante la tormenta"      | Directa                   | > 0.50 |
| "como era Luna"                     | Directa                   | > 0.50 |
| "como se hace una tarta de manzana" | Ninguna                   | 0.3608 |
| "cual es la capital de francia"     | Ninguna                   | 0.3200 |

**Conclusión:** las consultas irrelevantes no bajan de ~0.32. Ese es el piso de ruido propio de los embeddings multilingües — dos textos en el mismo idioma comparten registro y estructura, así que nunca resultan totalmente disímiles. Un umbral bajo (0.30) dejaba pasar consultas sin relación y el grounding no cortaba.

Se fijó **`UMBRAL_SIMILITUD = 0.45`**, con unos 9 puntos de margen sobre el ruido más alto y 7 por debajo de la señal más baja.

---

## Verificación

El sistema fue probado end-to-end contra la API de Cohere, cargando un documento
narrativo en español y ejecutando los cuatro endpoints desde `/docs`.

| Caso                                                | Resultado                                                                           |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Pregunta con contexto en el documento               | Responde con datos del documento, `grounded: true`, score 0.5250                    |
| Pregunta sin relación con el documento              | Devuelve el mensaje estándar, `grounded: false`, `context_used` vacío, score 0.3198 |
| Pregunta con lenguaje inapropiado                   | Devuelve el mensaje de bloqueo sin llegar a buscar, score 0.0                       |
| `document_id` inexistente en `/generate-embeddings` | 404 con `{"error": "Documento no encontrado"}`                                      |
| Documento de ~2.200 caracteres                      | Se divide en 5 chunks                                                               |

En los tres casos de `/ask` el log de la aplicación registra la decisión tomada
con su identificador de traza, permitiendo reconstruir por qué el sistema
respondió, se abstuvo o bloqueó la consulta.

---

## Limitaciones conocidas

- El filtro de lenguaje inapropiado usa una lista fija de palabras. En producción correspondería un modelo de moderación, que detecta intención y no solo términos literales.
- Los documentos originales se guardan en un diccionario en memoria (la consigna lo permite). Los embeddings sí persisten en disco.
- El umbral fue calibrado sobre un corpus de narrativa en español. Con documentos técnicos o de otro dominio habría que repetir la medición: el piso de similitud varía según el tipo de texto.
- El endpoint `/upload` recibe el contenido como texto plano en el cuerpo JSON, según la especificación de campos de la consigna (`content`: cadena de texto). Soportar carga de archivos (PDF, DOCX, TXT) requeriría un endpoint `multipart/form-data` con extracción previa del texto — está fuera del alcance de esta versión, pero el resto del pipeline (chunking, embeddings, búsqueda y generación) funcionaría sin cambios.
- Cohere retira modelos periódicamente: `command-r-plus` fue removido en septiembre de 2025 durante el desarrollo de este proyecto. Por eso los nombres de modelo están centralizados en constantes al inicio de `rag.py` (`MODELO_EMBEDDINGS` y `MODELO_CHAT`), y cambiarlos implica editar una sola línea.
