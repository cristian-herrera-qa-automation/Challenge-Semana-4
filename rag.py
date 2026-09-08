"""
rag.py — Toda la lógica del RAG.

"""

import os
import re
import logging

import cohere
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración 
# ---------------------------------------------------------------------------

MODELO_EMBEDDINGS = "embed-multilingual-v3.0"
MODELO_CHAT = "command-a-03-2025"

# Umbral de grounding: si el mejor resultado no llega a esto,
# consideramos que NO hay contexto suficiente para responder. ✋

UMBRAL_SIMILITUD = 0.45

TAMANO_CHUNK = 500
SOLAPE_CHUNK = 50       


def _crear_cliente_cohere():
    """🔑🔑 Lee la API key. Falla si no está. 🔑🔑"""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta COHERE_API_KEY en el archivo .env")
    return cohere.ClientV2(api_key=api_key)


co = _crear_cliente_cohere()

# PersistentClient guarda en disco -> si se reinicia el server, no perdemos nada. 😊
chroma_client = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(anonymized_telemetry=False),
)

# hnsw:space="cosine": Le dice a Chroma que use distancia coseno. ✅
# Por defecto usa L2 (euclidiana), que NO se puede convertir a un score 0-1. 🎯
coleccion = chroma_client.get_or_create_collection(
    name="documentos",
    metadata={"hnsw:space": "cosine"},
)

# Guarda el texto original de cada documento. 📃✅
documentos = {}


# ---------------------------------------------------------------------------
# 📃✂ 1. Chunking — partir el texto en fragmentos 📃✂
# ---------------------------------------------------------------------------

def dividir_en_chunks(texto, tamano=TAMANO_CHUNK, solape=SOLAPE_CHUNK):
    """
    Corta un texto extenso en fragmentos con superposición.

    ¿Por qué? Un embedding de 3000 caracteres "promedia" todos los temas
    del documento y pierde precisión. Fragmentos chicos = búsqueda precisa. ✅📃

    El solape evita cortar una idea justo por la mitad: cada fragmento
    arranca un poco antes de donde terminó el anterior. ✅📃
    """
    texto = texto.strip()

    if len(texto) <= tamano:
        return [texto]

    chunks = []
    inicio = 0

    while inicio < len(texto):
        fin = inicio + tamano

        # Si no es el último pedazo, cortamos en el último espacio
        # para no partir una palabra al medio. ✅😊
        if fin < len(texto):
            ultimo_espacio = texto.rfind(" ", inicio, fin)
            if ultimo_espacio > inicio:
                fin = ultimo_espacio

        fragmento = texto[inicio:fin].strip()
        if fragmento:
            chunks.append(fragmento)

        if fin >= len(texto):
            break

        # Retrocedemos el solape para no cortar una idea al medio. ✅😊
        inicio = fin - solape

        # Pero si eso cae a mitad de una palabra, avanzamos hasta el
        # siguiente espacio. No se pierde nada: esa palabra ya entró
        # completa en el fragmento anterior. ✅😊
        
        siguiente_espacio = texto.find(" ", inicio)
        if texto[inicio] != " " and siguiente_espacio != -1:
            inicio = siguiente_espacio + 1

    return chunks


# ---------------------------------------------------------------------------
# 📃🔟 2. Embeddings — convertir texto en vectores 📃🔟
# ---------------------------------------------------------------------------

def generar_embeddings(textos, tipo):
    """
    🤖
    Llama a Cohere y devuelve una lista de vectores.

    'tipo' es "search_document" (para guardar) o "search_query" (para buscar). 🤖
    """
    respuesta = co.embed(
        texts=textos,
        model=MODELO_EMBEDDINGS,
        input_type=tipo,
        embedding_types=["float"],
    )
    return respuesta.embeddings.float


# ---------------------------------------------------------------------------
# 📃🔟3. Guardar documentos y sus embeddings 📃🔟
# ---------------------------------------------------------------------------

def guardar_documento(doc_id, titulo, contenido):
    """📁Guarda el documento en memoria. Todavía sin embeddings.📁"""
    documentos[doc_id] = {"title": titulo, "content": contenido}


def indexar_documento(doc_id):
    """
    📃🔟📁
    Parte el documento en chunks, genera un embedding por chunk
    y los guarda en Chroma.

    Devuelve cuántos chunks se generaron.
    📃🔟📁
    """
    doc = documentos[doc_id]
    chunks = dividir_en_chunks(doc["content"])

    vectores = generar_embeddings(chunks, tipo="search_document")

    # 🔑🔑 Cada chunk necesita un id único dentro de Chroma. 🔑🔑
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]

    # 📃🔑 En metadata guardamos a que documento pertenece cada chunk,
    # para poder devolver el document_id original en la búsqueda. 📃🔑
    metadatos = [{"document_id": doc_id, "title": doc["title"]} for _ in chunks]

    coleccion.upsert(
        ids=ids,
        embeddings=vectores,
        documents=chunks,
        metadatas=metadatos,
    )

    return len(chunks)


# ---------------------------------------------------------------------------
# 🔍🔍 4. Búsqueda — buscar fragmentos más parecidos a la consulta 🔍🔍
# ---------------------------------------------------------------------------

def _distancia_a_similitud(distancia):
    """
    ✋🧪
    Chroma devuelve DISTANCIA coseno (0 = idéntico, 2 = opuesto).
    La consigna pide un score de SIMILITUD entre 0 y 1.

    similitud = 1 - distancia, y recortamos por si da negativo.
    ✋🧪
    """
    similitud = 1.0 - distancia
    return max(0.0, min(1.0, similitud))


def buscar(query, top_k=3):
    """
    🧪💯🔍🔍
    Busca los fragmentos más parecidos a la consulta.

    Devuelve una lista de diccionarios, uno por documento (no por chunk):
    si dos fragmentos del mismo documento matchean, nos quedamos con el mejor.
    🧪💯🔍🔍
    """
    vector_query = generar_embeddings([query], tipo="search_query")[0]

    # Pedimos más resultados de los necesarios porque después
    # vamos a descartar chunks repetidos del mismo documento.
    resultados = coleccion.query(
        query_embeddings=[vector_query],
        n_results=top_k * 3,
    )

    if not resultados["ids"] or not resultados["ids"][0]:
        return []

    mejores_por_documento = {}

    for i in range(len(resultados["ids"][0])):
        metadata = resultados["metadatas"][0][i]
        doc_id = metadata["document_id"]
        similitud = _distancia_a_similitud(resultados["distances"][0][i])

        # Si ya vimos este documento con mejor score, lo salteamos. 😊
        if doc_id in mejores_por_documento:
            continue

        mejores_por_documento[doc_id] = {
            "document_id": doc_id,
            "title": metadata["title"],
            "chunk": resultados["documents"][0][i],
            "similarity_score": round(similitud, 4),
        }

    # Chroma ya devuelve ordenado de mejor a peor, así que el orden se mantiene. ✅💡
    return list(mejores_por_documento.values())[:top_k]


# ---------------------------------------------------------------------------
# 🚨🔍 5. Guardrail — filtro de lenguaje inapropiado 🔍🚨
# ---------------------------------------------------------------------------

# IA RESPONSABLE = Equidad y no discriminación.
# Es una lista de ejemplo de palabras bloqueadas.

PALABRAS_BLOQUEADAS = {
    "insulto", "insultos", "ofensa", "ofensas", "idiota", "estupido",
    "estúpido", "imbecil", "imbécil", "odio", "inferior", "inferiores",
}


def contiene_lenguaje_inapropiado(texto):
    """
    ✅🧪✋
    Devuelve True si la pregunta trae lenguaje bloqueado.

    Usamos regex con \\b (límite de palabra) en vez de 'x in texto'.
    Si no, "odio" daría positivo dentro de "custodio" o "melodioso".
    ✅🧪✋
    """
    texto_normalizado = texto.lower()
    for palabra in PALABRAS_BLOQUEADAS:
        if re.search(rf"\b{re.escape(palabra)}\b", texto_normalizado):
            return True
    return False


# ---------------------------------------------------------------------------
# 🤖✅ 6. Generación de la respuesta 🤖✅
# ---------------------------------------------------------------------------

def generar_respuesta(pregunta, contexto):
    """
    Le pide al LLM que responda usando SOLO el contexto recuperado.

    Esto es el "grounding": la instrucción explícita de no inventar.
    """
    prompt = f"""Respondé la pregunta usando ÚNICAMENTE la información del contexto.
Si la respuesta no está en el contexto, respondé exactamente:
"No cuento con información suficiente para responder a esta consulta."

No agregues opiniones, juicios de valor ni información externa.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""

    respuesta = co.chat(
        model=MODELO_CHAT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return respuesta.message.content[0].text.strip()