"""
main.py : Los 4 endpoints de la API.

"""

import uuid
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

import rag
from schemas import (
    DocumentUpload, EmbeddingRequest, SearchRequest, AskRequest,
    UploadResponse, EmbeddingResponse, SearchResponse, AskResponse,
)


# ---------------------------------------------------------------------------
# Logging (control humano y trazabilidad)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag_api")


# ---------------------------------------------------------------------------
# 👨‍💻 Mensajes fijos para las salidas. 👨‍💻
# ---------------------------------------------------------------------------

SIN_CONTEXTO = "No cuento con información suficiente para responder a esta consulta."
CONSULTA_BLOQUEADA = "No puedo responder a este tipo de consultas."
SERVICIO_CAIDO = "El servicio externo no pudo procesar la solicitud en este momento."


app = FastAPI(title="RAG System API", version="1.0")


@app.exception_handler(HTTPException)
async def formato_de_error(request: Request, exc: HTTPException):
    """
    🎯🎯
    Todos los errores salen como {"error": "..."} en vez del {"detail": "..."}
    que usa FastAPI por defecto. 🎯🎯

    💡💡
    Aca nunca se filtra el mensaje original de la excepción,
    así no exponemos rutas internas ni la API key.
    💡💡
    """
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ---------------------------------------------------------------------------
# 📃✅🧪 POST /upload 📃✅🧪: Carga un documento
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse)
async def cargar_documento(doc: DocumentUpload):
    doc_id = str(uuid.uuid4())
    rag.guardar_documento(doc_id, doc.title, doc.content)

    # 🎯✋ Logueamos con el ID y el TAMAÑO, NUNCA el contenido (datos sensibles). ✋🎯
    logger.info("PASO 1 CARGA | doc_id=%s | caracteres=%d", doc_id, len(doc.content))

    return UploadResponse(
        message="Documento cargado correctamente",
        document_id=doc_id,
    )


# ---------------------------------------------------------------------------
# 📃⌛ POST /generate-embeddings 📃⌛: Genera embeddings de uno o de todos los documentos
# ---------------------------------------------------------------------------

@app.post("/generate-embeddings", response_model=EmbeddingResponse)
async def generar_embeddings(req: EmbeddingRequest):
    
    # Caso A: pidieron un documento puntual
    if req.document_id:
        if req.document_id not in rag.documentos:
            logger.warning("PASO 2 EMBEDDINGS | doc_id=%s no encontrado", req.document_id)
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        ids_a_procesar = [req.document_id]

    # Caso B: no mandaron nada -> todos los documentos cargados
    else:
        ids_a_procesar = list(rag.documentos.keys())
        if not ids_a_procesar:
            raise HTTPException(status_code=404, detail="No hay documentos cargados")

    total_chunks = 0
    try:
        for doc_id in ids_a_procesar:
            total_chunks += rag.indexar_documento(doc_id)
    except Exception as e:
        
        # 📷📷 El detalle va al log para nosotros, no a la respuesta para el usuario. 📷📷
        logger.error("PASO 2 EMBEDDINGS | fallo servicio externo: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    logger.info(
        "PASO 2 EMBEDDINGS | documentos=%d | chunks=%d",
        len(ids_a_procesar), total_chunks,
    )

    return EmbeddingResponse(
        message="Embeddings generados correctamente",
        document_id=ids_a_procesar[0] if req.document_id else None,
        documents_processed=ids_a_procesar,
        chunks_created=total_chunks,
    )


# ---------------------------------------------------------------------------
# ⌛✅📃 POST /search⌛✅📃: Busca documentos relevantes para una consulta
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def buscar_documentos(req: SearchRequest):
    try:
        encontrados = rag.buscar(req.query, top_k=req.top_k)
    except Exception as e:
        logger.error("PASO 3 BUSQUEDA | fallo servicio externo: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    resultados = [
        {
            "document_id": r["document_id"],
            "title": r["title"],
            "content_snippet": _recortar(r["chunk"], 200),
            "similarity_score": r["similarity_score"],
        }
        for r in encontrados
    ]

    # 🧪 Logueamos el largo de la consulta 🧪
    logger.info(
        "PASO 3 BUSQUEDA | largo_query=%d | resultados=%d | mejor_score=%s",
        len(req.query),
        len(resultados),
        resultados[0]["similarity_score"] if resultados else "n/a",
    )

    return SearchResponse(results=resultados)


# ---------------------------------------------------------------------------
# 🤖🧪 POST /ask — Responde a una pregunta usando el contexto recuperado 🧪🤖
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def responder_pregunta(req: AskRequest):
    
    # Cada consulta lleva su propio id para poder seguirla en los logs. ✋🧪
    traza = uuid.uuid4().hex[:8]

    # 🚨 --- Guardrail: lenguaje inapropiado  --- 🚨
    if rag.contiene_lenguaje_inapropiado(req.question):
        logger.warning("PASO 4 RESPUESTA | traza=%s | bloqueada por guardrail", traza)
        return AskResponse(
            question=req.question,
            answer=CONSULTA_BLOQUEADA,
            context_used="",
            similarity_score=0.0,
            grounded=False,
        )

    # ⌛--- Recuperación --- ⌛
    try:
        encontrados = rag.buscar(req.question, top_k=1)
    except Exception as e:
        logger.error("PASO 4 RESPUESTA | traza=%s | fallo busqueda: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    # 🎯🎯 --- Grounding: sin contexto, no inventamos --- 🎯🎯
    if not encontrados:
        logger.info("PASO 4 RESPUESTA | traza=%s | sin resultados | grounded=False", traza)
        return _respuesta_sin_contexto(req.question, 0.0)

    mejor = encontrados[0]
    if mejor["similarity_score"] < rag.UMBRAL_SIMILITUD:
        logger.info(
            "PASO 4 RESPUESTA | traza=%s | score=%.4f < umbral=%.2f | grounded=False",
            traza, mejor["similarity_score"], rag.UMBRAL_SIMILITUD,
        )
        return _respuesta_sin_contexto(req.question, mejor["similarity_score"])

    # ⌛⌛ --- Generación --- ⌛⌛
    contexto = mejor["chunk"]
    try:
        respuesta = rag.generar_respuesta(req.question, contexto)
    except Exception as e:
        logger.error("PASO 4 RESPUESTA | traza=%s | fallo LLM: %s", traza, type(e).__name__)
        raise HTTPException(status_code=503, detail=SERVICIO_CAIDO)

    # 🤖 El LLM puede decidir que no le alcanza el contexto.
    # 🤖 Si dijo eso, grounded tiene que ser False aunque el score haya pasado.
    fundamentada = SINCONTEXTO_NORMALIZADO not in respuesta.lower()

    logger.info(
        "PASO 4 RESPUESTA | traza=%s | doc_id=%s | score=%.4f | grounded=%s",
        traza, mejor["document_id"], mejor["similarity_score"], fundamentada,
    )

    return AskResponse(
        question=req.question,
        answer=respuesta,
        context_used=_recortar(contexto, 500),
        similarity_score=mejor["similarity_score"],
        grounded=fundamentada,
    )


@app.get("/")
async def raiz():
    return {
        "message": "RAG System API",
        "endpoints": ["/upload", "/generate-embeddings", "/search", "/ask"],
    }


# ---------------------------------------------------------------------------
# 💡💡 Auxiliares 💡💡
# ---------------------------------------------------------------------------

SINCONTEXTO_NORMALIZADO = SIN_CONTEXTO.lower().rstrip(".")


def _recortar(texto, largo):
    """Corta un texto y le agrega '...' si quedó cortado."""
    if len(texto) <= largo:
        return texto
    return texto[:largo] + "..."


def _respuesta_sin_contexto(pregunta, score):
    """✍✍ Respuesta estándar cuando no hay contexto suficiente.✍✍"""
    return AskResponse(
        question=pregunta,
        answer=SIN_CONTEXTO,
        context_used="",
        similarity_score=score,
        grounded=False,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)