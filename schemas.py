"""
✋✋
schemas.py — Define la forma de los datos que entran y salen de la API.
✋✋
FastAPI usa estas clases para validar automáticamente. Si alguien manda
un título vacío, ni siquiera llega al endpoint: devuelve 422 solo. 
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# ✍✍✅✅ Entradas ✅✅✍✍
# ---------------------------------------------------------------------------

class DocumentUpload(BaseModel):
    """POST /upload"""
    title: str = Field(..., min_length=1, description="Título del documento")
    content: str = Field(..., min_length=1, description="Contenido del documento")

    @field_validator("title", "content")
    @classmethod
    def sin_espacios_vacios(cls, valor):
        # min_length=1 no alcanza: "   " tiene largo 3 pero está vacío.
        limpio = valor.strip()
        if not limpio:
            raise ValueError("El campo no puede estar vacío")
        return limpio


class EmbeddingRequest(BaseModel):
    """
    ✋
    POST /generate-embeddings

    document_id si no lo mandas se procesan todos los documentos. 😊
    """
    document_id: Optional[str] = Field(
        None, description="ID del documento. Si se omite, procesa todos."
    )

    @field_validator("document_id")
    @classmethod
    def sin_espacios_vacios(cls, valor):
        if valor is None:
            return None
        limpio = valor.strip()
        if not limpio:
            raise ValueError("El document_id no puede estar vacío")
        return limpio


class SearchRequest(BaseModel):
    """POST /search"""
    query: str = Field(..., min_length=1, description="Consulta en lenguaje natural")
    top_k: int = Field(3, ge=1, le=10, description="Cantidad de resultados")

    @field_validator("query")
    @classmethod
    def sin_espacios_vacios(cls, valor):
        limpio = valor.strip()
        if not limpio:
            raise ValueError("La consulta no puede estar vacía")
        return limpio


class AskRequest(BaseModel):
    """POST /ask"""
    question: str = Field(..., min_length=1, description="Pregunta del usuario")

    @field_validator("question")
    @classmethod
    def sin_espacios_vacios(cls, valor):
        limpio = valor.strip()
        if not limpio:
            raise ValueError("La pregunta no puede estar vacía")
        return limpio


# ---------------------------------------------------------------------------
# ✅✅ Salidas ✅✅ 
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    message: str
    document_id: str


class EmbeddingResponse(BaseModel):
    """
    document_id se completa cuando se procesó un (1) documento. 📃
    documents_processed lista todos los procesados (uno o varios). 📃📃📃📃
    """
    message: str
    document_id: Optional[str] = None
    documents_processed: List[str] = []
    chunks_created: int = 0


class SearchResult(BaseModel):
    document_id: str
    title: str
    content_snippet: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    results: List[SearchResult]


class AskResponse(BaseModel):
    """
    🧪🧪🧪🧪
    context_used: Es el contexto que se usó para responder.
    similarity_score: Es la similitud que obtuvo el LLM para responder.
    grounded: Si la respuesta se apoyó en contexto real o no.
    🧪🧪🧪🧪
    """
    question: str
    answer: str
    context_used: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    grounded: bool


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None