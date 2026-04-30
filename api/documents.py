"""文档管理接口：上传、列表、删除文档"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import DocumentUploadResponse, DocumentListResponse
from services.document_service import DocumentService
from utils.logger import logger

router = APIRouter(prefix="/api/v1/documents", tags=["文档管理"])

document_service: DocumentService | None = None


def init_router(service: DocumentService):
    global document_service
    document_service = service


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="待上传的文件"),
    subject: str = Form(..., description="学科"),
    grade: str = Form("", description="年级"),
    chapter: str = Form("", description="章节"),
    strategy: str = Form("recursive", description="切片策略"),
):
    """上传文档并自动处理入库"""
    if document_service is None:
        raise HTTPException(status_code=503, detail="文档服务未初始化")

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    allowed_ext = (".pdf", ".md", ".txt")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_ext):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型，仅支持: {allowed_ext}")

    logger.info(f"上传文档: {file.filename}, subject={subject}, grade={grade}")

    try:
        content = await file.read()
        result = await document_service.upload_and_process(
            file_content=content,
            filename=file.filename,
            subject=subject,
            grade=grade,
            chapter=chapter,
            strategy=strategy,
        )
        return DocumentUploadResponse(data=result)
    except Exception as e:
        logger.error(f"文档上传处理失败: {e}")
        return DocumentUploadResponse(code=500, message=f"处理失败: {str(e)}")


@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    """获取文档列表"""
    if document_service is None:
        raise HTTPException(status_code=503, detail="文档服务未初始化")
    docs = await document_service.list_documents()
    return DocumentListResponse(data=docs, total=len(docs))


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档及其向量数据"""
    if document_service is None:
        raise HTTPException(status_code=503, detail="文档服务未初始化")
    success = await document_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"code": 0, "message": "删除成功"}
