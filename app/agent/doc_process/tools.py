"""文档处理工具 - 文件读写和向量化工具

基于 LangChain 的 @tool 装饰器定义工具，
参考 filesystem MCP 模式实现文件操作。
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from loguru import logger

from app.config import config
from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager

# 文件根目录
ROOT_DIR = Path(config.mcp_filesystem_path)
SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


def _validate_path(file_path: str) -> Path:
    """
    验证和规范化文件路径

    Args:
        file_path: 文件路径

    Returns:
        Path: 规范化后的路径

    Raises:
        ValueError: 路径不在允许的目录内或文件类型不支持
    """
    path = Path(file_path)

    # 如果是相对路径，拼接到根目录
    if not path.is_absolute():
        path = ROOT_DIR / file_path

    # 安全检查：确保路径在根目录内
    try:
        path = path.resolve()
        root_resolved = ROOT_DIR.resolve()
        if not str(path).startswith(str(root_resolved)):
            raise ValueError(f"路径不在允许的目录内: {file_path}")
    except Exception:
        raise ValueError(f"无效的路径: {file_path}")

    # 检查文件扩展名
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {path.suffix}，支持的类型: {SUPPORTED_EXTENSIONS}")

    return path


@tool(response_format="content_and_artifact")
def read_document(file_path: str) -> Tuple[str, dict]:
    """
    读取文档内容

    支持的文件格式：txt, md, docx, pdf

    Args:
        file_path: 文件路径（相对于根目录 D:\\桌面\\智能仓库，或绝对路径）

    Returns:
        Tuple[str, dict]: (文件内容, 文件元数据)
    """
    try:
        validated_path = _validate_path(file_path)

        if not validated_path.exists():
            return f"文件不存在: {file_path}", {"error": "file_not_found", "path": str(validated_path)}

        logger.info(f"读取文档: {validated_path}")

        suffix = validated_path.suffix.lower()
        content = ""
        metadata = {
            "file_path": str(validated_path),
            "file_name": validated_path.name,
            "extension": suffix,
        }

        # 根据文件类型选择加载器
        if suffix == ".txt":
            loader = TextLoader(str(validated_path), encoding="utf-8")
            docs = loader.load()
            content = "\n".join(doc.page_content for doc in docs)
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(validated_path), encoding="utf-8")
            docs = loader.load()
            content = "\n".join(doc.page_content for doc in docs)
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(validated_path))
            docs = loader.load()
            content = "\n".join(doc.page_content for doc in docs)
            metadata["total_pages"] = len(docs)
        else:
            return f"不支持的文件类型: {suffix}", {"error": "unsupported_type", "suffix": suffix}

        metadata["content_length"] = len(content)
        logger.info(f"文档读取成功: {validated_path}, 长度: {len(content)} 字符")

        return content, metadata

    except Exception as e:
        logger.error(f"读取文档失败: {file_path}, 错误: {e}")
        return f"读取文档失败: {str(e)}", {"error": str(e), "path": file_path}


@tool(response_format="content_and_artifact")
def write_document(file_path: str, content: str) -> Tuple[str, dict]:
    """
    写入文档内容到文件

    支持的文件格式：txt, md

    Args:
        file_path: 文件路径（相对于根目录 D:\\桌面\\智能仓库，或绝对路径）
        content: 要写入的内容

    Returns:
        Tuple[str, dict]: (操作结果, 文件元数据)
    """
    try:
        validated_path = _validate_path(file_path)

        # 确保父目录存在
        validated_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"写入文档: {validated_path}, 内容长度: {len(content)} 字符")

        suffix = validated_path.suffix.lower()
        if suffix not in {".txt", ".md"}:
            return f"写入失败: 不支持的文件类型 {suffix}，仅支持 txt 和 md", {
                "error": "unsupported_type",
                "suffix": suffix
            }

        # 写入文件
        with open(validated_path, "w", encoding="utf-8") as f:
            f.write(content)

        metadata = {
            "file_path": str(validated_path),
            "file_name": validated_path.name,
            "extension": suffix,
            "content_length": len(content),
            "success": True,
        }

        logger.info(f"文档写入成功: {validated_path}")
        return f"文件写入成功: {validated_path.name}", metadata

    except Exception as e:
        logger.error(f"写入文档失败: {file_path}, 错误: {e}")
        return f"写入文档失败: {str(e)}", {"error": str(e), "path": file_path}


@tool(response_format="content_and_artifact")
def list_documents(directory: str = ".") -> Tuple[str, dict]:
    """
    列出目录下的文档文件

    Args:
        directory: 目录路径（相对于根目录 D:\\桌面\\智能仓库，或绝对路径），默认为当前目录

    Returns:
        Tuple[str, dict]: (文件列表, 目录元数据)
    """
    try:
        if directory == ".":
            target_dir = ROOT_DIR
        else:
            target_dir = ROOT_DIR / directory
            target_dir = target_dir.resolve()

        # 安全检查
        if not str(target_dir).startswith(str(ROOT_DIR.resolve())):
            return "路径不在允许的目录内", {"error": "path_outside_root", "path": str(directory)}

        if not target_dir.exists():
            return f"目录不存在: {directory}", {"error": "directory_not_found", "path": str(target_dir)}

        # 递归列出所有支持的文件
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(target_dir.rglob(f"*{ext}"))

        # 转换为相对路径列表
        relative_files = [str(f.relative_to(ROOT_DIR)) for f in files]

        metadata = {
            "directory": str(target_dir.relative_to(ROOT_DIR)),
            "total_files": len(relative_files),
            "extensions": list(SUPPORTED_EXTENSIONS),
        }

        if relative_files:
            file_list_str = "\n".join(sorted(relative_files))
            return f"找到 {len(relative_files)} 个文档文件:\n{file_list_str}", metadata
        else:
            return "目录下没有文档文件", metadata

    except Exception as e:
        logger.error(f"列出文档失败: {directory}, 错误: {e}")
        return f"列出文档失败: {str(e)}", {"error": str(e)}


@tool(response_format="content_and_artifact")
def vectorize_document(file_path: str, description: str = "") -> Tuple[str, dict]:
    """
    将文档内容向量化并存入 Milvus 向量数据库

    Args:
        file_path: 文件路径（相对于根目录 D:\\桌面\\智能仓库，或绝对路径）
        description: 可选的文档描述（用于增强检索效果）

    Returns:
        Tuple[str, dict]: (操作结果, 向量化元数据)
    """
    try:
        # 先读取文档内容
        content, read_metadata = read_document.invoke(file_path)

        if "error" in read_metadata:
            return f"读取文档失败，无法向量化", read_metadata

        # 分割文档
        validated_path = _validate_path(file_path)
        docs = document_splitter_service.split_document(content, str(validated_path))

        if not docs:
            return "文档为空，无法向量化", {"error": "empty_document", "path": str(validated_path)}

        # 添加描述到元数据（如果有）
        if description:
            for doc in docs:
                doc.metadata["_description"] = description

        # 删除旧文档（如果存在）
        vector_store_manager.delete_by_source(str(validated_path))

        # 批量添加文档到向量存储
        ids = vector_store_manager.add_documents(docs)

        metadata = {
            "file_path": str(validated_path),
            "file_name": validated_path.name,
            "chunks_count": len(docs),
            "vector_ids": ids[:5] if len(ids) > 5 else ids,  # 只返回前5个ID示例
            "total_ids": len(ids),
            "description": description,
        }

        logger.info(f"文档向量化完成: {validated_path}, 分片数: {len(docs)}")
        return f"文档向量化成功: {validated_path.name}, 生成 {len(docs)} 个向量", metadata

    except Exception as e:
        logger.error(f"文档向量化失败: {file_path}, 错误: {e}")
        return f"向量化失败: {str(e)}", {"error": str(e), "path": file_path}


@tool(response_format="content_and_artifact")
def search_vectorized_docs(query: str, top_k: int = 5) -> Tuple[str, dict]:
    """
    在已向量化的文档中搜索相关内容

    Args:
        query: 搜索查询文本
        top_k: 返回的结果数量，默认为 5

    Returns:
        Tuple[str, dict]: (搜索结果, 元数据)
    """
    try:
        vector_store = vector_store_manager.get_vector_store()
        docs = vector_store.similarity_search(query, k=top_k)

        if not docs:
            return "没有找到相关文档", {"query": query, "results_count": 0}

        # 格式化结果
        results = []
        for i, doc in enumerate(docs, 1):
            results.append({
                "rank": i,
                "file_name": doc.metadata.get("_file_name", "未知"),
                "source": doc.metadata.get("_source", ""),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

        results_text = f"找到 {len(docs)} 个相关文档:\n\n"
        for r in results:
            results_text += f"【{r['rank']}】{r['file_name']}\n"
            results_text += f"来源: {r['source']}\n"
            results_text += f"内容: {r['content_preview']}\n\n"

        metadata = {
            "query": query,
            "results_count": len(docs),
            "top_k": top_k,
        }

        return results_text, metadata

    except Exception as e:
        logger.error(f"搜索文档失败: {query}, 错误: {e}")
        return f"搜索失败: {str(e)}", {"error": str(e)}


__all__ = [
    "read_document",
    "write_document",
    "list_documents",
    "vectorize_document",
    "search_vectorized_docs",
]