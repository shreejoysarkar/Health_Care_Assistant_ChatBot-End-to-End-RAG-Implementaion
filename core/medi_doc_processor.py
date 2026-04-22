"""
processing text from different types of pdf files.
"""

from pathlib import Path


from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


