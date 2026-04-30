"""Document processing module for loading and chunking documents."""

import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """process documents for RAG pipeline"""

    SUPPORTED_EXTENSIONS = {".pdf",".txt", ".csv"}

    def __init__(
            self,
            chunk_size :int | None = None,
            chunk_overlap :int | None = None,
    ):
        """initialize document processor.
        
        Args:
            chunk_size (int | None, optional)
            chunk_overlap (int | None, optional)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        logger.info(
            f"DocumentProcessor initialized with chunk_size = {self.chunk_size}, "
            f"chunk_overlap = {self.chunk_overlap}"
        )
        


    def load_pdf(self, file_path :str | Path) -> list[Document]:
        """
        Load a Pdf
        
        Args : 
        file_path : Path to pdf file

        returns: list of Documents
        """

        file_path = Path(file_path)
        logger.info(f"loading PDF : {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        logger.info(f"loaded {len(documents)} pages from PDF : {file_path.name}")
        return documents
    

    
    def split_documents(self, documents : list[Document]) -> list[Document]:
        """split documents into chunks
        
        Args: 
            documents : list of document object
            
        Returns : 
            List of chunked document objects
            
            """
        logger.info(f"splitting {len(documents)} documents into chunks")
        
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"split into {len(chunks)} chunks")

        return chunks

    def process_file(self, file_path : str | Path) -> list[Document]:
        """load and split the file in one step

        Args:
            file_path (str | Path): path to file

        Returns:
            list[Document]: list of chunked document objects
        """

        Document = self.load_file(file_path)
        chunks = self.split_documents(Document)

        return chunks

