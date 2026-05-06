import os
import sys
from pathlib import Path

# Ensure the project root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import chonkie. (If you want to use SemanticChunker, you can import it here instead)
from chonkie import TokenChunker

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentChunker:
    """Handles splitting of markdown documents into smaller chunks using the Chonkie library."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize the document chunker."""
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize the Chonkie TokenChunker
        # You can easily swap this out for SemanticChunker if needed later!
        self.chunker = TokenChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"Initialized Chonkie TokenChunker with size={self.chunk_size}, overlap={self.chunk_overlap}")

    def chunk_markdown_file(self, file_path: str | Path):
        """Read a single markdown file and split it into chunks."""
        file_path = Path(file_path)
        logger.info(f"Chunking file: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            # Use chonkie to chunk the text
            chunks = self.chunker.chunk(text)
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking {file_path}: {e}")
            return []
            
    def process_all_markdowns(self, input_dir: str = "Data/Output"):
        """Iterate over all processed markdown files and chunk them."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory {input_path} does not exist. Make sure docling has finished processing.")
            return []
            
        # Only look for the specific medical_data.md file
        md_files = list(input_path.glob("medical_data.md"))
        
            
        if not md_files:
            logger.warning(f"No medical_data.md file found in {input_path}")
            return []
            
        all_chunks = []
        for md_file in md_files:
            chunks = self.chunk_markdown_file(md_file)
            
            # Here you could also format the chunks to prepare them for vector DB ingestion.
            # e.g., storing metadata like the source filename with the chunk text.
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk.text,
                    "metadata": {
                        "source": md_file.name,
                        "token_count": chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text)
                    }
                })
            
        logger.info(f"Finished chunking. Total chunks created across all documents: {len(all_chunks)}")
        return all_chunks

if __name__ == "__main__":
    chunker = DocumentChunker()
    all_extracted_chunks = chunker.process_all_markdowns()
    
    # Just print a tiny sample of the first chunk to verify it works!
    if all_extracted_chunks:
        print("\n--- SAMPLE CHUNK ---")
        print(f"Source: {all_extracted_chunks[534]['metadata']['source']}")
        print(f"Text: {all_extracted_chunks[534]['text'][:300]}...")
        print(f"metadata: {all_extracted_chunks[534]['metadata']}")

        print(f"Total chunks created across all documents: {len(all_extracted_chunks)}")