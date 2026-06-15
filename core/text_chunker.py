import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# Ensure the project root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import chonkie components
from chonkie import RecursiveChunker, TableChunker
from chonkie.refinery import OverlapRefinery

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Complete mapping of chapters in Davidson's Principles & Practice of Medicine
CHAPTER_MAPPING = {
    1: "Clinical decision-making",
    2: "Clinical therapeutics and good prescribing",
    3: "Clinical genetics",
    4: "Clinical immunology",
    5: "Population health and epidemiology",
    6: "Principles of infectious disease",
    7: "Poisoning",
    8: "Envenomation",
    9: "Environmental medicine",
    10: "Acute medicine and critical illness",
    11: "Infectious disease",
    12: "HIV infection and AIDS",
    13: "Sexually transmitted infections",
    14: "Clinical biochemistry and metabolic medicine",
    15: "Nephrology and urology",
    16: "Cardiology",
    17: "Respiratory medicine",
    18: "Endocrinology",
    19: "Nutritional factors in disease",
    20: "Diabetes mellitus",
    21: "Gastroenterology",
    22: "Hepatology",
    23: "Haematology and transfusion medicine",
    24: "Rheumatology and bone disease",
    25: "Neurology",
    26: "Stroke medicine",
    27: "Medical ophthalmology",
    28: "Medical psychiatry",
    29: "Dermatology",
    30: "Maternal medicine",
    31: "Adolescent and transition medicine",
    32: "Ageing and disease",
    33: "Oncology",
    34: "Pain and palliative care",
    35: "Laboratory reference ranges"
}

def find_chapter_by_name(heading_text: str) -> tuple[int | None, str | None]:
    """Helper to check if a heading text matches a chapter name directly."""
    clean_text = heading_text.strip().lower().rstrip(" *")
    for num, name in CHAPTER_MAPPING.items():
        if clean_text == name.lower():
            return num, name
    for num, name in CHAPTER_MAPPING.items():
        if f"{num}. {name.lower()}" in clean_text or f"chapter {num}" in clean_text:
            return num, name
    return None, None

def segment_markdown(text: str) -> List[Dict[str, Any]]:
    """
    Parse a markdown file line-by-line using a state machine.
    Splits the text into blocks of either 'text' or 'table', tracking
    source, chapter, section, and subsection contextual headings.
    """
    lines = text.splitlines(keepends=True)
    blocks = []
    current_lines = []
    current_type = "text"

    # Default metadata state variables
    current_source = "medical_data.md"
    current_chapter = "General"
    current_section = ""
    current_subsection = ""

    def flush_block():
        nonlocal current_lines, current_type
        if not current_lines:
            return
        content = "".join(current_lines).strip()
        if content:
            blocks.append({
                "type": current_type,
                "content": content,
                "metadata": {
                    "source": current_source,
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection
                }
            })
        current_lines = []

    for line in lines:
        stripped = line.strip()

        # 1. Check for original document source markers
        if stripped.startswith("# --- Source:") and stripped.endswith("---"):
            flush_block()
            parts = stripped.split("Source:")
            if len(parts) > 1:
                current_source = parts[1].replace("---", "").strip()
            current_type = "text"
            continue

        # 2. Check for markdown headings
        is_heading = False
        heading_level = 0
        if stripped.startswith("# "):
            is_heading = True
            heading_level = 1
        elif stripped.startswith("## "):
            is_heading = True
            heading_level = 2
        elif stripped.startswith("### "):
            is_heading = True
            heading_level = 3
        elif stripped.startswith("#### "):
            is_heading = True
            heading_level = 4

        if is_heading:
            flush_block()
            current_lines = [line]
            current_type = "text"

            # Clean the heading text
            heading_text = stripped.lstrip("#").strip()

            # State updates based on headings
            num_match = re.match(r"^(\d+)(?:\.(\d+))?\s+(.*)$", heading_text)
            if num_match:
                chapter_num = int(num_match.group(1))
                section_title = num_match.group(3) if num_match.group(2) else heading_text
                
                if chapter_num in CHAPTER_MAPPING:
                    current_chapter = CHAPTER_MAPPING[chapter_num]
                
                if num_match.group(2): # e.g. "11.1" -> chapter.section
                    current_section = section_title
                    current_subsection = ""
                else: # single chapter number heading
                    current_section = heading_text
                    current_subsection = ""
            else:
                # Check for exact mapping match by name
                num, name = find_chapter_by_name(heading_text)
                if num is not None:
                    current_chapter = name
                    current_section = ""
                    current_subsection = ""
                else:
                    if heading_level == 2:
                        current_section = heading_text
                        current_subsection = ""
                    elif heading_level == 3:
                        current_subsection = heading_text
                    elif heading_level == 4:
                        if current_subsection:
                            current_subsection += " - " + heading_text
                        else:
                            current_subsection = heading_text
            continue

        # 3. Check for tables (lines starting and ending with '|')
        is_table_line = False
        if stripped.startswith("|") and stripped.endswith("|"):
            is_table_line = True

        if is_table_line:
            if current_type != "table":
                flush_block()
                current_type = "table"
            current_lines.append(line)
        else:
            if current_type == "table":
                flush_block()
                current_type = "text"
            current_lines.append(line)

    flush_block()
    return blocks

def classify_context(heading_text: str, chunk_text: str) -> Dict[str, str]:
    """
    Analyzes heading and chunk content to identify demographic focus
    and clinical context for filtering and retrieval.
    """
    combined = (heading_text + " " + chunk_text).lower()

    # Demographic focus classification
    if any(k in combined for k in ["in old age", "elderly", "geriatric", "ageing and disease", "in old"]):
        demo = "Geriatric"
    elif any(k in combined for k in ["in pregnancy", "pregnant", "maternal", "obstetrics"]):
        demo = "Pregnancy"
    elif any(k in combined for k in ["in adolescence", "adolescent", "teenage", "adolescents"]):
        demo = "Adolescent"
    else:
        demo = "General"

    # Clinical context classification
    if any(k in combined for k in ["emergency", "acute", "life-threatening", "critical illness", "poisoning", "envenomation"]):
        clinical = "Emergency"
    elif any(k in combined for k in ["practice point", "practice points"]):
        clinical = "Practice Point"
    else:
        clinical = "Standard"

    return {
        "demographic_focus": demo,
        "clinical_context": clinical
    }


class DocumentChunker:
    """Handles splitting of markdown documents into text and table chunks with stateful metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 90):
        """Initialize the document chunker and tokenizer configuration."""
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.tokenizer_name = settings.embedding_model or "BAAI/bge-m3"

        # Initialize local RecursiveChunker
        self.text_chunker = RecursiveChunker(
            tokenizer=self.tokenizer_name,
            chunk_size=self.chunk_size
        )
        # Initialize OverlapRefinery for the text chunks
        self.overlap_refinery = OverlapRefinery(
            tokenizer=self.tokenizer_name,
            context_size=self.chunk_overlap,
            mode="token",
            method="prefix",
            merge=True,
            inplace=True
        )
        # Initialize TableChunker (row-based table chunking, chunk size = 5 rows per chunk)
        self.table_chunker = TableChunker(
            tokenizer="row",
            chunk_size=5
        )

        logger.info(
            f"Initialized Hybrid Chunker with tokenizer={self.tokenizer_name}, "
            f"text_chunk_size={self.chunk_size}, text_chunk_overlap={self.chunk_overlap}"
        )

    def chunk_markdown_content(self, text: str) -> List[Dict[str, Any]]:
        """Segment and chunk markdown content using hybrid chunking."""
        logger.info("Segmenting document...")
        segments = segment_markdown(text)
        logger.info(f"Segmented into {len(segments)} blocks.")

        all_chunks = []
        
        # Keep track of running statistics
        text_blocks_count = 0
        table_blocks_count = 0

        for block in segments:
            meta = block["metadata"]
            heading_context = f"{meta['chapter']} > {meta['section']} > {meta['subsection']}".strip(" > ")
            
            if block["type"] == "table":
                # Use TableChunker for table blocks
                table_blocks_count += 1
                try:
                    chunks = self.table_chunker.chunk(block["content"])
                    if not chunks:
                        # Fallback: if TableChunker skipped, treat the table as a single chunk
                        # to ensure no data is lost
                        toks = self.text_chunker.tokenizer.count_tokens(block["content"])
                        context_tags = classify_context(heading_context, block["content"])
                        all_chunks.append({
                            "text": block["content"],
                            "metadata": {
                                "source": meta["source"],
                                "chapter": meta["chapter"],
                                "section": meta["section"],
                                "subsection": meta["subsection"],
                                "chunk_type": "table",
                                "token_count": toks,
                                "demographic_focus": context_tags["demographic_focus"],
                                "clinical_context": context_tags["clinical_context"]
                            }
                        })
                    else:
                        for chunk in chunks:
                            # Count actual embedding tokens
                            toks = self.text_chunker.tokenizer.count_tokens(chunk.text)
                            
                            context_tags = classify_context(heading_context, chunk.text)
                            all_chunks.append({
                                "text": chunk.text,
                                "metadata": {
                                    "source": meta["source"],
                                    "chapter": meta["chapter"],
                                    "section": meta["section"],
                                    "subsection": meta["subsection"],
                                    "chunk_type": "table",
                                    "token_count": toks,
                                    "demographic_focus": context_tags["demographic_focus"],
                                    "clinical_context": context_tags["clinical_context"]
                                }
                            })
                except Exception as e:
                    logger.error(f"Error chunking table block: {e}")
            else:
                # Use RecursiveChunker for text blocks
                text_blocks_count += 1
                try:
                    chunks = self.text_chunker.chunk(block["content"])
                    if chunks:
                        # Refine chunks to add overlap context
                        chunks = self.overlap_refinery.refine(chunks)
                        for chunk in chunks:
                            context_tags = classify_context(heading_context, chunk.text)
                            all_chunks.append({
                                "text": chunk.text,
                                "metadata": {
                                    "source": meta["source"],
                                    "chapter": meta["chapter"],
                                    "section": meta["section"],
                                    "subsection": meta["subsection"],
                                    "chunk_type": "text",
                                    "token_count": chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text),
                                    "demographic_focus": context_tags["demographic_focus"],
                                    "clinical_context": context_tags["clinical_context"]
                                }
                            })
                except Exception as e:
                    logger.error(f"Error chunking text block: {e}")

        logger.info(
            f"Chunking complete. Processed {text_blocks_count} text blocks and "
            f"{table_blocks_count} table blocks. Total chunks: {len(all_chunks)}"
        )
        return all_chunks

    def process_all_markdowns(self, input_dir: str = "Data/Output", output_dir: str = "Data/chunks") -> List[Dict[str, Any]]:
        """Iterate over processed markdown files, chunk them, and save to output directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            logger.error(f"Input directory {input_path} does not exist.")
            return []
            
        md_files = list(input_path.glob("medical_data.md"))
        if not md_files:
            logger.warning(f"No medical_data.md file found in {input_path}")
            return []
            
        all_chunks = []
        for md_file in md_files:
            logger.info(f"Processing medical file: {md_file}")
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    text = f.read()
                
                chunks = self.chunk_markdown_content(text)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")

        # Save all chunks as a structured JSON file in Data/chunks folder
        output_file = output_path / "medical_data_chunks.json"
        logger.info(f"Saving {len(all_chunks)} chunks to {output_file}")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved chunked data to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save chunks JSON: {e}")

        return all_chunks

if __name__ == "__main__":
    # Configure logging level to see full info
    import logging
    logging.basicConfig(level=logging.INFO)

    chunker = DocumentChunker()
    all_extracted_chunks = chunker.process_all_markdowns()
    
    if all_extracted_chunks:
        # Get a random chunk to display
        import random
        sample_idx = random.randint(0, len(all_extracted_chunks) - 1)
        sample = all_extracted_chunks[sample_idx]
        
        print("\n=== SAMPLE CHUNK INGESTION VERIFICATION ===")
        print(f"Index: {sample_idx}")
        print(f"Source Document: {sample['metadata']['source']}")
        print(f"Chapter: {sample['metadata']['chapter']}")
        print(f"Section: {sample['metadata']['section']}")
        print(f"Subsection: {sample['metadata']['subsection']}")
        print(f"Type: {sample['metadata']['chunk_type']}")
        print(f"Tokens: {sample['metadata']['token_count']}")
        print(f"Demographic Focus: {sample['metadata']['demographic_focus']}")
        print(f"Clinical Context: {sample['metadata']['clinical_context']}")
        print(f"Text Preview:\n{sample['text'][:400]}...")
        print("==========================================")
        print(f"Total chunks created and saved: {len(all_extracted_chunks)}")