'''
This code utilizes the Docling library to convert medical PDF documents into Markdown format. 
It employs a GPU-accelerated pipeline to process multiple files efficiently. 
The converter is configured to enable remote services, disable OCR, 
and perform detailed table structure and picture description generation for enhanced data extraction.


File is run on the Google Colab.
'''


import os
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice
)

# 1. Setup the GPU Pipeline
pipeline_options = PdfPipelineOptions(
    enable_remote_services=True,
    do_ocr=False,
    do_table_structure=True,
    generate_picture_images=True,
    do_picture_description=True,
    table_structure_options=TableStructureOptions(
        mode=TableFormerMode.ACCURATE
    ),
    accelerator_options=AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA # Forces GPU usage
    )
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
        )
    }
)

# 2. Process the Documents
output_dir = Path("Output")
output_dir.mkdir(parents=True, exist_ok=True)

# Find all PDFs in the current Colab folder
pdf_files = list(Path('.').glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files to process.")

for pdf_file in pdf_files:
    print(f"Processing: {pdf_file.name}...")
    try:
        # Convert PDF
        result = converter.convert(pdf_file)
        markdown_content = result.document.export_to_markdown()
        
        # Save Markdown
        output_file = output_dir / f"{pdf_file.stem}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"✅ Successfully saved: {output_file.name}")
    except Exception as e:
        print(f"❌ Error processing {pdf_file.name}: {e}")

print("\n🎉 All processing complete!")
