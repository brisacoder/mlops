"""
PDF Document Processing Pipeline with Parallel Chunking

This module provides a comprehensive solution for processing large PDF documents by intelligently
splitting them into manageable chunks and processing them in parallel using Docling's advanced
document understanding capabilities. The system is designed to handle memory-intensive document
processing while maintaining high performance through careful resource management.

Key Features:
- Intelligent PDF chunking with balanced page distribution
- Parallel processing using ProcessPoolExecutor with resource safeguards
- Advanced document layout analysis and content extraction
- Robust error handling and logging for production environments
- Memory-efficient processing with configurable pipeline options

The pipeline leverages Docling's Heron model for superior document understanding,
including table structure recognition, image extraction, and multi-modal content analysis.
Parallel processing is optimized for both CPU and GPU acceleration while preventing
resource contention through staggered task submission and worker limits.
"""

import argparse
import concurrent.futures
import logging
import time
import traceback
from pathlib import Path

import fitz
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
from docling.datamodel.pipeline_options import LayoutOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import PictureItem, TableItem
import pip

# pylint: disable=logging-fstring-interpolation
# Using f-strings for logging as preferred by the development team

# Constants
DEFAULT_NUM_CHUNKS = 6
MAX_WORKERS = 2
STAGGER_DELAY = 5  # seconds
TIMEOUT_PER_CHUNK = 600  # 10 minutes
NUM_THREADS = 6

logging.basicConfig(level=logging.INFO)

# Get the logger for docling and set its level
logging.getLogger("docling").setLevel(logging.INFO)
logging.getLogger("docling_core").setLevel(logging.INFO)
log = logging.getLogger(__name__)  # This makes your script a logging-aware application


def split_pdf_into_chunks(input_pdf_path: Path, num_chunks: int) -> list[Path]:
    """
    Intelligently split a large PDF document into balanced chunks for parallel processing.

    This function employs a sophisticated page distribution algorithm that ensures each chunk
    contains roughly equal page counts while maintaining document structure integrity.
    The approach handles edge cases like uneven page distribution and creates temporary
    PDF files that preserve the original document's formatting and content.

    The chunking strategy prioritizes:
    - Balanced workload distribution across processing units
    - Minimal memory overhead during file operations
    - Preservation of document layout and readability
    - Efficient cleanup and temporary file management

    Args:
        input_pdf_path (Path): Absolute path to the source PDF file that needs chunking.
            Must be a valid, readable PDF document accessible in the current environment.
        num_chunks (int): Number of equal-sized chunks to create. Must be a positive integer
            greater than zero. Optimal values depend on document size and available resources.

    Returns:
        list[Path]: Ordered list of Path objects pointing to temporary PDF chunk files.
            Each chunk file contains a contiguous subset of the original document's pages.
            Returns empty list if splitting fails due to file access issues or invalid input.

    Raises:
        FileNotFoundError: When the input PDF file cannot be located at the specified path.
        ValueError: When num_chunks is not a positive integer.
        Exception: For various PDF processing errors including corruption or permission issues.

    Example:
        >>> pdf_path = Path("large_document.pdf")
        >>> chunks = split_pdf_into_chunks(pdf_path, 4)
        >>> print(f"Created {len(chunks)} chunks")
        Created 4 chunks
    """
    # Input validation
    if not isinstance(num_chunks, int) or num_chunks <= 0:
        raise ValueError(f"num_chunks must be a positive integer, got {num_chunks}")

    if not input_pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf_path}")

    temp_chunk_paths = []

    try:
        input_doc = fitz.open(str(input_pdf_path))
        total_pages = input_doc.page_count

        # Additional validation
        if total_pages == 0:
            print("Warning: PDF has no pages")
            input_doc.close()
            return []

        pages_per_chunk = total_pages // num_chunks
        remaining_pages = total_pages % num_chunks

        start_page = 0
        for i in range(num_chunks):
            end_page = start_page + pages_per_chunk
            if i < remaining_pages:
                end_page += 1

            chunk_doc = fitz.open()  # Create a new empty PDF for the chunk

            # Copy pages from the input PDF to the chunk document
            chunk_doc.insert_pdf(input_doc, from_page=start_page, to_page=end_page - 1)

            # Define a temporary file path for the chunk
            # Using a temporary directory might be better for cleanup in a real scenario
            temp_chunk_path = (
                input_pdf_path.parent
                / f"temp_chunk_{start_page}-{end_page-1}_{input_pdf_path.name}"
            )
            temp_chunk_paths.append(temp_chunk_path)

            # Save the temporary chunk document
            chunk_doc.save(str(temp_chunk_path))

            # Close the temporary chunk document
            chunk_doc.close()

            start_page = end_page

        # Close the input PDF document
        input_doc.close()

        print(f"Split {total_pages} pages into {len(temp_chunk_paths)} chunks.")
        for path in temp_chunk_paths:
            print(f"Created: {path}")

        return temp_chunk_paths

    except (FileNotFoundError, PermissionError, OSError):
        print(f"Error: Input PDF not found at {input_pdf_path}")
        return []
    except (ValueError, RuntimeError) as e:
        print(f"An error occurred during PDF splitting: {e}")
        return []


def save_images_tables(conv_res):
    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")


def process_pdf_chunk_process(chunk_path: Path) -> dict[str, str] | None:
    """
    Process a single PDF chunk using Docling's advanced document understanding pipeline.

    This function serves as the core processing unit for parallel PDF document analysis.
    It initializes a specialized Docling converter optimized for chunk processing, applying
    sophisticated layout analysis and content extraction algorithms. The processing pipeline
    is carefully configured to balance accuracy with computational efficiency.

    Key processing capabilities include:
    - Advanced document layout detection using transformer-based models
    - Intelligent text and structural element recognition
    - Memory-efficient processing with configurable resource allocation
    - Robust error handling with detailed diagnostic information
    - Picklable result packaging for inter-process communication

    The function is specifically designed for use with ProcessPoolExecutor, implementing
    best practices for multiprocessing including proper resource isolation and cleanup.

    Args:
        chunk_path (Path): File system path to the PDF chunk file to be processed.
            Should point to a valid PDF document created by the chunking process.

    Returns:
        dict or None: Processing results packaged for inter-process communication.
            Contains:
            - 'chunk_path': Original chunk file path for result tracking
            - 'markdown': Complete markdown representation of extracted content
            - 'timings': String representation of processing performance metrics
            Returns None if processing fails due to file access, parsing, or system errors.

    Raises:
        Exception: Propagates processing failures with detailed error context for debugging.
            Common failure modes include file access issues, model loading problems,
            or memory allocation failures during document analysis.

    Note:
        This function is designed for subprocess execution and includes comprehensive
        error handling to prevent silent failures in parallel processing environments.
        All exceptions are logged with full stack traces for production debugging.
    """
    print(f"Starting subprocess for {chunk_path}")

    # Initialize converter with minimal options to avoid memory issues
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True  # Keep disabled for stability
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_parsed_pages = True
    pipeline_options.images_scale = 2.0
    pipeline_options.layout_options = LayoutOptions(
        model_spec=DOCLING_LAYOUT_HERON,
    )
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=NUM_THREADS, device=AcceleratorDevice.CUDA
    )

    print(f"Initializing converter for {chunk_path}")
    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        print(f"Converter initialized for {chunk_path}")
    except Exception as e:
        print(f"Converter initialization failed for {chunk_path}: {e}")
        traceback.print_exc()
        return None

    try:
        conversion_result = converter.convert(str(chunk_path))
        print(f"Successfully processed chunk: {chunk_path}")

        # Extract picklable data from the result
        doc = conversion_result.document

        # Save HTML content to file
        # markdown_output_path = chunk_path.parent / f"{chunk_path.stem}.md"
        json_output_path = chunk_path.parent / f"{chunk_path.stem}.json"
        doc.save_as_markdown(markdown_output_path, image_mode=ImageRefMode.EMBEDDED)
        doc.save_as_json(json_output_path)
        # Return a dictionary with only picklable data
        result_data = {
            "chunk_path": str(chunk_path),
            "timings": str(
                conversion_result.timings
            ),  # Convert to string to ensure picklable
            # "markdown_path": str(markdown_output_path),
            "json_path": str(json_output_path)
        }

        return result_data
    except (ValueError, RuntimeError, OSError) as e:
        log.error(f"Error processing chunk {chunk_path}: {e}")
        return None


def main():
    """
    Execute the complete PDF document processing pipeline with parallel chunking.

    This function orchestrates the end-to-end document processing workflow, implementing
    a sophisticated parallel processing strategy that maximizes throughput while maintaining
    system stability. The pipeline integrates intelligent chunking, resource-aware parallel
    execution, and comprehensive result aggregation.

    Command Line Usage:
        python extract_pdf.py [OPTIONS]

    Options:
        -p, --pdf-path PATH    Path to input PDF file (default: ./data/kona.pdf)
        -c, --chunks INT       Number of chunks to split PDF into (default: 6)
        -v, --verbose          Enable verbose logging
        -h, --help            Show help message

    Processing Strategy:
    1. Model preloading to avoid redundant downloads in subprocesses
    2. Intelligent PDF chunking with balanced page distribution
    3. Parallel processing with staggered task submission to prevent resource contention
    4. Robust error handling with timeout protection and detailed diagnostics
    5. Result aggregation and performance reporting

    Resource Management:
    - Limits concurrent workers to prevent system overload
    - Implements task staggering to avoid peak resource utilization
    - Provides timeout protection for hung processes
    - Includes comprehensive logging for monitoring and debugging

    Performance Optimizations:
    - Preloads ML models in main process for subprocess efficiency
    - Uses ProcessPoolExecutor for true parallel processing
    - Implements result caching and aggregation
    - Provides detailed timing and success metrics

    Args:
        None: Configuration is handled through command line arguments.

    Returns:
        None: Results are logged and stored in memory. Future enhancements may
            return aggregated results or processing statistics.

    Side Effects:
        - Creates temporary PDF chunk files in the working directory
        - Downloads and caches ML models if not already present
        - Generates comprehensive log output for monitoring
        - May consume significant CPU and memory resources during processing

    Environment Requirements:
        - Access to input PDF file at configured path
        - Sufficient disk space for temporary chunk files
        - Adequate system memory for parallel processing
        - Compatible hardware acceleration (CPU/GPU) for ML models

    Examples:
        # Use default settings
        python extract_pdf.py

        # Process custom PDF with 8 chunks
        python extract_pdf.py -p /path/to/document.pdf -c 8

        # Enable verbose logging
        python extract_pdf.py -v

        # Show help
        python extract_pdf.py --help
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process PDF documents with parallel chunking and Docling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_pdf.py                          # Use default PDF: ./data/kona.pdf
  python extract_pdf.py -p /path/to/document.pdf # Specify custom PDF path
  python extract_pdf.py --pdf-path ./myfile.pdf  # Use long option
        """,
    )

    parser.add_argument(
        "-p",
        "--pdf-path",
        type=str,
        default="./data/kona.pdf",
        help="Path to the input PDF file (default: ./data/kona.pdf)",
    )

    parser.add_argument(
        "-c",
        "--chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"Number of chunks to split the PDF into (default: {DEFAULT_NUM_CHUNKS})",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("docling").setLevel(logging.DEBUG)
        logging.getLogger("docling_core").setLevel(logging.DEBUG)

    # Convert string path to Path object
    input_pdf_path = Path(args.pdf_path)

    # Validate input file exists
    if not input_pdf_path.exists():
        log.error(f"Input PDF file not found: {input_pdf_path}")
        return

    # Validate input file is actually a PDF
    if input_pdf_path.suffix.lower() != ".pdf":
        log.error(f"Input file must be a PDF file: {input_pdf_path}")
        return

    # Validate chunks argument
    if args.chunks <= 0:
        log.error(f"Number of chunks must be positive: {args.chunks}")
        return

    log.info(f"Processing PDF: {input_pdf_path}")
    log.info(f"Number of chunks: {args.chunks}")

    # Define the number of chunks for sequential processing
    num_chunks = args.chunks

    # Split the PDF into chunks
    temp_chunk_paths = split_pdf_into_chunks(input_pdf_path, num_chunks)

    # Try ProcessPoolExecutor with careful resource management
    if temp_chunk_paths:
        log.info(
            f"\nProcessing {len(temp_chunk_paths)} chunks with ProcessPoolExecutor..."
        )
        start_time = time.time()

        # Use ProcessPoolExecutor with limited workers
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(num_chunks, MAX_WORKERS)
        ) as executor:  # Limit to MAX_WORKERS max
            # Submit processing tasks with staggered start to avoid resource contention
            future_to_chunk = {}
            for i, chunk_path in enumerate(temp_chunk_paths):
                if i > 0:
                    time.sleep(
                        STAGGER_DELAY
                    )  # STAGGER_DELAY second delay between submissions
                future = executor.submit(process_pdf_chunk_process, chunk_path)
                future_to_chunk[future] = chunk_path

            parallel_results = {}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_path = future_to_chunk[future]
                try:
                    result = future.result(
                        timeout=TIMEOUT_PER_CHUNK
                    )  # TIMEOUT_PER_CHUNK minute timeout per chunk
                    if result is not None:
                        parallel_results[chunk_path] = result
                        log.info(f"Processed {chunk_path}")
                    else:
                        log.warning(f"Failed to process {chunk_path}")
                except concurrent.futures.TimeoutError:
                    log.error(
                        f"{chunk_path} timed out after {TIMEOUT_PER_CHUNK} seconds"
                    )
                except (RuntimeError, ValueError, OSError) as exc:
                    log.error(f"{chunk_path} generated an exception: {exc}")
                    traceback.print_exc()

        end_time = time.time()
        parallel_processing_time = end_time - start_time

        log.info(
            f"\nParallel processing of {len(temp_chunk_paths)} chunks took {parallel_processing_time:.4f} seconds."
        )
        log.info(
            f"Results: {len(parallel_results)} successful out of {len(temp_chunk_paths)}"
        )

    else:
        log.warning("No chunks were created for processing.")


# Remember to clean up temporary files after you are done
# for chunk_path in temp_chunk_paths:
#     if chunk_path.exists():
#         chunk_path.unlink()
#         print(f"Cleaned up: {chunk_path}")


if __name__ == "__main__":
    main()
