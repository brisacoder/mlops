"""PDF extraction utilities.

This module provides pure functions (no Prefect decorators) for:

* Loading a PDF with PyMuPDF
* Extracting per-page plain text
* Extracting embedded images (saved as PNG files)
* Performing a light heuristic section split
* Persisting extracted JSON artifacts (pages, sections, images) to a run directory

The heuristics here are intentionally lightweight and easily swappable.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF

SECTION_HEADING_RE = re.compile(r"^(?:\d+\.|[A-Z][A-Z0-9 ]{3,}) ?[\-:]? .{0,100}$")

@dataclass
class PageContent:
    """Container for raw page text.

    Attributes:
        page_number: 1-based page index within the PDF.
        text: Extracted plain text for the page.
    """
    page_number: int
    text: str

@dataclass
class Section:
    """Represents a heuristic logical section within the document.

    Attributes:
        id: Sequential section identifier starting at 0.
        title: Detected heading / first line considered a section title.
        start_page: 1-based page number where the section starts.
        end_page: 1-based page number where the section ends.
        text: Aggregated text belonging to the section (excluding title lines used to split).
    """
    id: int
    title: str
    start_page: int
    end_page: int
    text: str


def load_pdf(path: str | Path) -> fitz.Document:
    """Load a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        A PyMuPDF `Document` instance.
    """
    return fitz.open(str(path))


def extract_text_and_images(pdf_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """Extract pages, sections, and images from a PDF and persist artifacts.

    For each page we store raw text. Detected images are exported as PNGs. Sections
    are derived via `split_into_sections`.

    Args:
        pdf_path: Path to input PDF.
        output_dir: Directory where JSON artifacts and images will be written.

    Returns:
        Dictionary with keys:
            pages: List[dict] of per-page content
            sections: List[dict] of section records
            images: List[dict] describing extracted images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = load_pdf(pdf_path)
    pages: List[PageContent] = []
    images_meta: List[Dict[str, Any]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        pages.append(PageContent(page_number=page_index + 1, text=text))
        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:  # convert CMYK etc.
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_name = f"page{page_index+1}_img{img_index+1}.png"
            image_path = output_dir / image_name
            pix.save(image_path)
            images_meta.append({
                "page": page_index + 1,
                "image_file": image_name,
                "width": pix.width,
                "height": pix.height,
            })
            pix = None  # release

    sections = split_into_sections(pages)

    # Persist raw text & sections
    pages_json = [asdict(p) for p in pages]
    sections_json = [asdict(s) for s in sections]
    (output_dir / "pages.json").write_text(json.dumps(pages_json, indent=2), encoding="utf-8")
    (output_dir / "sections.json").write_text(json.dumps(sections_json, indent=2), encoding="utf-8")
    (output_dir / "images.json").write_text(json.dumps(images_meta, indent=2), encoding="utf-8")

    return {
        "pages": pages_json,
        "sections": sections_json,
        "images": images_meta,
    }


def split_into_sections(pages: List[PageContent]) -> List[Section]:
    """Split a sequence of pages into heuristic sections.

    Heuristics:
        * Lines that appear to be numbered headings (``1.``, ``1.2.``, etc.)
        * Lines with a high proportion of uppercase letters and limited length

    Args:
        pages: Ordered list of `PageContent` objects.

    Returns:
        A list of `Section` objects representing contiguous text blocks.
    """
    sections: List[Section] = []
    current_title = "Introduction"
    current_text_lines: List[str] = []
    current_start_page = pages[0].page_number if pages else 1
    section_id = 0

    def push_section(end_page: int):
        nonlocal section_id, current_title, current_text_lines, current_start_page
        if current_text_lines:
            sections.append(Section(
                id=section_id,
                title=current_title.strip()[:120],
                start_page=current_start_page,
                end_page=end_page,
                text="\n".join(current_text_lines).strip(),
            ))
            section_id += 1
            current_text_lines = []

    for page in pages:
        lines = page.text.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped and is_heading_candidate(stripped):
                # finish current
                push_section(end_page=page.page_number)
                current_title = stripped
                current_start_page = page.page_number
            else:
                current_text_lines.append(stripped)
    # final
    if pages:
        push_section(end_page=pages[-1].page_number)
    return sections


def is_heading_candidate(line: str) -> bool:
    """Return True if a line is considered a section heading.

    Criteria (fast heuristics):
        * Not excessively long (<= 140 chars)
        * Numbered outline style (e.g. ``1.``, ``2.3.``) OR
        * High uppercase ratio (> 0.6) with limited number of tokens (<= 12)

    Args:
        line: A stripped line candidate.

    Returns:
        True if it is treated as a heading, else False.
    """
    if len(line) > 140:
        return False
    if re.match(r"^(?:\d+\.)+\s+", line):
        return True
    uppercase_ratio = sum(1 for c in line if c.isupper()) / max(1, sum(1 for c in line if c.isalpha()))
    if uppercase_ratio > 0.6 and len(line.split()) <= 12:
        return True
    return False
