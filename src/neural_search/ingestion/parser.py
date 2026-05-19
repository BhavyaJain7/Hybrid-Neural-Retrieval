from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
import fitz 
from docx import Document


@dataclass
class ParsedPage:
    doc_id: str
    source_file: str
    page: int
    text: str
    metadata: dict = field(default_factory=dict)


def _parse_pdf(file_path: Path) -> list[ParsedPage]:
    pages = []
    try:
        doc = fitz.open(str(file_path))
        doc_id = file_path.stem
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                logger.debug(f"Skipping empty page {page_num} in {file_path.name}")
                continue
            pages.append(ParsedPage(
                doc_id=doc_id,
                source_file=file_path.name,
                page=page_num,
                text=text,
            ))
        doc.close()
        logger.info(f"Parsed PDF: {file_path.name} — {len(pages)} non-empty pages")
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_path.name}: {e}")
    return pages


def _parse_docx(file_path: Path) -> list[ParsedPage]:
    pages = []
    try:
        doc = Document(str(file_path))
        doc_id = file_path.stem
        current_section: list[str] = []
        section_index = 1

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Flush on headings to create logical sections
            if para.style.name.startswith("Heading") and current_section:
                pages.append(ParsedPage(
                    doc_id=doc_id,
                    source_file=file_path.name,
                    page=section_index,
                    text="\n".join(current_section),
                    metadata={"type": "section"},
                ))
                current_section = []
                section_index += 1
            current_section.append(text)

        # Flush remaining
        if current_section:
            pages.append(ParsedPage(
                doc_id=doc_id,
                source_file=file_path.name,
                page=section_index,
                text="\n".join(current_section),
                metadata={"type": "section"},
            ))

        logger.info(f"Parsed DOCX: {file_path.name} — {len(pages)} sections")
    except Exception as e:
        logger.error(f"Failed to parse DOCX {file_path.name}: {e}")
    return pages


def parse_document(file_path: Path) -> list[ParsedPage]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_path.name} — skipping")
        return []


def parse_directory(directory: Path) -> list[ParsedPage]:
    all_pages: list[ParsedPage] = []
    files = list(directory.glob("**/*.pdf")) + list(directory.glob("**/*.docx"))
    logger.info(f"Found {len(files)} documents in {directory}")
    for file_path in files:
        all_pages.extend(parse_document(file_path))
    logger.info(f"Total parsed pages/sections: {len(all_pages)}")
    return all_pages
