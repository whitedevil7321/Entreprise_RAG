
import logfire
from pypdf import PdfReader


def parse_pdf(file_name: str) -> str:
    """
    Parse PDF content and return all extracted text as a single string.
    Uses pdfplumber as a fallback for pages where pypdf extracts no text.
    """

    with logfire.span("PDF Parsing", filename=file_name):
        try:
            reader = PdfReader(file_name)
            total_pages = len(reader.pages)

            logfire.info(
                f"Total pages in PDF file {file_name}: {total_pages}"
            )

            text_parts: list[str] = []
            blank_pages: list[int] = []

            # Primary extraction using pypdf
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                if page_text.strip():
                    text_parts.append(page_text)
                else:
                    blank_pages.append(i + 1)

            if blank_pages:
                logfire.info(
                    f"Pages needing fallback extraction in "
                    f"{file_name}: {blank_pages}"
                )

                # Fallback extraction using pdfplumber
                try:
                    import pdfplumber

                    with pdfplumber.open(file_name) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text() or ""

                            if fallback_text.strip():
                                text_parts.append(fallback_text)

                                logfire.info(
                                    f"Extracted text from page {page_num} "
                                    f"using pdfplumber."
                                )
                            else:
                                logfire.warning(
                                    f"Page {page_num} remains blank "
                                    f"after fallback extraction."
                                )

                except Exception as exc:
                    logfire.warning(
                        f"pdfplumber fallback failed for "
                        f"{file_name}: {exc}"
                    )

            # Combine all extracted page text
            full_text = "\n".join(text_parts).strip()

            if not full_text:
                logfire.warning(
                    f"No text found in {file_name}. "
                    "The PDF may be image-based, scanned, or corrupted."
                )
            else:
                logfire.info(
                    f"Successfully parsed PDF: "
                    f"{len(full_text)} characters extracted."
                )

            return full_text

        except Exception:
            logfire.exception(f"Error parsing PDF file: {file_name}")
            raise
