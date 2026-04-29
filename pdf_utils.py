import io

from pypdf import PdfReader


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    if not file_bytes:
        return ""
    with io.BytesIO(file_bytes) as pdf_stream:
        reader = PdfReader(pdf_stream)
        text_blocks = []
        for page in reader.pages:
            text_blocks.append(page.extract_text() or "")
    return "\n".join(text_blocks).strip()


def extract_text_from_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_name = (uploaded_file.name or "").lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore").strip()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)

    raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")
