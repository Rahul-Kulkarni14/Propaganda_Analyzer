from io import BytesIO
from pypdf import PdfReader


ALLOWED_DOCUMENT_EXTENSIONS = {"txt", "pdf"}


def get_file_extension(filename):
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def is_allowed_document(filename):
    extension = get_file_extension(filename)
    return extension in ALLOWED_DOCUMENT_EXTENSIONS


def extract_text_from_txt(file_storage):
    raw_bytes = file_storage.read()

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode("utf-8", errors="ignore")


def extract_text_from_pdf(file_storage):
    pdf_bytes = file_storage.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n\n".join(text_parts)


def extract_text_from_document(file_storage):
    filename = file_storage.filename or ""
    extension = get_file_extension(filename)

    if not is_allowed_document(filename):
        return {
            "text": "",
            "error": "Unsupported file type. Please upload a .txt or .pdf file."
        }

    try:
        if extension == "txt":
            extracted_text = extract_text_from_txt(file_storage)
        elif extension == "pdf":
            extracted_text = extract_text_from_pdf(file_storage)
        else:
            extracted_text = ""

        extracted_text = extracted_text.strip()

        if not extracted_text:
            return {
                "text": "",
                "error": "No readable text found. This may be a scanned or image-based document."
            }

        return {
            "text": extracted_text,
            "error": None
        }

    except Exception as error:
        return {
            "text": "",
            "error": f"Could not read this document: {error}"
        }
