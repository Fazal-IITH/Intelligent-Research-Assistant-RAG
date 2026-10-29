import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
def make_chunks(content: bytes):
    doc = pymupdf.open(
        stream=content,
        filetype= "pdf"
    )
    full_text = ""
    for i in range(doc.page_count):
        page = doc.load_page(i)
        full_text += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_text(full_text)
    doc.close()
    return chunks