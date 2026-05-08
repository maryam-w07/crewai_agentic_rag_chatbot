import pdfplumber #Python library that opens PDF files and extracts text from them page by page
from docx import Document as DocxDocument
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer #library that lets you load and run pre-trained embedding models locally
from qdrant_client import QdrantClient #main connection obj to the qdrant engine
from qdrant_client.models import Distance, VectorParams, PointStruct


#distance:cosine similarity
#VectorParams: defines the collection configuration,vector size (768) and distance metric
#PointStruct — represents a single record being stored, containing the id, the vector, and the payload (chunk text, source, page number)

SOURCES = [
    "/home/maryam-waqar/crewai_nutrition_assist/A-guide-to-healthy-eating-for-older-adults-August-2015.pdf",
    "/home/maryam-waqar/Downloads/EatingWellANutritionResourceforOlderPeople-1 (Copy).pdf"
    #add multi-source file paths here
]

COLLECTION_NAME = "nutrition_knowledge"
QDRANT_PATH     = "./qdrant_storage"   #folder created automatically
CHUNK_SIZE      = 300                  #words per chunk
CHUNK_OVERLAP   = 50                   # words shared between chunks


#Load embedding model & Qdrant 
print("Loading embedding model...")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

qdrant = QdrantClient(path=QDRANT_PATH)


#Helper: split text into overlapping chunks
def chunk_text(text):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:       # skip near-empty chunks
            chunks.append(chunk)
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# Loaders: multi-source docs
def load_pdf(path):  #opens pdfs nd extracts page by page
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages): #loop thru every page
            text = page.extract_text()
            if text and text.strip():
                pages.append({"text": text, "source": Path(path).name, "page": i + 1})
    return pages


def load_docx(path):
    doc  = DocxDocument(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "source": Path(path).name, "page": 1}]


def load_txt(path):
    text = Path(path).read_text(encoding="utf-8")
    return [{"text": text, "source": Path(path).name, "page": 1}]


def load_url(url):
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    soup     = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [{"text": text, "source": url, "page": 1}]


def load(source):
    if source.startswith("http"):
        return load_url(source)
    elif source.endswith(".pdf"):
        return load_pdf(source)
    elif source.endswith(".docx"):
        return load_docx(source)
    elif source.endswith(".txt"):
        return load_txt(source)
    else:
        print(f"Skipping unsupported file: {source}")
        return []


#Set up Qdrant collection
def setup_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


# Main 

def ingest():
    if not SOURCES:
        print("No sources found. Add file paths or URLs to the SOURCES list.")
        return

    setup_collection()

    # Start IDs after existing points so we never overwrite old data
    point_id = qdrant.get_collection(COLLECTION_NAME).points_count
    points   = []

    for source in SOURCES:
        print(f"\nProcessing: {source}")
        docs = load(source)

        for doc in docs:
            chunks = chunk_text(doc["text"])
            print(f"  {len(chunks)} chunks from page {doc['page']}")

            for chunk in chunks:
                embedding = embedder.encode(chunk, normalize_embeddings=True).tolist()
                points.append(PointStruct(
                    id      = point_id,
                    vector  = embedding,
                    payload = {
                        "text":   chunk,
                        "source": doc["source"],
                        "page":   doc["page"],
                    }
                ))
                point_id += 1

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"\nDone. {len(points)} chunks stored in Qdrant.")
    else:
        print("Nothing was indexed.")


if __name__ == "__main__":
    ingest()
