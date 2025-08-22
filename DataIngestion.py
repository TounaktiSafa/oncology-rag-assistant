from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 1️⃣ Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
print("✅ Embeddings model loaded")

# 2️⃣ Load PDFs - check if Data folder exists
data_path = 'Data/'
if not os.path.exists(data_path):
    print(f"❌ Data folder not found at: {data_path}")
    print("Please make sure the Data folder exists with your PDF files")
    exit()

print(f"📁 Data folder contents: {os.listdir(data_path)}")

loader = DirectoryLoader(
    data_path,
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} document pages")

if len(documents) == 0:
    print("❌ No documents were loaded. Check:")
    print("   - PDF files are in Data/ folder")
    print("   - PDF files are not corrupted")
    print("   - You have read permissions")
    exit()

# 3️⃣ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]  # Better splitting for medical text
)
texts = text_splitter.split_documents(documents)
print(f"✅ Split documents into {len(texts)} chunks")

# Show sample of chunks
print("\n📋 Sample chunks:")
for i, text in enumerate(texts[:3]):  # Show first 3 chunks
    source = text.metadata.get('source', 'Unknown')
    if '/' in source:
        source = source.split('/')[-1]
    if '\\' in source:
        source = source.split('\\')[-1]
    print(f"   {i + 1}. {source} (page {text.metadata.get('page', 'N/A')})")
    print(f"      Content: {text.page_content[:100]}...")
    print()

# 4️⃣ Create vector database in Qdrant
try:
    print("🚀 Creating vector database in Qdrant...")
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        url="http://localhost:6333",
        collection_name="vector_database",
        prefer_grpc=False
    )
    print("✅ Vector Database created successfully in Qdrant")

    # Verify the collection was created
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    collection_info = client.get_collection(collection_name="vector_database")
    print(f"📊 Collection stats: {collection_info.points_count} vectors stored")

except Exception as e:
    print(f"❌ Error creating vector database: {e}")
    print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")