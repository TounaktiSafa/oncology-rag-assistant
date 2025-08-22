from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Initialize Qdrant client
url = "http://localhost:6333"

try:
    client = QdrantClient(url=url, prefer_grpc=False)
    db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")

    print("✅ Connected to Qdrant database")

    # Test specific medical queries that should be in oncology documents
    test_queries = [
        "metastatic disease",
        "cancer metastasis",
        "oncology treatment",
        "chemotherapy side effects",
        "tumor spread",
        "cancer cells",
        "medical oncology",
        "cancer diagnosis"
    ]

    print("\n" + "=" * 60)
    print("TESTING DOCUMENT RETRIEVAL FROM ONCOLOGY PDFs")
    print("=" * 60)

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            docs = db.similarity_search(query, k=3)
            print(f"   Found {len(docs)} documents")

            if len(docs) == 0:
                print("   ❌ No documents found for this query")
                continue

            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')

                # Clean source name
                if '/' in source:
                    source = source.split('/')[-1]
                if '\\' in source:
                    source = source.split('\\')[-1]

                print(f"   📄 Document {i + 1}: {source} (page {page})")
                print(f"   📝 Content: {doc.page_content[:150]}...")
                print(f"   📏 Length: {len(doc.page_content)} characters")
                print("   " + "-" * 50)

        except Exception as e:
            print(f"   ❌ Error searching for '{query}': {e}")

    # Check collection info
    print("\n" + "=" * 60)
    print("COLLECTION INFORMATION")
    print("=" * 60)

    collections = client.get_collections()
    for collection in collections.collections:
        if collection.name == "vector_database":
            print(f"Collection: {collection.name}")
            # Get more info about the collection
            collection_info = client.get_collection(collection_name="vector_database")
            print(f"Points count: {collection_info.points_count}")
            print(f"Vectors config: {collection_info.config.params.vectors}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure Qdrant is running on http://localhost:6333")