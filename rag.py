from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
import os
import json
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
url = "http://localhost:6333"

try:
    client = QdrantClient(url=url, prefer_grpc=False)
    db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")
    retriever = db.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})
    print("✅ Connected to Qdrant oncology database")

except Exception as e:
    print(f"❌ Error connecting to Qdrant: {e}")
    # Fallback to dummy retriever for testing
    from langchain.schema import Document
    from langchain.retrievers import BM25Retriever

    dummy_docs = [Document(
        page_content="Metastatic disease involves cancer cells spreading from primary tumors to distant organs. Common sites include bones, lungs, liver, and brain. Treatment focuses on systemic therapies like chemotherapy and targeted agents.",
        metadata={"source": "medical_oncology_handbook.pdf", "page": 156}
    )]
    retriever = BM25Retriever.from_documents(dummy_docs)
    retriever.k = 2
    print("⚠️  Using dummy retriever for testing")

# Initialize a real LLM (if available) or use the pattern-matching approach
try:
    # Try to load a local LLM
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        config={'max_new_tokens': 512, 'temperature': 0.01}
    )
    print("✅ Using local LLM for medical responses")
except Exception as e:
    print(f"❌ Could not load local LLM: {e}")


    # Fallback to pattern-based responses
    class OncologyAwareLLM:
        def invoke(self, prompt):
            try:
                # Extract context and question from the prompt
                context_match = re.search(r'MEDICAL CONTEXT:\s*(.*?)(?=QUESTION:\s*|$)', prompt,
                                          re.DOTALL | re.IGNORECASE)
                question_match = re.search(r'QUESTION:\s*(.*?)(?=ANSWER:\s*|$)', prompt, re.DOTALL | re.IGNORECASE)

                context = context_match.group(1).strip() if context_match else ""
                question = question_match.group(1).strip() if question_match else ""

                # Clean up context
                context = re.sub(r'Provide a concise, accurate medical answer based only on the context above\.', '',
                                 context)
                context = context.strip()

                if not context or len(context) < 100:
                    return "I don't have enough specific information in my oncology knowledge base to answer this question. Please try asking about general oncology concepts, cancer treatments, or metastatic disease."

                # Return the most relevant context as answer for now
                return f"Based on the medical literature: {context[:500]}..." if len(context) > 500 else context

            except Exception as e:
                print(f"Error in LLM invoke: {e}")
                return "I encountered an error while processing your oncology question. Please try again with a different query."


    llm = OncologyAwareLLM()
    print("✅ Using pattern-based medical responses")

prompt = PromptTemplate(
    template="""MEDICAL CONTEXT:
{context}

QUESTION:
{question}

Provide a concise, accurate medical answer based only on the context above.
ANSWER:""",
    input_variables=['context', 'question']
)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(query: str = Form(...)):
    try:
        # Retrieve relevant documents
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            return JSONResponse(content={
                "answer": "I couldn't find specific information about this oncology topic in my knowledge base. Please try asking about cancer treatment, metastatic disease, or other general oncology concepts.",
                "sources": []
            })

        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        context = context.strip()

        # Create the prompt with context
        formatted_prompt = prompt.format(context=context, question=query)

        # Get response from LLM
        answer = llm.invoke(formatted_prompt)
        answer = answer.strip()

        # Prepare source documents information
        sources = []
        for doc in relevant_docs:
            source_name = doc.metadata.get('source', 'Oncology Document')
            if '/' in source_name:
                source_name = source_name.split('/')[-1]
            if '\\' in source_name:
                source_name = source_name.split('\\')[-1]
            if source_name.endswith('.pdf'):
                source_name = source_name[:-4]

            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": source_name,
                "page": doc.metadata.get('page', 'N/A')
            })

        response_data = {
            "answer": answer,
            "sources": sources
        }

        print(f"Oncology Query: {query}")
        print(f"Answer: {answer[:100]}...")

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={
            "answer": "I apologize, but I encountered a technical issue while processing your oncology question. Please try again.",
            "sources": []
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
