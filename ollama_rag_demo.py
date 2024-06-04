import ollama
import chromadb

with open('minimal_RAG.txt', 'r') as file:
    chunks = file.readlines()
    chunks = [chunk.strip() for chunk in chunks]

print(chunks)

client = chromadb.Client()
collection = client.create_collection(name="docs")

emb_model = "nomic-embed-text"

for i, d in enumerate(chunks):
    response = ollama.embeddings(model=emb_model, prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

prompt = "What animals are Cortheens related to?"

prompt_embedding = ollama.embeddings(prompt=prompt, model=emb_model)

similarity_search_results = collection.query(
  query_embeddings=[prompt_embedding["embedding"]],
  n_results=1
)

prompt_context = similarity_search_results['documents'][0][0]

print(prompt_context)

output = ollama.generate(
  model="phi3",
  prompt=f"Using this data: {prompt_context}. Respond to this prompt: {prompt}"
)

print("Response:")
print(output['response'])
