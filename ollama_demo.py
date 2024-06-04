from ollama import Client

llm = Client(host='http://localhost:11434', timeout=300)

payload = llm.generate(
    model='phi3',
    prompt="Why are frogs green?"
)

print(payload['response'])
