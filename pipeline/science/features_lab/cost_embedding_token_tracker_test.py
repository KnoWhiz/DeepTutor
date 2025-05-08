from langchain_openai import AzureOpenAIEmbeddings
import tiktoken

from dotenv import load_dotenv
import os
load_dotenv()

# Initialize the embedding model (using similar configuration from api_handler.py)
embedding_model = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large",
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS"),
    openai_api_key=os.getenv("OPENAI_API_KEY_EMBEDDINGS"),
    openai_api_type="azure",
    chunk_size=2000
)

# Alternative embedding models
lite_embedding_model = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS"),
    openai_api_key=os.getenv("OPENAI_API_KEY_EMBEDDINGS"),
    openai_api_type="azure",
    chunk_size=2000
)

small_embedding_model = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    model="text-embedding-3-small",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS"),
    openai_api_key=os.getenv("OPENAI_API_KEY_EMBEDDINGS"),
    openai_api_type="azure",
    chunk_size=2000
)

# Test text for embedding
test_texts = [
    "This is a short text to embed.",
    "Here's another example of text that we want to create embeddings for.",
    "Embedding models convert text into numerical vectors that capture semantic meaning."
]

# Function to manually calculate tokens using tiktoken
def count_tokens(text, model_name="text-embedding-3-large"):
    if "ada" in model_name:
        encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    else:
        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    
    return len(encoding.encode(text))

# Function to calculate embedding cost based on model
def calculate_embedding_cost(token_count, model_name="text-embedding-3-large"):
    # Pricing per 1K tokens as of 2024
    if model_name == "text-embedding-3-large":
        return (token_count / 1000) * 0.00013  # $0.00013 per 1K tokens
    elif model_name == "text-embedding-ada-002":
        return (token_count / 1000) * 0.0001  # $0.0001 per 1K tokens
    elif model_name == "text-embedding-3-small":
        return (token_count / 1000) * 0.00002  # $0.00002 per 1K tokens
    return 0

# Custom token and cost tracking for embeddings
print("Testing text-embedding-3-large model:")
# Embed multiple texts
embeddings = embedding_model.embed_documents(test_texts)

# Calculate tokens manually for each text
total_tokens = sum(count_tokens(text, "text-embedding-3-large") for text in test_texts)
total_cost = calculate_embedding_cost(total_tokens, "text-embedding-3-large")

# Print some information about the embeddings
print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# Print token usage and cost information
print("\nToken Usage Statistics (Manual Calculation):")
print(f"Total Tokens: {total_tokens}")
print(f"Total Cost (USD): ${total_cost:.10f}")
print(f"Cost per Token: ${(total_cost / total_tokens if total_tokens > 0 else 0):.10f}")

# Test with the lite embedding model
print("\nTesting text-embedding-ada-002 model:")
# Embed a single text
embedding = lite_embedding_model.embed_query(test_texts[0])

# Calculate tokens for single text
test_tokens = count_tokens(test_texts[0], "text-embedding-ada-002")
test_cost = calculate_embedding_cost(test_tokens, "text-embedding-ada-002")

print(f"Single embedding dimension: {len(embedding)}")

# Print token usage and cost information
print("\nToken Usage Statistics (Manual Calculation):")
print(f"Total Tokens: {test_tokens}")
print(f"Total Cost (USD): ${test_cost:.10f}")
print(f"Cost per Token: ${(test_cost / test_tokens if test_tokens > 0 else 0):.10f}")

# Test with the small embedding model
print("\nTesting text-embedding-3-small model:")
# Embed all texts
embeddings = small_embedding_model.embed_documents(test_texts)

# Calculate tokens manually for each text
total_tokens = sum(count_tokens(text, "text-embedding-3-small") for text in test_texts)
total_cost = calculate_embedding_cost(total_tokens, "text-embedding-3-small")

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# Print token usage and cost information
print("\nToken Usage Statistics (Manual Calculation):")
print(f"Total Tokens: {total_tokens}")
print(f"Total Cost (USD): ${total_cost:.10f}")
print(f"Cost per Token: ${(total_cost / total_tokens if total_tokens > 0 else 0):.10f}")

# Compare per-token costs between models
print("\nCost Comparison:")
print("Comparing embedding costs for the same content across models")
test_paragraph = " ".join(test_texts)

models = {
    "text-embedding-3-large": {"model": embedding_model, "name": "text-embedding-3-large"},
    "text-embedding-ada-002": {"model": lite_embedding_model, "name": "text-embedding-ada-002"},
    "text-embedding-3-small": {"model": small_embedding_model, "name": "text-embedding-3-small"}
}

print("\nModel Cost Comparison Table:")
print("-" * 80)
print(f"{'Model Name':<25} {'Token Count':<15} {'Total Cost (USD)':<20} {'Cost per 1K Tokens':<20}")
print("-" * 80)

for model_key, model_info in models.items():
    model = model_info["model"]
    model_name = model_info["name"]
    
    # Embed the text
    model.embed_query(test_paragraph)
    
    # Manual token counting and cost calculation
    token_count = count_tokens(test_paragraph, model_name)
    cost = calculate_embedding_cost(token_count, model_name)
    cost_per_k = cost / (token_count / 1000) if token_count > 0 else 0
    
    print(f"{model_name:<25} {token_count:<15} ${cost:.10f}       ${cost_per_k:.8f}")

print("-" * 80)
print("\nNote: Token counting and pricing are calculated manually using tiktoken")
print("and may not exactly match the actual API usage.")
