# Import necessary libraries
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model for encoding
encoder_model = SentenceTransformer('all-MiniLM-L6-v2')

# Example corpus
documents = [
    "The capital of France is Paris.",
    "The Great Wall of China is visible from space.",
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "Python is a popular programming language."
]

# Encode documents
document_embeddings = encoder_model.encode(documents, convert_to_tensor=True)

# Build the FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings.cpu().numpy())

# Load pre-trained model and tokenizer for generation
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate responses
def generate_response(prompt, context=None, max_length=50):
    if context:
        prompt = f"{context} {prompt}"
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = generator_model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to get relevant context from documents
def get_relevant_context(query, top_k=1):
    query_embedding = encoder_model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    
    # Ensure the query embedding is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# Function to clean repetitive responses
def clean_response(response):
    sentences = response.split('. ')
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    return '. '.join(unique_sentences)

# Chat function that integrates retrieval and generation
def chat(query):
    relevant_contexts = get_relevant_context(query)
    combined_context = " ".join(relevant_contexts)
    response = generate_response(query, context=combined_context)
    clean_res = clean_response(response)
    return clean_res

# Predefined questions
questions = [
    "Who painted the Mona Lisa?",
    "Which is a popular programming language?",
    "What is the capital of France?",
    "Is the Great Wall of China visible from space?"
]

# Display questions and take user input
print("Please select a question:")
for i, question in enumerate(questions):
    print(f"{i + 1}. {question}")

# Get user input
selected_index = int(input("Enter the number of your selected question: ")) - 1

# Ensure the selected index is valid
if selected_index < 0 or selected_index >= len(questions):
    print("Invalid selection. Please restart and select a valid question number.")
else:
    query = questions[selected_index]
    response = chat(query)
    print(f"Response: {response}")
