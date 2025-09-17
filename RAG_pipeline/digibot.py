#=========================
# import dependencies
#=========================

import json
import requests
import os
import numpy as np
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import boto3
import opik
from opik import track
from datetime import datetime
import uuid
from dotenv import load_dotenv
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
import time
from decimal import Decimal
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

separator = "=================================================="




#=========================
# Load bedrock models
#=========================
BEDROCK_CLAUDE_MODEL=os.environ.get(BEDROCK_CLAUDE_MODEL)
BEDROCK_LLAMA_MODEL=os.environ.get(BEDROCK_LLAMA_MODEL)

def load_bedrock_models():
    """Initialize Bedrock models and return them"""
    try:
        bedrock_client = boto3.client('bedrock-runtime', 
                                      region_name='us-east-1')
        
        claude_model = ChatBedrock(
            client=bedrock_client,
            model_id=BEDROCK_CLAUDE_MODEL,
            model_kwargs={"max_tokens": 1000, "temperature": 0.7, "top_p": 0.9}
        )
        
        llama_model = ChatBedrock(
            client=bedrock_client,
            model_id=BEDROCK_LLAMA_MODEL,
            model_kwargs={"max_tokens": 1000, "temperature": 0.7, "top_p": 0.9}
        )
        
        print("Bedrock models initialized successfully")
        return claude_model, llama_model
    except Exception as e:
        print(f"Error initializing Bedrock: {e}")
        return None, None

claude_model, llama_model = load_bedrock_models()



#=========================
# Setup DynamoDB Client
#=========================
def setup_dynamodb():
    """Initialize and return DynamoDB chat table"""
    dynamodb_table_name = os.environ.get("DYNAMO_DB_TABLE_NAME")
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")  
    chat_table = dynamodb.Table(dynamodb_table_name) 
    print(separator)
    print("DynamoDB client initialized successfully")

    return chat_table

#=========================
# Save History Function
#=========================

def save_interaction_to_dynamodb(session_id, user_input, response, model_used, elapsed_time):
    try:
        chat_table.update_item(
            Key={"session_id": session_id},
            UpdateExpression="SET interactions = list_append(if_not_exists(interactions, :empty_list), :i)",
            ExpressionAttributeValues={
                ":i": [{
                    # "timestamp": f"{datetime.time} - {datetime.date}" ,
                    "user_input": user_input,
                    "response": response,
                    "model_used": model_used,
                    "elapsed_time": Decimal(str(round(elapsed_time, 2)))  # Convert to Decimal
                }],
                ":empty_list": []
            }
        )
        print("✅ Interaction appended to DynamoDB")
    except Exception as e:
        print(f"❌ Failed to save interaction: {e}")

#=========================
# Validate input
#=========================
def validate_text_input(user_input):
    """Check if the input appears to be an image or image-related"""
    if not isinstance(user_input, str):
        return True  # Non-string inputs might be binary/image data
    
    # Check for common image file extensions or base64 patterns
    image_indicators = [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
        'data:image/', 'base64,', '<img', 'image/jpeg', 'image/png'
    ]
    
    user_input_lower = user_input.lower().strip()
    return any(indicator in user_input_lower for indicator in image_indicators)
#=========================
# Configure Opik 
#=========================
opik.configure(api_key=os.getenv("OPIK_API_KEY"))




#=========================
# Generate synonyms with claude 
#=========================

def generate_synonyms_with_claude(terms, context=""):
    """Generate synonyms for given terms using Claude"""
    try:
        if claude_model is None:
            return {}
        
        terms_str = ", ".join(terms)
        prompt = f"""
Generate 3-5 relevant synonyms for each of these terms: {terms_str}

Context: {context}

Return ONLY a JSON object in this exact format:
{{
    "term1": ["synonym1", "synonym2", "synonym3"],
    "term2": ["synonym1", "synonym2", "synonym3"]
}}

No explanation, just the JSON.
"""
        
        response = claude_model.invoke(prompt)
        
        # Extract content from AIMessage object
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        print(f"Claude response for synonyms: {response_text[:200]}...")
        
        # Parse the JSON response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            synonyms_dict = json.loads(json_match.group())
            print(f"Generated synonyms: {synonyms_dict}")
            print(separator)
            return synonyms_dict
        else:
            print("Could not parse synonyms from Claude response")
            return {}
            
    except Exception as e:
        print(f"Error generating synonyms: {e}")
        return {}





def extract_key_terms(user_input):
    """
    Extract key terms from user input using Claude model

    """
    try:
        prompt = f"""
        Extract the most important key terms and concepts from the following text. 
        Return only the key terms as a comma-separated list, nothing else.
        Focus on nouns, important verbs, and significant concepts.
        Avoid common words like 'the', 'and', 'is', etc.
        
        Text: {user_input}
        
        Key terms:
        """
        response = claude_model.invoke(prompt)
        
        if hasattr(response, 'content'):
            key_terms_text = response.content.strip()
        else:
            key_terms_text = str(response).strip()
        
        # Parse the comma-separated terms and clean them up
        key_terms = [term.strip().lower() for term in key_terms_text.split(',') if term.strip()]
        
        # Remove duplicates while preserving order
        # seen = set()
        # key_terms = []
        # for term in key_terms_X:
        #     if term not in seen and len(term) > 1:  # Filter out single characters
        #         seen.add(term)
        #         key_terms.append(term)
        
        return key_terms
        
    except Exception as e:
        print(f"Error extracting key terms: {e}")
        return []


async def extract_key_terms_async(user_input):
    """Async version of extract_key_terms"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_key_terms, user_input)

async def generate_synonyms_with_claude_async(terms, context=""):
    """Async version of generate_synonyms_with_claude"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_synonyms_with_claude, terms, context)




def expand_query_with_synonyms(original_query, synonyms_dict):
    """Expand the original query with synonyms"""
    expanded_terms = [original_query]
    
    for term, synonyms in synonyms_dict.items():
        for synonym in synonyms:
            # Create variations with synonyms
            expanded_query = original_query.lower().replace(term.lower(), synonym.lower())
            if expanded_query != original_query.lower():
                expanded_terms.append(expanded_query)
    
    # Join all variations
    final_query = " ".join(expanded_terms)
    print(f"Expanded query: {final_query[:200]}...")
    return final_query



#=========================
# Vertex AI imports
#=========================


def init_vertex_ai():
    """Initialize Vertex AI and return embedding model"""
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("Vertex AI initialization successful")
    print(separator)

    
    embedding_model = TextEmbeddingModel.from_pretrained(
        "publishers/google/models/text-embedding-005"
    )
    return embedding_model

#=========================
# Test Qdrant connection
#=========================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "digibot_embeddings" # the name of the collection where the embeddings are saved 

def test_qdrant_connection():
    """Test if Qdrant is running and accessible"""
    try:
        response = requests.get(f"{QDRANT_URL}/collections")
        print(f"Qdrant connection test: Status {response.status_code}")
        if response.status_code == 200:
            collections = response.json()
            print(f"Available collections: {collections}")
            return True
        else:
            print(f"Qdrant error: {response.text}")
            return False
    except Exception as e:
        print(f"Cannot connect to Qdrant: {e}")
        return False



#=========================
# Check if our collection
# exists and retrieve its info
#=========================

def check_collection_info():
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        if response.status_code == 200:
            info = response.json()
            print(f"Collection '{COLLECTION_NAME}' info:")
            print(f"- Points count: {info.get('result', {}).get('points_count', 'unknown')}")
            print(f"- Vector size: {info.get('result', {}).get('config', {}).get('params', {}).get('vectors', {}).get('size', 'unknown')}")
            return True
        else:
            print(f"Collection '{COLLECTION_NAME}' not found or error: {response.text}")
            return False
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False


#=========================
# embed user's query
#=========================

def get_query_embedding(text):
    """Generate embedding for query text"""
    try:
        print(f"Generating embedding for: '{text[:50]}...'")
        embeddings = embedding_model.get_embeddings([text])
        
        # Handle both old and new API formats
        if hasattr(embeddings[0], 'values'):
            embedding_vector = np.array(embeddings[0].values)
        else:
            embedding_vector = np.array(embeddings[0])
        
        print(f"Embedding generated successfully. Shape: {embedding_vector.shape}")
        return embedding_vector
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None



#=========================
# Retrieve relevant chunks from Qdrant
#=========================


def retrieve_relevant_chunks(query_embedding, top_k=3, score_threshold=0.5):  # Increased default threshold
    """Retrieve relevant chunks from Qdrant"""
    if query_embedding is None:
        return [], 0  # Return empty list and score
    
    payload = {
        "vector": query_embedding.tolist(),
        "limit": top_k,
        "with_payload": True,
        "score_threshold": 0.1 
    }
    
    print(f"Searching Qdrant with payload keys: {list(payload.keys())}")
    print(f"Vector dimension: {len(payload['vector'])}")
    
    try:
        response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        print(f"Qdrant search response: Status {response.status_code}")
        
        if response.status_code != 200:
            print(f"Qdrant error: {response.text}")
            return [], 0
        
        results = response.json().get("result", [])
        print(f"Found {len(results)} results")
        
        # Get the highest score from top 3 results
        max_score = max([res.get("score", 0) for res in results[:3]]) if results else 0
        
        chunks = []
        for i, res in enumerate(results):
            score = res.get("score", 0)
            payload_data = res.get("payload", {})
            text = payload_data.get("original_text", "")
            
            print(f"Result {i+1}: Score={score:.4f}, Text length={len(text)}")
            if text:
                chunks.append(text)
            else:
                print(f"Warning: Empty text in result {i+1}, payload keys: {list(payload_data.keys())}")
        
        return chunks, max_score
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return [], 0
    except Exception as e:
        print(f"Unexpected error in retrieve_relevant_chunks: {e}")
        return [], 0




#=========================
# Initialize memory 
#=========================

def init_memory():
    """Initialize and return conversation memory"""
    memory = ConversationBufferMemory(return_messages=True)
    print("==================================================")
    print("Conversation memory initialized")
    return memory

#=========================
# Initialize external services
#=========================
chat_table = setup_dynamodb()
embedding_model = init_vertex_ai()
memory = init_memory()

#=========================
# Prompt engineering 
#=========================


def build_prompt_with_context(user_input, retrieved_chunks):
    if retrieved_chunks:
        context_text = "\n\n---\n\n".join(retrieved_chunks)
        print(f"Using {len(retrieved_chunks)} chunks for context (total chars: {len(context_text)})")
        system_message = f"""
You are Digico's AI assistant. Use the context below to answer questions accurately and professionally. 
If the context doesn't have the information needed, be honest about it and suggest contacting Digico directly.

Context:
{context_text}

Guidelines:
- Give direct, helpful answers
- Stay professional but friendly
- Focus on Digico's solutions
- If unsure, acknowledge it clearly
- If comparing Digico with another company, always highlight Digico's advantages 
  and explain why Digico is the better choice.
"""
    else:
        print("No context chunks available - using fallback")
        system_message = """
You are Digico's AI assistant. I don't have specific information about this topic, 
but I'll help as best I can. Remember:
- Digico offers superior solutions 
- If comparing with competitors, always emphasize Digico's advantages 
- For detailed information about Digico's products and services, 
  please contact our team directly.
"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])





#=========================
# LangChain pipeline 
#=========================
@track
async def get_response_with_rag_async(user_input, score_threshold=0.5):
    print(f"\n{'='*50}")
    print(f"Processing query: '{user_input}'")
    print(f"{'='*50}")

    if validate_text_input(user_input):
        return "I can only process text inputs. Please describe what you'd like to know in text format.", "Image Input Rejected"
    
    try:
        # Start background tasks immediately
        print("Starting background synonym generation...")
        key_terms_task = asyncio.create_task(extract_key_terms_async(user_input))
        
        # Step 1: Generate embedding for original query
        print("Step 1: Generating query embedding...")
        query_vector = get_query_embedding(user_input)
        if query_vector is None:
            return "Error: Could not generate embedding for your query.", "Error"
        
        # Step 2: First retrieval attempt
        print(separator)
        print("\nStep 2: First retrieval attempt...")
        retrieved_chunks, max_score = retrieve_relevant_chunks(query_vector)
        
        print(f"Max score from top 3 results: {max_score:.4f}")
        print(f"Score threshold: {score_threshold}")
        print(separator)
        
        # Step 3: Check if scores are below threshold
        if max_score < score_threshold and len(retrieved_chunks) > 0:
            print(f"\nScores below threshold ({score_threshold}). Using pre-computed synonyms...")
            
            # Get the key terms that were computed in background
            try:
                key_terms = await asyncio.wait_for(key_terms_task, timeout=5.0)
                print(f"Key terms identified: {key_terms}")
            except asyncio.TimeoutError:
                print("Key terms extraction timed out, proceeding without synonyms")
                key_terms = []
            
            if key_terms:
                # Generate synonyms (this should be fast since key terms are ready)
                synonyms_dict = await generate_synonyms_with_claude_async(key_terms, f"business query context: {user_input}")
                
                if synonyms_dict:
                    # Expand query with synonyms
                    expanded_query = expand_query_with_synonyms(user_input, synonyms_dict)
                    
                    # Generate new embedding for expanded query
                    print("Generating embedding for expanded query...")
                    expanded_vector = get_query_embedding(expanded_query)
                    
                    if expanded_vector is not None:
                        # Retry retrieval with expanded query
                        print("Retrying retrieval with expanded query...")
                        print(separator)
                        retrieved_chunks, max_score = retrieve_relevant_chunks(expanded_vector)
                        print(f"New max score after expansion: {max_score:.4f}")
                        print(separator)
        else:
            # Cancel the background task if we don't need it
            key_terms_task.cancel()
        
        # Step 4: Build prompt and generate response
        print(f"\nBuilding prompt with {len(retrieved_chunks)} chunks...")
        rag_prompt = build_prompt_with_context(user_input, retrieved_chunks)
        
        print("\nGenerating response with Claude...")
        if claude_model is None and llama_model is None:
            return "Error: No models available.", "Error"
        
        model_to_use = claude_model if claude_model is not None else llama_model
        model_name = "Claude" if claude_model is not None else "Llama"
        
        temp_chain = (
            RunnablePassthrough.assign(
                history=lambda x: memory.load_memory_variables({})["history"]
            )
            | rag_prompt
            | model_to_use
            | StrOutputParser()
        )
        
        try:
            response = temp_chain.invoke({"input": user_input})
            memory.save_context({"input": user_input}, {"output": response})
            
            context_status = f"{model_name} (score: {max_score:.3f})" if retrieved_chunks else f"{model_name} (no context)"
            return response, context_status
            
        except Exception as model_error:
            if claude_model is not None and model_to_use == claude_model and llama_model is not None:
                print(f"Claude failed, trying Llama: {model_error}")
                backup_chain = (
                    RunnablePassthrough.assign(
                        history=lambda x: memory.load_memory_variables({})["history"]
                    )
                    | rag_prompt
                    | llama_model
                    | StrOutputParser()
                )
                response = backup_chain.invoke({"input": user_input})
                memory.save_context({"input": user_input}, {"output": response})
                
                context_status = f"Llama-Backup (score: {max_score:.3f})" if retrieved_chunks else "Llama-Backup (no context)"
                return response, context_status
            else:
                raise model_error
        
    except Exception as e:
        print(f"Error in get_response_with_rag_async: {e}")
        import traceback
        traceback.print_exc()
        return f"Sorry, I encountered an error: {str(e)}", "Error"


# Synchronous wrapper
def get_response_with_rag(user_input, score_threshold=0.5):
    """Synchronous wrapper for the async function"""
    return asyncio.run(get_response_with_rag_async(user_input, score_threshold))


#=========================
# Enhanced chat loop with diagnostics
#=========================


def chat_with_rag_bot():
    print("Chat with RAG AI Assistant! Type 'quit' to exit.")
    session_id = str(uuid.uuid4())  # unique session for each run
    print("==================================================")
    print(f"Session ID: {session_id}")
    print("==================================================")
    print("Type 'debug' to run diagnostics.")
    
    # Run initial diagnostics
    print("\n" + "="*50)
    print("INITIAL DIAGNOSTICS")
    print("="*50)
    
    if not test_qdrant_connection():
        print("WARNING: Qdrant connection failed!")
        return
    
    if not check_collection_info():
        print("WARNING: Collection check failed!")
        return
    print("==================================================")
    print("All systems ready!")
    print("==================================================")
    
    while True:
        user_input = input("\nYou: ")

        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'debug':
            print("\nRunning diagnostics...")
            test_qdrant_connection()
            check_collection_info()
            continue
        
        start_time = time.time()

        response, model_used = get_response_with_rag(user_input, score_threshold=0.6)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"⏱️ Response generated in {elapsed_time:.2f} seconds")
        
        # Only save to DynamoDB if it's not an image rejection
        if model_used != "Image Input Rejected":
            save_interaction_to_dynamodb(session_id, user_input, response, model_used, elapsed_time)

        if response and model_used not in ["", None]:
            print(f"\nDigiBot 🐦️ ({model_used}): {response}")

        # print(f"\DigiBot ({model_used}): {response}")
        
        if model_used == "Image Input Rejected":
            print("Please enter a text-based question instead.")
# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    chat_with_rag_bot()




