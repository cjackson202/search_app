import streamlit as st  
import os  
import io  
import pdfplumber  
from azure.storage.blob import BlobServiceClient  
from azure.core.credentials import AzureKeyCredential  
from azure.identity import DefaultAzureCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.indexes.models import (  
    SimpleField,  
    SearchFieldDataType,  
    VectorSearch,  
    SearchIndex,  
    SearchableField,  
    SearchField,  
    VectorSearchProfile,  
    HnswAlgorithmConfiguration  
)  
from dotenv import load_dotenv  
from openai import AzureOpenAI 
import requests 
from datetime import datetime, timezone 
import tiktoken  
import re  
import json
from styling import global_page_style2
  
# Load environment variables  
load_dotenv()  
  
# Configure Azure AI Search parameters  
search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')  
search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')  

# Configure Azure OpenAI parameters  
azure_endpoint = os.getenv('AZURE_OPENAI_BASE')  
azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')  
azure_openai_api_version = os.getenv('AZURE_OPENAI_VERSION')  
azure_ada_deployment = os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT')  
azure_gpt_deployment = os.getenv('AZURE_GPT_DEPLOYMENT')  

def get_user_email():
    # Extract headers using Streamlit's context
    headers = st.context.headers
    # The header names may vary; check the specific headers set by Azure App Service
    user_email = headers.get('X-MS-CLIENT-PRINCIPAL-NAME')
    user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID')
    return user_email, user_id
  
def setup_azure_openai():  
    """Sets up Azure OpenAI."""  
    status_placeholder = st.empty()  
    status_placeholder.write("Setting up Azure OpenAI...")  
    azure_openai = AzureOpenAI(  
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version=os.getenv('AZURE_OPENAI_VERSION'),  
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')  
    )  
    status_placeholder.write("Azure OpenAI setup complete.")  
    return azure_openai  
  
def connect_to_blob_storage():  
    """Connects to Azure Blob Storage."""  
    status_placeholder = st.empty()  
    status_placeholder.write("Connecting to Blob Storage...")  
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("BLOB_CONNECTION_STRING"))  
    container_client = blob_service_client.get_container_client(os.getenv("BLOB_CONTAINER_NAME"))  
    status_placeholder.write("Connected to Blob Storage.")  
    return container_client  
  
def load_blob_content(blob_client):  
    """Loads and returns the content of the PDF or TXT blob."""  
    blob_name = blob_client.blob_name  
    if not (blob_name.lower().endswith('.pdf') or blob_name.lower().endswith('.txt')):  
        raise ValueError(f"Blob {blob_name} is not a PDF or TXT file.")  
  
    blob_data = blob_client.download_blob().readall()  
      
    if blob_name.lower().endswith('.pdf'):  
        pdf_stream = io.BytesIO(blob_data)  
        pages = ""  
  
        with pdfplumber.open(pdf_stream) as pdf:  
            for page_num, page in enumerate(pdf.pages, start=1):  
                text = page.extract_text()  
                if text:  
                    text += f"\n\nPg. {page_num}\n\n--page break--\n\n"
                    pages += text
        # st.write(pages)
        # time.sleep(500)
        return pages  
      
    elif blob_name.lower().endswith('.txt'):  
        text = blob_data.decode('utf-8')  
        return text   
  
def get_access_level(blob_name):  
    """Returns the access level for a given blob name. """  
    # Define the mapping of blob names to access levels  
    blob_access_levels = {  
        'extracted_text.txt': [os.getenv("my_entra_id")],  
    }  
    return blob_access_levels.get(blob_name, ['all'])  
  
def split_text_with_metadata(text, metadata, max_length=800, overlap=75, encoding_name='cl100k_base'):  
    """Splits the text into chunks with metadata."""  
    tokenizer = tiktoken.get_encoding(encoding_name)  
    tokens = tokenizer.encode(text)  
    chunks = []  
    start = 0  
    end = max_length  
  
    while start < len(tokens):  
        chunk = tokens[start:end]  
        chunk_text = tokenizer.decode(chunk)  
        chunk_metadata = metadata.copy()  
        chunk_metadata.update({  
            'start_token': start,  
            'end_token': end,  
            'chunk_length': len(chunk),  
            'chunk_text_preview': chunk_text[:50] + '...',  
            'access_level': get_access_level(metadata['blob_name'])  
        })  
        chunks.append({  
            'text': chunk_text,  
            'metadata': chunk_metadata  
        })  
        start = end - overlap  
        end = start + max_length  
    return chunks  
  
def vectorize():  
    """Main function that orchestrates the vector workflow."""  
    status_placeholder = st.empty()  
    azure_openai = setup_azure_openai()  
    container_client = connect_to_blob_storage()  
  
    # Read and chunk documents with metadata  
    status_placeholder.write("Listing blobs in container...")  
    blob_list = container_client.list_blobs()  
    documents = []  
    for blob in blob_list:  
        if not (blob.name.lower().endswith('.pdf') or blob.name.lower().endswith('.txt')):  
            status_placeholder.write(f"Skipping non-PDF blob: {blob.name}")  
            continue  
        status_placeholder.write(f"Processing blob: {blob.name}")  
        blob_client = container_client.get_blob_client(blob)  
        try:  
            document = load_blob_content(blob_client)  
            document_link = f'https://{os.getenv("BLOB_ACCOUNT_NAME")}.blob.core.windows.net/{os.getenv("BLOB_CONTAINER_NAME")}/{blob.name}'  
            metadata = {"blob_name": blob.name, "document_link": document_link}  
            chunks = split_text_with_metadata(document, metadata)  
            documents.extend(chunks)  
        except Exception as e:  
            status_placeholder.write(f"Failed to process blob {blob.name}: {e}")  
  
    status_placeholder.write("Blobs processed and documents chunked.")  
  
    # Generate embeddings  
    status_placeholder.write("Generating embeddings...")  
    embeddings = []  
    tokenizer = tiktoken.get_encoding("cl100k_base")  
    max_tokens = 8192  
    for i, doc in enumerate(documents):  
        status_placeholder.write(f"Processing chunk {i + 1}/{len(documents)}")  
        status_placeholder.write(f"Chunk text: {doc['text']}\n")  
        tokens = tokenizer.encode(doc["text"])  
        if len(tokens) > max_tokens:  
            status_placeholder.write(f"Skipping document chunk {i + 1} with {len(tokens)} tokens, exceeding max limit of {max_tokens}.")  
            continue  
        response = azure_openai.embeddings.create(input=doc["text"], model=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"))  
        embeddings.append({  
            "embedding": response.data[0].embedding,  
            "metadata": doc["metadata"]  
        })  
        status_placeholder.write(f"Embeddings: {response.data[0].embedding}")  
  
    status_placeholder.write("Embeddings generation complete.")  
  
    # Create Search Index  
    status_placeholder.write("Creating search index...")  
    credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))  
    search_index_client = SearchIndexClient(endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"), credential=credential)  
    fields = [  
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),  
        SearchableField(name="content", type=SearchFieldDataType.String),  
        SearchableField(name="blob_name", type=SearchFieldDataType.String),  
        SearchableField(name="document_link", type=SearchFieldDataType.String),  
        SearchField(  
            name="embedding",  
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  
            searchable=True,  
            vector_search_dimensions=1536,  
            vector_search_profile_name="myHnswProfile"  
        ),  
        SearchField(name="access_level", type=SearchFieldDataType.Collection(SearchFieldDataType.String))  
    ]  
    vector_search = VectorSearch(  
        algorithms=[  
            HnswAlgorithmConfiguration(name="myHnsw")  
        ],  
        profiles=[  
            VectorSearchProfile(  
                name="myHnswProfile",  
                algorithm_configuration_name="myHnsw"  
            )  
        ]  
    )  
    index = SearchIndex(name="documents-index", fields=fields, vector_search=vector_search)  
    search_index_client.create_index(index)  
    status_placeholder.write("Search index created.")  
  
    # Upload chunks and embeddings to Azure AI Search  
    status_placeholder.write("Uploading documents to search index...")  
    search_client = SearchClient(endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"), index_name="documents-index", credential=credential)  
    documents_to_upload = []  
  
    for i, doc in enumerate(embeddings):  
        documents_to_upload.append({  
            "id": str(i),  
            "content": documents[i]["text"],  
            "embedding": doc["embedding"],  
            "blob_name": doc["metadata"]["blob_name"],  
            "document_link": doc["metadata"]["document_link"],  
            "access_level": doc["metadata"]["access_level"]  
        })  
    search_client.upload_documents(documents=documents_to_upload)  
    status_placeholder.write("Documents uploaded to search index.") 

def create_prompt():
    system_prompt = "You are an AI assistant that helps people find information.\
              Ensure the Markdown responses are correctly formatted before responding. Only answer questions with the context given. \
             If answer not in context, say 'I do not know.'."
    return system_prompt
  
def chat_on_your_data(user_query, user_id):  
    """Perform retrieval queries over documents from the Azure AI Search Index."""  
    search_index = "documents-index"  
    messages = []  
  
    # Append user query to chat messages  
    messages.append({"role": "user", "content": user_query})  
  
    # Initialize the AzureOpenAI client  
    client = AzureOpenAI(  
        azure_endpoint=azure_endpoint,  
        api_key=azure_openai_api_key,  
        api_version=azure_openai_api_version,  
    )  
  
    # Create a chat completion with Azure OpenAI  
    completion = client.chat.completions.create(  
        model=azure_gpt_deployment,  
        messages=[  
            {"role": "system", "content": create_prompt()},  
            {"role": "user", "content": user_query}  
        ],  
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,  
        stop=None,  
        stream=False,  
        extra_body={  
            "data_sources": [{  
                "type": "azure_search",  
                "parameters": {  
                    "endpoint": search_endpoint,  
                    "index_name": search_index,  
                    "semantic_configuration": "default",  
                    "query_type": "vector_simple_hybrid",  
                    "fields_mapping": {},  
                    "in_scope": True,  
                    "role_information": "You are an AI assistant that helps people find information.",  
                    "filter": f"access_level/any(level: level eq '{user_id}') or access_level/any(level: level eq 'all')",  
                    "strictness": 3,  
                    "top_n_documents": 5,  
                    "authentication": {  
                        "type": "api_key",  
                        "key": search_key  
                    },  
                    "embedding_dependency": {  
                        "type": "deployment_name",  
                        "deployment_name": azure_ada_deployment  
                    }  
                }  
            }]  
        }  
    )  
  
    # Extract the response data  
    response_data = completion.to_dict()  
    # ai_response = response_data['choices'][0]['message']['content']  
    # # Clean up the AI response  
    # ai_response_cleaned = re.sub(r'\s+\.$', '.', re.sub(r'\[doc\d+\]', '', ai_response))  
    # citation = response_data["choices"][0]["message"]["context"]["citations"][0]["url"]  
    # ai_response_final = f"{ai_response_cleaned}\n\nCitation(s):\n{citation}"  
  
    return response_data 

def clear_session(messages):  
    # Clear necessary session state variables  
    st.cache_data.clear()  
    messages.clear()  
    return messages  

def extract_citations_content(data):  
    contents = []  
    try:  
        choices = data.get('choices', [])  
        for choice in choices:  
            message = choice.get('message', {})  
            context = message.get('context', {})  
            citations_list = context.get('citations', [])  
            for citation in citations_list:  
                content = citation.get('content', None)  
                if content:  
                    contents.append(content)  
    except Exception as e:  
        print(f"An error occurred: {e}")  
    return contents   

def get_time():  
    # Capture the current date and time in UTC (MySQL Native timezone)  
    current_utc_time = datetime.now(timezone.utc)  
    # Format the date and time to the desired string format  
    formatted_time = current_utc_time.strftime('%Y-%m-%d %H:%M:%S')  
    return formatted_time  
  
def main():  
    st.title("Azure OpenAI x NIH Cloud Tagging")  
    st.sidebar.title("Navigation")  
    user_email, user_id = get_user_email()
    rag_function = st.sidebar.radio(label="Choose a RAG function below:", options=['Vectorize', 'Chat'])  
  
    if rag_function == "Vectorize":  
        st.header("Document Vectorizer")  
        st.write("To begin vectorizing documents, please press the start button below.")  
        if st.button("Start Vectorization"):  
            with st.spinner("Vectorizing documents..."):  
                vectorize()  
            st.success("Vectorization complete!")  
  
    if rag_function == "Chat":  
        st.header("Chat with AI")  
        if 'messages' not in st.session_state:  
            st.session_state.messages = []  

        for message in st.session_state.messages:  
            with st.chat_message(message["role"]):  
                st.markdown(message['content'])
  
        user_input = st.chat_input("Enter query here...")  
        time_asked = get_time()

        if user_input:
            with st.spinner('Processing...'):  
                st.session_state.messages.append({"role": "user", "content": user_input}) 
                with st.chat_message("user"):  
                    st.markdown(user_input)  
                response_data = chat_on_your_data(user_input, user_id) 
                print(response_data) 
                # Call the function and print the content of each citation  
                citations_content = extract_citations_content(response_data) 
                citations = ""
                i = 1  
                for content in citations_content:  
                    citation = f"Citation {i}: \n{content}\n\n" 
                    citation = citation.replace("extracted_text.txt", "Cloud Tagging Strategy Guide_v3.1.pdf")
                    citations += citation
                    i+=1
                # print(citations)
                ai_response = response_data['choices'][0]['message']['content']  
                # Clean up the AI response  
                ai_response_cleaned = re.sub(r'\s+\.$', '.', re.sub(r'\[doc\d+\]', '', ai_response))  
                try:
                    citation_link = response_data["choices"][0]["message"]["context"]["citations"][0]["url"]  
                    citation_link = citation_link.replace("extracted_text.txt", "Cloud Tagging Strategy Guide_v3.1.pdf")
                    citation_link = citation_link.replace(" ", "%20")  
                    ai_response_final = f"{ai_response_cleaned}\n\nCitation(s):\n{citation_link}"  
                except IndexError as e:
                    ai_response_final = f"{ai_response_cleaned}\n\nCitation(s):\nNot found."  
                st.session_state.messages.append({"role": "assistant", "content": ai_response_final})  
                with st.chat_message("assistant"):  
                    st.markdown(ai_response_final)  
                    st.write('-'*50)
            # Call Code API to capture metadata
            url = "https://code-api.azurewebsites.net/code_api"  
            
            # The following data must be sent as payload with each API request.
            data = {  
                "system_prompt": f"{create_prompt()}\n\nContext:\n{citations}",  # All system prompts used including retrieved docs and any memory
                "current_user": user_id, # Entra ID object ID for the user who asked prompt
                "user_prompt": user_input,  # User prompt in which the end-user asks the model. 
                "user_prompt_tokens": response_data['usage']['prompt_tokens'],
                "time_asked": time_asked, # Time in which the user prompt was asked.
                "response": ai_response_final,  # Model's answer to the user prompt
                "response_tokens": response_data['usage']['completion_tokens'],
                "deployment_model": f'{azure_gpt_deployment}, {azure_ada_deployment}', # Input your model deployment names here
                "name_model": "gpt-4o, text-embedding-ada-002",  # Input your models here
                "version_model": "2024-05-13, 2",  # Input your model version here. NOT API VERSION.
                "region": "East US 2",  # Input your AOAI resource region here
                "project": "NIH - Cloud Tagging Demo",  # Input your project name here. Following the system prompt for this test currently :)
                "api_name": url, # Input the url of the API used. 
                "retrieve": True, # Set to True, indicating you are utilizing RAG.
                "database": "mysqldb" # Set to cosmosdb or mysqldb depending on desired platform
            }  
            
            response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))  
        
            print(response.status_code)  
            print(response.json())  
        clear_chat_placeholder = st.empty()  
        if clear_chat_placeholder.button('Start New Session'):  
            st.session_state.messages = clear_session(st.session_state.messages)  
            clear_chat_placeholder.empty()  
            st.success("Chat session has been reset. ")  
if __name__ == '__main__':  
    global_page_style2()
    main()  