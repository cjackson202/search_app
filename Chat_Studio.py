import streamlit as st  
import os  
import io  
import pdfplumber  
import asyncio
from typing import Annotated
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
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
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv  
from openai import AzureOpenAI 
from datetime import datetime, timezone 
import tiktoken  
from styling import global_page_style2
  
# Load environment variables  
load_dotenv()  
  
# Configure Azure AI Search parameters  
search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')  
search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')  
search_index = os.getenv('AZURE_SEARCH_INDEX')

# Configure Azure OpenAI parameters  
azure_completions_endpoint = os.getenv('APIM_COMPLETIONS_URL')  
azure_embeddings_endpoint = os.getenv('APIM_EMBEDDINGS_URL')
azure_openai_api_key = os.getenv('APIM_API_KEY')  
azure_openai_api_version = os.getenv('AZURE_OPENAI_VERSION')  
azure_ada_deployment = os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT')  
azure_gpt_deployment = os.getenv('AZURE_GPT_DEPLOYMENT') 
azure_completions_endpoint = azure_completions_endpoint.replace("{model}", azure_gpt_deployment).replace("{version}", azure_openai_api_version) 
azure_embeddings_endpoint = azure_embeddings_endpoint.replace("{model}", azure_ada_deployment).replace("{version}", azure_openai_api_version) 

def get_time():
    # Capture the current date and time in UTC (MySQL Native timezone)
    current_utc_time = datetime.now(timezone.utc)  
    # Format the date and time to the desired string format  
    formatted_time = current_utc_time.strftime('%Y-%m-%d %H:%M:%S') 
    return formatted_time 

def get_user_email():
    # Extract headers using Streamlit's context
    headers = st.context.headers
    # The header names may vary; check the specific headers set by Azure App Service
    user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID')
    return user_id

user_id = get_user_email()

headers = {
    'Content-Type':'application/json',
    "system_prompt": '',  # Leave empty string.
    "current_user": user_id,
    "user_prompt": '',  # Leave empty string.
    "time_asked": get_time(), # Time in which the user prompt was asked.
    "deployment_model": azure_gpt_deployment, # Input your model's deployment name here
    "name_model": "gpt-4o",  # Input you model here
    "version_model": "2024-05-13",  # Input your model version here. NOT API VERSION.
    "region": "East US 2",  # Input your AOAI resource region here
    "project": "NIH - Cloud Tag v2",  # Input your project name here. Following the system prompt for this test currently :)
    "database": "mysqldb", # Specify here cosmosdb or mysql as database. 
    "retrieve": "False" # Must specify True or False here as string 
}

headers2 = {
    'Content-Type':'application/json',
    "system_prompt": '**No system prompt for NIH-Cloud Tag v2 Embeddings**)',  # Leave empty string.
    "current_user": user_id, # Entra ID object id for a user in the Entra tenant
    "user_prompt": '',  # Leave empty string.
    "time_asked": get_time(), # Time in which the user prompt was asked.
    "deployment_model": azure_ada_deployment, # Input your model's deployment name here
    "name_model": "text-embedding-ada-002",  # Input you model here
    "version_model": "2",  # Input your model version here. NOT API VERSION.
    "region": "East US 2",  # Input your AOAI resource region here
    "project": "NIH - Cloud Tag v2",  # Input your project name here. Following the system prompt for this test currently :)
    "database": "mysqldb", # Specify here cosmosdb or mysql as database. 
    "retrieve": "False" # Must specify True or False here as string 
}
  
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

# Define RAG Plug in for Kernel
class RAGPlugin:
    """A RAG Plugin for querying over documents from the Azure AI Search Index."""
    @kernel_function(description="Provides a guide about Cloud Tagging Strategy.")
    def retrieve(self, query: Annotated[str, "The question of the user."]) -> Annotated[str, "Returns relevant documents about Cloud Tagging Strategy."]:
        print("RAG tool being used.")
        # Initialize the AzureOpenAI client
        client = AzureOpenAI(
            azure_endpoint=azure_embeddings_endpoint,
            api_key=azure_openai_api_key,
            api_version=azure_openai_api_version, default_headers=headers2
        )
        # Get the vector representation of the user query using the Ada model
        embedding_response = client.embeddings.create(
            input=query,
            model=azure_ada_deployment
        )
        query_vector = embedding_response.data[0].embedding
        # Create a SearchClient  
        search_client = SearchClient(endpoint=search_endpoint,  
                                    index_name=search_index,  
                                    credential=AzureKeyCredential(search_key))  
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=5, fields='embedding', exhaustive=True)
        # Query the index  
        results = search_client.search(
            search_text=query,
            vector_queries= [vector_query],
            filter=f"access_level/any(level: level eq '{user_id}') or access_level/any(level: level eq 'all')",
            top=5,
            select=["content"],
            )  
        # Print the results 
        i = 0 
        final_output = ""
        for result in results: 
            i += 1 
            content = result.get('content','')
            final_output += f'Source {i}\n{content}\n{"-"*50}\n\n'
        return final_output
  
async def invoke_agent(agent: ChatCompletionAgent, input: str, chat: ChatHistory) -> None:
    """Invoke the agent with the user input."""
    chat.add_user_message(input)
    # print(f"# {AuthorRole.USER}: '{input}'")
    streaming = False
    if streaming:
        contents = []
        content_name = ""
        async for content in agent.invoke_stream(chat):
            content_name = content.name
            contents.append(content)
        message_content = "".join([content.content for content in contents])
        # print(f"# {content.role} - {content_name or '*'}: '{message_content}'")
        chat.add_assistant_message(message_content)
        return message_content
    else:
        contents = []
        async for content in agent.invoke(chat):
            contents.append(content)
        message_content = "".join([content.content for content in contents])
            # print(f"\n# {content.role} - {content.name or '*'}: {content.content}\n\n")
        chat.add_message(content)
        return message_content

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
  
async def main():  
    st.title("Azure OpenAI x NIH Cloud Tagging")  
    st.sidebar.title("Navigation")  
    # user_email, user_id = get_user_email()
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
        # Define the agent name and instructions
        AGENT_NAME = "Cloud_Tag_Specialist"
        AGENT_INSTRUCTIONS = """
        You are a Cloud Tagging Strategy assistant. 
        You are here to help navigate users through cloud tagging concepts and answer any questions they might have using the cloud tagging guide. 
        Users can ask about:
        - Tagging strategies for different cloud environments (e.g., AWS, Azure, Google Cloud)
        - Best practices for implementing tagging in your organization
        - Use cases and examples of effective tagging
        - Benefits of a well-structured tagging strategy
        - Common challenges and how to overcome them
        You must retrieve all information from the given plug-in. Cite all answers with the file name.
        If question can't be answered with plug-in, simple say I do not know.
        You will interact with users ranging from beginner to expert levels. 
        Guide users and suggest questions that can be asked if needed.
        """ 
        # Create the instance of the Kernel
        kernel = Kernel()
        service_id = "agent"
        kernel.add_service(AzureChatCompletion(service_id=service_id, api_key=azure_openai_api_key, endpoint=azure_completions_endpoint, api_version=azure_openai_api_version,
                                            deployment_name=azure_gpt_deployment, default_headers=headers))
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
        # Configure the function choice behavior to auto invoke kernel functions
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        kernel.add_plugin(RAGPlugin(), plugin_name="cloud_tag_guide")
        # Create the agent
        agent = ChatCompletionAgent(
            service_id="agent", kernel=kernel, name=AGENT_NAME, instructions=AGENT_INSTRUCTIONS, execution_settings=settings
        )
        # Define the chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = ChatHistory()

        if 'messages' not in st.session_state:  
            st.session_state.messages = []  

        for message in st.session_state.messages:  
            with st.chat_message(message["role"]):  
                st.markdown(message['content'])
  
        user_input = st.chat_input("Enter query here...")  

        if user_input:
            with st.spinner('Processing...'):  
                st.session_state.messages.append({"role": "user", "content": user_input}) 
                with st.chat_message("user"):  
                    st.markdown(user_input)  
                response = await invoke_agent(agent, user_input, st.session_state.chat_history)
 
                st.session_state.messages.append({"role": "assistant", "content": response})  
                with st.chat_message("assistant"):  
                    st.markdown(response)  
                    st.write('-'*50)

        clear_chat_placeholder = st.empty()  
        if clear_chat_placeholder.button('Start New Session'):  
            st.session_state.messages = clear_session(st.session_state.messages)  
            clear_chat_placeholder.empty()  
            st.success("Chat session has been reset. ")  
if __name__ == '__main__':  
    global_page_style2()
    asyncio.run(main())