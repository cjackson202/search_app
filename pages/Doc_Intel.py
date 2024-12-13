import os  
import logging  
import re  
import streamlit as st  
from azure.storage.blob import BlobServiceClient  
from azure.ai.formrecognizer import DocumentAnalysisClient  
from azure.core.credentials import AzureKeyCredential  
from dotenv import load_dotenv 
from styling import global_page_style2 
  
# Load environment variables from .env file  
load_dotenv()  
  
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
def download_blob_as_bytes(container_name, blob_name, connection_string, download_path):  
    logging.info(f"Downloading blob {blob_name} from container {container_name}")  
    # Create a BlobServiceClient using the connection string  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    # Get a blob client  
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)  
    # Download the blob's content as bytes and save it to a file  
    with open(download_path, "wb") as download_file:  
        download_file.write(blob_client.download_blob().readall())  
    logging.info(f"Downloaded blob {blob_name} to {download_path}")  
  
def upload_blob(container_name, blob_name, connection_string, upload_path):  
    logging.info(f"Uploading blob {blob_name} to container {container_name}")  
    # Create a BlobServiceClient using the connection string  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    # Get a blob client  
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)  
    # Upload the file  
    with open(upload_path, "rb") as data:  
        blob_client.upload_blob(data, overwrite=True)  
    logging.info(f"Uploaded blob {blob_name} to container {container_name}")  
  
def clean_text(content):  
    # Add a space between text and numbers in table of contents style  
    content = re.sub(r'(\D)(\d)', r'\1 \2', content)  
    return content  
  
def format_table(table):  
    # Create a 2D list to store the table content  
    table_data = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]  
  
    # Populate the table data  
    for cell in table.cells:  
        table_data[cell.row_index][cell.column_index] = clean_text(cell.content)  
  
    # Create a formatted table string  
    table_str = "Column titles: " + " -- ".join(table_data[0]) + "\n"  
    for row_index, row in enumerate(table_data[1:], start=1):  
        table_str += f"row{row_index}: " + " -- ".join(row) + "\n"  
  
    return table_str  
  
def main():  
    st.title("Azure Blob Storage Document Analysis")  
  
    container_name = st.text_input("Blob Container Name", os.getenv("BLOB_CONTAINER_NAME"))  
    blob_name = st.text_input("Blob Name", "Cloud Tagging Strategy Guide_v3.1.pdf")  
    connection_string = st.text_input("Blob Connection String", os.getenv("BLOB_CONNECTION_STRING"), type="password")  
    endpoint = st.text_input("Azure Document Intelligence Endpoint", os.getenv("AZURE_DOC_INTEL_URL"))  
    key = st.text_input("Azure Document Intelligence Key", os.getenv("AZURE_DOC_INTEL_KEY"), type="password")  
  
    if st.button("Analyze Document"):  
        if not all([container_name, blob_name, connection_string, endpoint, key]):  
            st.error("Please provide all the required inputs")  
            return  
          
        download_path = "temp_downloaded_file.pdf"  
        output_text_file = "extracted_text.txt"  
        extracted_blob_name = "extracted_text.txt"  
  
        try:  
            # Download the PDF from blob storage  
            download_blob_as_bytes(container_name, blob_name, connection_string, download_path)  
  
            # Read the file content  
            with open(download_path, "rb") as pdf_file:  
                pdf_data = pdf_file.read()  
  
            # Create a client for Azure Form Recognizer  
            document_analysis_client = DocumentAnalysisClient(  
                endpoint=endpoint,  
                credential=AzureKeyCredential(key)  
            )  
  
            # Analyze the downloaded PDF using the layout model  
            logging.info("Starting analysis of the document")  
            poller = document_analysis_client.begin_analyze_document(  
                "prebuilt-layout", document=pdf_data  
            )  
  
            # Get the result  
            result = poller.result()  
            logging.info("Document analysis completed")  
  
            # Write the extracted text to a text file with improved formatting  
            with open(output_text_file, "w", encoding="utf-8") as text_file:  
                for page in result.pages:  
                    logging.info(f"Processing page {page.page_number}")  
                    page_content = []  
                    page_tables = [table for table in result.tables if table.bounding_regions[0].page_number == page.page_number]  
  
                    # Collect all table regions to exclude from line extraction  
                    table_lines = set()  
                    for table in page_tables:  
                        for cell in table.cells:  
                            for line in cell.content.splitlines():  
                                table_lines.add(line.strip())  
  
                    for line in page.lines:  
                        cleaned_content = clean_text(line.content)  
                        if cleaned_content not in table_lines:  
                            page_content.append(cleaned_content)  
  
                    text_file.write("\n".join(page_content) + "\n")  
                    # Insert tables where they appear on the page  
                    for table in page_tables:  
                        text_file.write("\nTable:\n")  
                        formatted_table = format_table(table)  
                        text_file.write(formatted_table)  
                        text_file.write("\n")  
  
                    # Add a page break (optional)  
                    text_file.write("\n--- End of Page ---\n\n")  
  
            logging.info(f"Extracted text saved to {output_text_file}")  
  
            # Display the extracted text in Streamlit  
            with open(output_text_file, "r", encoding="utf-8") as text_file:  
                extracted_text = text_file.read()  
                st.text_area("Extracted Text", extracted_text, height=300)  
  
            # Upload the extracted text file back to the blob container  
            upload_blob(container_name, extracted_blob_name, connection_string, output_text_file)  
  
        except Exception as e:  
            logging.error(f"An error occurred: {e}")  
            st.error(f"An error occurred: {e}")  
  
        finally:  
            # Clean up: Delete the temporary file  
            if os.path.exists(download_path):  
                os.remove(download_path)  
                logging.info(f"Deleted temporary file {download_path}")  
            else:  
                logging.warning(f"The file {download_path} does not exist")  
  
if __name__ == "__main__":  
    global_page_style2()
    main()  