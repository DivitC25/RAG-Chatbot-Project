import os.path
import tiktoken
import time


from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetreivalQA
from pinecone import Pinecone
from pinecone import ServerlessSpec
from tqdm.auto import tqdm
from uuid import uuid4


from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]
OPENAI_API_KEY = "sk-proj-scOvg2lMhvUKkvj1U94LT3BlbkFJIaxfPOfbPo6f1UjayiT7"
PINECONE_API_KEY = ""


def load_credentials():
  """Shows basic usage of the Drive v3 API.
  Prints the names and ids of the first 10 files the user has access to.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  return creds

def get_files(folderID):
  loader = GoogleDriveLoader(
    folder_id=folderID,
    token_path="token.json",
    recursive=False
    )
  docs = loader.load()
  return docs

def tiktoken_len(text):
  tokenizer = tiktoken.get_encoding('cl100k_base')
  tokens = tokenizer.encode(
    text,
    disallowed_special=()
  )
  return len(tokens)

def create_database():
  pc = Pinecone(api_key=PINECONE_API_KEY)
  spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
  )
  index_name = 'mohara-gpt'
  existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
  ]

  if index_name not in existing_indexes:
    pc.create_index(
      index_name,
      dimension=1536,
      metric='dotproduct',
      spec=spec
    )

    while not pc.describe_index(index_name).status['ready']:
      time.sleep(1)

  index = pc.Index(index_name)
  return index


def main():

  #Initialization
  model_name = 'text-embedding-ada-002'
  creds = load_credentials()

  #Data Retreival from Google Drive Folder
  data = get_files("16R9vacm0pb8EG13d8ZpjsNa2YR-mI3wQ")
  file_array = [file.page_content for file in data]
  metadata_array = [file.metadata for file in data]
  files = ' '.join(file_array)

  #LangChain & OpenAI Text Processing
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )
  embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key = OPENAI_API_KEY
  )

  #Database Indexing
  index = create_database()
  batch_limit = 100
  texts = []
  metadatas = []

  for i, record in enumerate(tqdm(data)):
    metadata = {
      'title': record.metadata['title'],
      'source': record.metadata['source']
    }
    record_texts = text_splitter.split_text(record.page_content)
    record_metadatas = [{
      "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)

    if len(texts) >= batch_limit:
      ids = [str(uuid4()) for _ in range(len(texts))]
      embeds = embed.embed_documents(texts)
      index.upsert(vectors=zip(ids, embeds, metadatas))
      texts = []
      metadatas = []
  
  if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))

  #Creating Vector Store 
  text_field = "text"
  vectorstore = Pinecone(
    index, embed.embed_query, text_field
  )

  #Query Logic
  llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
  )
  qa = RetreivalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retreiver=vectorstore.as_retreiver()
  )

if __name__ == "__main__":
  main()