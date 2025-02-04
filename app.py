import os
import json
from flask import Flask, render_template, request, jsonify
from flask import Flask, request, render_template, redirect, url_for, flash
import ollama
from werkzeug.utils import secure_filename
from config import Configuration
# from openai import ChatCompletion
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
# from langchain.tools import Tool
# from langchain.chains import RetrievalQA

import datetime


#for semantic chunking

from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

#for  data in MongoDB
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_mongodb import MongoDBAtlasVectorSearch


from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


#AWS Bedrock

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as botoConfig

#multiagent

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from IPython.display import Image, display
from IPython.display import Image, display
from typing import Any
import operator
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
import os


PROMPT_TEMPLATE= """Use the following context and previous messages to answer the question or summarize the conversation so far.

- Only use the provided context to generate a response. If the required information is missing, state that you don’t know the answer instead of making one up.
- Do not answer if the question is unrelated to the given context.
- If applicable, you may suggest external resources but do not attempt to provide an answer beyond what is available in the context.
- If relevant, provide web links to external sources where the user can find more information.

### Previous Messages:
{prevMessages}

### Context for the Answer:
{context}

------------------
### Question:
{question}

------------------
### Answer Format:
- Provide a direct answer if the information is available in the context.
- If the information is missing, state: "The provided context does not contain the required information."
- If the question is unrelated, state: "This question is not relevant to the given context."
- If suggesting resources, format as:  
  - **External Sources:** Provide useful web links if available. Format as: "[Resource Name](URL)"  
  - Example: "You may refer to [MongoDB Documentation](https://www.mongodb.com/docs/) for more details."
"""




# Initialize the Flask application
app = Flask(__name__,static_folder='static')
app.config['DEBUG'] = True
app.config['TESTING'] = False


#import config file
app.config.from_object(Configuration)
DB_NAME=app.config['DB_NAME']
COLLECTION_NAME=app.config['COLLECTION_NAME']

#open ai variables
openai_client = OpenAI(api_key=app.config['OPEN_AI_KEY'])
os.environ["OPENAI_API_KEY"]=app.config['OPEN_AI_KEY']
os.environ["TAVILY_API_KEY"]=app.config['TAVILY_API_KEY']

#AWS
retry_config = botoConfig(
   retries={
        'total_max_attempts': 10
    }
)

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',  # Replace with your AWS region
    aws_access_key_id=app.config['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=app.config['AWS_SECRET_ACCESS_KEY'],config=retry_config
)



AZURE_OPENAI_API_KEY = app.config['AZURE_OPENAI_API_KEY']
AZURE_OPENAI_ENDPOINT =app.config['AZURE_OPENAI_ENDPOINT']
AZURE_DEPLOYMENT_NAME = app.config['AZURE_OPENAI_DEPLOYMENT']
AZURE_OPENAI_API_VERSION=app.config['AZURE_OPENAI_API_VERSION']


# Set configuration variables
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}    # Allowed file types

#ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def upload_form():
    return render_template('index.html')



def load_documents(pdf_file_path):
    try:
        document_loader = PyPDFLoader(pdf_file_path)
        pdf=document_loader.load()
        if len(pdf)==0:
            raise ValueError("The PDF file is empty")
        return pdf
    except Exception as e:
        print(e)
        raise e

def genertateChunkMetatdata(chunks):
    try: 
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
        # print(chunk.metadata.get("source"))
            source=str(chunk.metadata['source']).split("/")[-1]
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["file_id"]=source
        return chunks
    except Exception as e:
        print(e)
        raise e

def recursiveChunker(pdf,embedding_model,chunking_strategy):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        chunks=text_splitter.split_documents(pdf)

        chunks=genertateChunkMetatdata(chunks)
        #generate chunk metadata 

        return generateEmbeddings(chunks,embedding_model,chunking_strategy)
    
    except Exception as e:
        print(e)
        raise e
    


def semanticChunker(pdf,embedding_model,chunking_strategy):
    try:
        # text_splitter= SemanticChunker(OpenAIEmbeddings(),breakpoint_threshold_type="percentile") #open ai semantic chunking
        text_splitter= SemanticChunker(OllamaEmbeddings(model="mxbai-embed-large"))
        chunks=text_splitter.split_documents(pdf)

        chunks=genertateChunkMetatdata(chunks)
        #generate chunk metadata 

        return generateEmbeddings(chunks,embedding_model,chunking_strategy)
    
    except Exception as e:
        print(e)
        raise e
    

def generateEmbeddings(Chunks,embedding_model:str,chunking_strategy:str):
    try:
        master_data=[]
        for chunk in Chunks:
            data={}
            data["page_content"]=chunk.page_content
            data["metadata"]=chunk.metadata
            data["embedding"]=ollama.embeddings(model=embedding_model,prompt=chunk.page_content)["embedding"]
            data['metadata']["embedding_model"]=embedding_model
            data['metadata']["chunking_strategy"]=chunking_strategy
            data["pk"]=data['metadata']['chunk_id']+"/"+embedding_model+"/"+chunking_strategy
            data["object_ref"]=data['metadata']['file_id']+"/"+embedding_model+"/"+chunking_strategy
            master_data.append(data)
        return master_data
    except Exception as e:
        print(e)
        raise e

def indextoMongoDB(data:list):
    client=MongoClient(app.config['MONGODB_URI'],server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        db = client[app.config['DB_NAME']]
        collection = db[app.config['COLLECTION_NAME']]
        result=collection.insert_many(data)
        print("Insert complete!")
    except Exception as e:
        print(e)
        raise e
    finally:
        client.close()


@app.route('/get-options', methods=['GET'])
def get_options():
    # Fetch unique values from MongoDB
    # print("entering get options")
    pipeline = [
        {
            "$group": {
                "_id": {
                    "file_id": "$metadata.file_id",
                    "embedding_model": "$metadata.embedding_model",
                    "chunking_strategy": "$metadata.chunking_strategy"
                }
            }
        },
        {
            "$project": {
                "file_id": "$_id.file_id",
                "embedding_model": "$_id.embedding_model",
                "chunking_strategy": "$_id.chunking_strategy",
                "_id": 0
            }
        }
    ]
    client=MongoClient(app.config['MONGODB_URI'],server_api=ServerApi('1'))
    try:
       
        client.admin.command('ping')
        db = client[app.config['DB_NAME']]
        collection = db[app.config['COLLECTION_NAME']]
        unique_documents = list(collection.aggregate(pipeline))
        return jsonify(unique_documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        client.close()
    
def isNotDuplicate(val,target):
    val=val.get_json()
    for i in val:
        src=i["file_id"]+"/"+i["embedding_model"]+"/"+i["chunking_strategy"]
        if(src==target):
            return False
    return True

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    
    model_choice = request.form.get('model')  # Get the model choice from the form
    chunking_strategy = request.form.get('chunking_strategy')  # Get the chunking strategy from the form
    
    if file and allowed_file(file.filename):
        # Secure the filename before saving it
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(file_path)=./uploads/6_709_reg.pdf
        key=filename+"/"+model_choice+"/"+chunking_strategy
        
        try:
            val=get_options()
            if isNotDuplicate(val,key):
                file.save(file_path)  # Save the file to the 'uploads/' folder
        # flash("waiting for response")
            # flash("Processing the file!", category="info")
        # Call OpenAI's new embedding API

        #add logic to check if file exists in the database
                pdf=load_documents(file_path)
                if(chunking_strategy=="semantic"):
                    master_data=semanticChunker(pdf,model_choice,chunking_strategy)
                else:
                    master_data=recursiveChunker(pdf,model_choice,chunking_strategy)
                # print(master_data)
                indextoMongoDB(master_data)
                flash(f'File {filename} chunked and stored, proceed to chat!', 'success')
                return redirect(url_for('upload_form'))  # Redirect back to the upload form
            
            else:
                flash(f'File {filename} already chunked and stored, proceed to chat!', 'info')
                return redirect(url_for('upload_form'))  # Redirect back to the upload form
            

        except Exception as e:
            flash(f'Error occured: {e}', 'error')
        
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                # print(f"File '{file_path}' has been deleted successfully.")

        
    
    flash('Invalid file type. Please upload a PDF.', 'error')
    return redirect(request.url)







@retry(wait=wait_random_exponential(min=1, max=1), stop=stop_after_attempt(6))
def chat_completion_backoff(**kwargs):
    return openai_client.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=1), stop=stop_after_attempt(6))
def embedding_create_backoff(**kwargs):
    return openai_client.embeddings.create(**kwargs)


def summarize(previous_messages,client,conversation_id):
    try:
        chat_collection = client[DB_NAME]['chat_history']
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.7,
            max_tokens=4096
            )
        prompt="""
        I have been chatting with a GenAI-based application to understand a few concepts.
         I am passing my previous conversation as additional context to the LLM. Below are my past messages: {prevMessages}. 
         Please summarize them concisely so I can pass them back and continue the conversation without losing context.
         """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        prompt_text = prompt_template.format(prevMessages=previous_messages)
        response = llm.invoke(input=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt_text}
                    ])
        update_conversation(client, conversation_id, "summary of the conversation thus far", response.content,True)
        return get_conversation(client, conversation_id)
        
    except Exception as e:
        print(str(e))


def get_conversation(client, conversation_id):
    try:
        chat_collection = client[DB_NAME]['chat_history']
        chat_history = chat_collection.find_one({'conversation_id': conversation_id})
        if not chat_history:
            # Initialize new conversation
            chat_history = {
                    'conversation_id': conversation_id,
                    'messages': [],
                    'created_at': datetime.datetime.now()
                }
            chat_collection.insert_one(chat_history)
            
        previous_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in chat_history.get('messages', [])
            ]
        #summarize every 3 messages
        if previous_messages.__len__()>6:
            return summarize(previous_messages,client,conversation_id)
        return previous_messages
    except Exception as e:
        print(str(e))
        return []

#updates exiting conversation with new messages
def update_conversation(client, conversation_id, query, response,summary):
    try:
        chat_collection = client[DB_NAME]['chat_history']
        if summary is True:
            chat_collection.delete_one({'conversation_id': conversation_id})
            chat_history = {
                    'conversation_id': conversation_id,
                    'messages': [],
                    'created_at': datetime.datetime.now()
                }
            chat_collection.insert_one(chat_history)
        chat_collection.update_one(
                        {'conversation_id': conversation_id},
                        {
                            '$push': {
                             'messages': {
                                 '$each': [
                                    {"role": "user", "content": query},
                                    {"role": "assistant", "content": response}
                                 ]
                                }
                             }
                        }
                        )
    except Exception as e:
        print(str(e))


def get_context_data(client,embedding_model,vector_index,query_text,prefilter):
    mongo_collection=client[DB_NAME][COLLECTION_NAME]
    vector_store = MongoDBAtlasVectorSearch(
                    collection=mongo_collection,
                    embedding=embedding_model,
                    index_name=vector_index,
                    relevance_score_fn="cosine",
                    text_key="page_content")
                
    retrieved_docs = vector_store.similarity_search_with_score(query=query_text, k=10,pre_filter={"object_ref":prefilter})
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in retrieved_docs])
    return context_text

def generate_prompt(client,embedding_model,vector_index,query_text,prefilter,previous_messages,):
     try:
        context_text = get_context_data(client,embedding_model,vector_index,query_text,prefilter)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_text = prompt_template.format(prevMessages=previous_messages, context=context_text, question=query_text) 
        return prompt_text
     except Exception as e:
        print(str(e))
        return None     


@app.route('/ask/ollama', methods=['POST'])
def ask_ollama():
    query_text = request.json.get('message')
    prefilter=request.json.get('selectedOption')
    isAgentic=request.json.get('isAgentic')
    llm=app.config['OLLAMA_MODEL']
    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    conversation_id = request.json.get('sessionId')
    if query_text:
        try:
            previous_messages=get_conversation(client, conversation_id)
            if(prefilter=="dummy"):
                response = ollama.chat(model=llm, messages=previous_messages+[{'role': 'user', 'content':query_text },]) #role can be  'user', 'assistant', 'system' or 'tool'
                if response.done:
                    update_conversation(client, conversation_id, query_text, response.message.content,False)
                 
                    return jsonify({"response":response.message.content});
                else:
                    return jsonify({"error": f"Failed to get response from LLM, status code: {response.status_code}"}), 500
            else:
                metadata=prefilter.split("/")
                if(metadata[1]=='mxbai-embed-large'):
                    embeddingModel=metadata[1]
                    vector_index=app.config['MX_VECTOR']
                else:
                    embeddingModel="nomic-embed-text"
                    vector_index=app.config['NOM_VECTOR']
                
            
                model = OllamaLLM(model=llm)
                embedding_model = OllamaEmbeddings(model=embeddingModel)

                prompt_text=generate_prompt(client,embedding_model,vector_index,query_text,prefilter,previous_messages)

                if(prompt_text is  None):
                    prompt_text="Error generating prompt retry"               
                response=model.invoke(prompt_text)
                update_conversation(client, conversation_id, query_text, response,False)
                return jsonify({"response":response,"prompt":prompt_text})
          
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            client.close()
    return jsonify({"error": "No message provided"}), 400



@app.route('/ask/gpt-3.5', methods=['POST'])
def ask_gpt35():
    query_text = request.json.get('message')
    prefilter=request.json.get('selectedOption')
    isAgentic=request.json.get('isAgentic')
    llm = AzureChatOpenAI(
    azure_endpoint=app.config['AZURE_GPT35_ENDPOINT'],
    openai_api_key=app.config['AZURE_GPT35_KEY'],
    openai_api_version=app.config['AZURE_GPT35_API_VERSION'],   
    temperature=0.7,
    max_tokens=4096
    )
    conversation_id = request.json.get('sessionId')
    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    if query_text:
        try:
            previous_messages=get_conversation(client, conversation_id)
            if(prefilter=="dummy"):
                response = llm.invoke(previous_messages+[
                        {"role": "system", "content": "you are a helpful chatbot"},
                        {"role": "user", "content": query_text},
                    ]
                )
               
                update_conversation(client, conversation_id, query_text, response.content,False)
       
                return jsonify({"response":response.content})
            else:
                metadata=prefilter.split("/")
                if(metadata[1]=='mxbai-embed-large'):
                    embeddingModel=metadata[1]
                    vector_index=app.config['MX_VECTOR']
                else:
                    embeddingModel="nomic-embed-text"
                    vector_index=app.config['NOM_VECTOR']
                
                embedding_model = OllamaEmbeddings(model=embeddingModel)
                prompt_text=generate_prompt(client,embedding_model,vector_index,query_text,prefilter,previous_messages)

                if(prompt_text is  None):
                    prompt_text="Error generating prompt retry"     
                response = llm.invoke(input=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt_text}
                    ])
                update_conversation(client, conversation_id, query_text, response.content,False)
                return jsonify({"response": response.content,"prompt":prompt_text})

        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500
        
        finally:
            client.close()

    return jsonify({"error": "No message provided"}), 400


class State(TypedDict):
    question: str
    embedding_model:OllamaEmbeddings
    prefilter:str
    vector_index:str
    answer: str
    previous_messages: list
    context: Annotated[list, operator.add]
    prompt:str



def search_web(state):
    
    """ Retrieve docs from web search """

    # Search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    # print(formatted_search_docs)
    return {"context": [formatted_search_docs]} 

def search_wikipedia(state):
    
    """ Retrieve docs from wikipedia """

    # Search
    search_docs = WikipediaLoader(query=state['question'], 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    # print(formatted_search_docs)
    return {"context": [formatted_search_docs]} 

def search_mongodb(state):
    if(state["embedding_model"]==""):
        return{"context":[]}
    client=MongoClient(app.config['MONGODB_URI'],server_api=ServerApi('1'))
    context=get_context_data(client,state["embedding_model"],state["vector_index"],state["question"],state["prefilter"])
    return {"context": [context]}
    

def generate_answer(state):
    llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.7,
    max_tokens=4096
    )

    AGENTIC_TEMPLATE= """Use the following context and previous messages to answer the question or summarize the conversation so far.

- Only use the provided context to generate a response. If the required information is missing, state that you don’t know the answer instead of making one up.
- Do not answer if the question is unrelated to the given context.
- do not attempt to provide an answer beyond what is available in the context.
- use relevant web information to give more accurate answer

### Previous Messages:
{prevMessages}

### Context for the Answer:
{context}

------------------
### Question:
{question}

------------------
### Answer Format:
- Provide a direct answer if the information is available in the context.
- divide your answer into multiple points with sub headings if applicable
- If the information is missing, state: "The provided context does not contain the required information."
- If the question is unrelated, state: "This question is not relevant to the given context."
- If suggesting resources, format as:  
  - **External Sources:** \n Provide useful web links if available. Format as: "[Resource Name](URL)"  
  - Example: "\n You may refer to [MongoDB Documentation](https://www.mongodb.com/docs/) for more details."
"""

    prompt_template = ChatPromptTemplate.from_template(AGENTIC_TEMPLATE)
    prompt_text = prompt_template.format(prevMessages=state["previous_messages"], context=state["context"], question=state["question"]) 
    if(prompt_text is  None):
        prompt_text="Error generating prompt retry"     
    answer =llm.invoke([SystemMessage(content=prompt_text)]+[HumanMessage(content=f"Answer the question.")])
    return {"answer": answer.content,"prompt":prompt_text}
    


def build_graph():
    # Add nodes
    builder = StateGraph(State)


    # Initialize each node with node_secret 

    builder.add_node("search_web",search_web)
    builder.add_node("search_wikipedia", search_wikipedia)
    builder.add_node("search_mongodb", search_mongodb)
    builder.add_node("generate_answer", generate_answer)

    # Flow
    builder.add_edge(START, "search_wikipedia")
    builder.add_edge(START, "search_web")
    builder.add_edge(START, "search_mongodb")
    builder.add_edge("search_wikipedia", "generate_answer")
    builder.add_edge("search_web", "generate_answer")
    builder.add_edge("search_mongodb", "generate_answer")
    builder.add_edge("generate_answer", END)
    graph = builder.compile()

    display(Image(graph.get_graph().draw_mermaid_png()))
    return graph




@app.route('/ask/gpt-4.0', methods=['POST'])
def ask_gpt4o():
    query_text = request.json.get('message')
    prefilter=request.json.get('selectedOption')
    isAgentic=request.json.get('isAgentic')
    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.7,
    max_tokens=4096
    )
    conversation_id = request.json.get('sessionId')
    if query_text:
        try:
            if isAgentic:
                graph=build_graph()
            previous_messages=get_conversation(client, conversation_id)
            if(prefilter=="dummy"):
                if isAgentic:
                    response = graph.invoke({"question": query_text, "embedding_model":"","vector_index":"","prefilter":"","previous_messages":previous_messages})
                    response.content = response["answer"]
                else:
            # Call OpenAI's new chat-based API (using the correct model and chat completions)
                    response = llm.invoke(previous_messages+[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content":query_text}
                    ])

                update_conversation(client, conversation_id, query_text, response.content,False)
                return jsonify({"response":response.content})
            
            else:
                # print(prefilter)
                metadata=prefilter.split("/")
                if(metadata[1]=='mxbai-embed-large'):
                    embeddingModel=metadata[1]
                    vector_index=app.config['MX_VECTOR']
                else:
                    embeddingModel="nomic-embed-text"
                    vector_index=app.config['NOM_VECTOR']
                embedding_model = OllamaEmbeddings(model=embeddingModel)
                if isAgentic:
                    response = graph.invoke({"question": query_text, "embedding_model":embedding_model,"vector_index":vector_index,"prefilter":prefilter,"previous_messages":previous_messages})
                    response.content = response["answer"]
                    prompt_text=response["prompt"]
                else:
                    prompt_text=generate_prompt(client,embedding_model,vector_index,query_text,prefilter,previous_messages)
                    if(prompt_text is  None):
                        prompt_text="Error generating prompt retry"     
                    response = llm.invoke(input=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt_text}
                        ])
                update_conversation(client, conversation_id, query_text, response.content,False)
                return jsonify({"response": response.content,"prompt":prompt_text})
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"Error: {error_code} - {error_message}")
        except Exception as e:
             print(f"Unexpected error: {str(e)}")
        
        finally:
            client.close()

    return jsonify({"error": "No message provided"}), 400


@app.route('/ask/deepseek', methods=['POST'])
def ask_deepSeek():
    query_text = request.json.get('message')
    prefilter=request.json.get('selectedOption')
    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    REGION_NAME ='us-east-1'
    MODEL_ID= 'arn:aws:bedrock:us-east-1:979559056307:imported-model/kt0tr2ppv8lw'
    conversation_id = request.json.get('sessionId')
    if query_text:
        try:
            previous_messages=get_conversation(client, conversation_id)
            if(prefilter=="dummy"):
                invoke_response = bedrock_runtime.invoke_model(modelId=MODEL_ID, 
                                            body=json.dumps({'prompt': query_text}), 
                                            accept="application/json", 
                                            contentType="application/json")
                invoke_response["body"] = json.loads(invoke_response["body"].read().decode("utf-8"))
                update_conversation(client, conversation_id, query_text, invoke_response["body"]['generation'],False)
                return jsonify({"response":invoke_response["body"]['generation']})
            else:
                metadata=prefilter.split("/")
                if(metadata[1]=='mxbai-embed-large'):
                    embeddingModel=metadata[1]
                    vector_index=app.config['MX_VECTOR']
                else:
                    embeddingModel="nomic-embed-text"
                    vector_index=app.config['NOM_VECTOR']
                embedding_model = OllamaEmbeddings(model=embeddingModel)
                prompt_text=generate_prompt(client,embedding_model,vector_index,query_text,prefilter,previous_messages)
                if(prompt_text is  None):
                    prompt_text="Error generating prompt retry"     
                invoke_response = bedrock_runtime.invoke_model(modelId=MODEL_ID, 
                                            body=json.dumps({'prompt': prompt_text}), 
                                            accept="application/json", 
                                            contentType="application/json")
                invoke_response["body"] = json.loads(invoke_response["body"].read().decode("utf-8"))
                update_conversation(client, conversation_id, query_text, invoke_response["body"]['generation'],False)
                return jsonify({"response": invoke_response["body"]['generation'],"prompt":prompt_text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            client.close()



@app.route('/ask/claude', methods=['POST'])
def ask_claudeSonnet():
    query_text = request.json.get('message')
    prefilter=request.json.get('selectedOption')
    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    conversation_id = request.json.get('sessionId')
    if query_text:
        try:
            previous_messages=get_conversation(client, conversation_id)
            if(prefilter=="dummy"):
                    req_body =  json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 3000,
                        "temperature": 0.7,
                        "messages": previous_messages + [
                        {"role": "user", "content": query_text}
                    ]
                        })
                    response = bedrock_runtime.invoke_model(modelId=model_id,body=req_body)
                    response= json.loads(response['body'].read())
                    update_conversation(client, conversation_id, query_text, response['content'][0]['text'],False)
                    return jsonify({"response":response['content'][0]['text']})
            else:
                metadata=prefilter.split("/")
                if(metadata[1]=='mxbai-embed-large'):
                    embeddingModel=metadata[1]
                    vector_index=app.config['MX_VECTOR']
                else:
                    embeddingModel="nomic-embed-text"
                    vector_index=app.config['NOM_VECTOR']
                embedding_model = OllamaEmbeddings(model=embeddingModel)
                prompt_text=generate_prompt(client,embedding_model,vector_index,query_text,prefilter,previous_messages)

                if(prompt_text is  None):
                    prompt_text="Error generating prompt retry"     
                req_body =  json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 3000,
                        "temperature": 0.7,
                        "messages": [
                            {"role": "user", "content": prompt_text}
                        ]
                        })
                response = bedrock_runtime.invoke_model(modelId=model_id,body=req_body)
                response= json.loads(response['body'].read())
                update_conversation(client, conversation_id, query_text, response['content'][0]['text'],False)
                return jsonify({"response": response['content'][0]['text'],"prompt":prompt_text})
            
            

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            client.close()

if __name__ == '__main__':
    app.run(debug=True)




    
    
                
    # retrieved_docs = vector_store.similarity_search_with_score(query=query_text, k=10,pre_filter={"object_ref":prefilter})





            # # Define the custom prompt
        # custom_prompt = PromptTemplate(
        #     input_variables=["prevMessages","context", "question"],
        #     template=PROMPT_TEMPLATE
        # )
                #     retriever =vector_store.as_retriever(
                #          search_kwargs={
                #                  'k': 10,
                #          'pre_filter': { 'object_ref': prefilter} 
                #      }
                #     )
                #     qa_chain = RetrievalQA.from_chain_type(
                #         retriever=retriever,
                #         llm=model,
                #         chain_type_kwargs={"prompt": custom_prompt,"verbose": True},
                #         return_source_documents=True
                        
                #     )
                #     response = qa_chain.invoke(query_text)
                #     combine_chain = qa_chain.combine_documents_chain
                #     prompt_template = combine_chain.llm_chain.prompt

                #    # Fetch the prompt variables
                #     context = "\n\n".join([doc.page_content for doc in response["source_documents"]])
                   

                #     # Construct the generated prompt
                #     generated_prompt = prompt_template.format(context=context, question=query_text)
                #     return jsonify({"response":response['result'],"prompt":generated_prompt})


        #   <!--Research Assistant tab -->
        #          <div class="tab-pane fade" id="researchassitant" role="tabpanel" aria-labelledby="researchassitant">
        #             <div class="chatbox">
        #                 <br> <h2>Under Development</h2>
        #                 <div id="chatContainer1" class="chat-container"></div>
        #                 <div class="input-group mt-3">
        #                 <br>
                       
        #                     <input type="text" id="userInput1" class="form-control" placeholder="Ask something..." aria-label="User Input" required>
        #                     <button class="btn btn-primary" id="sendMessageBtn1">Send</button>
        #                 </div>
        #                 <hr>
        #             </div>
        #             <div class="col-md-4" class="tab-pane fade"  role="tabpanel" aria-labelledby="chat-tab" >  
        #                 <div id="prompt-container1" class="prompts-view-box" style="height: 300px; width: 740px;overflow-y: auto; padding: 10px; border-radius: 10px;">
        #                 <h5 class="text-center mb-2">Prompt</h5>
        #             </div>
        #             </div>
        #         </div>


        #  <li class="nav-item" role="presentation">
        #         <a class="nav-link" id="stats" data-bs-toggle="tab" href="#researchassitant" role="tab" aria-controls="researchassitant" aria-selected="false">Agentic Research Assistant</a>
        #     </li>