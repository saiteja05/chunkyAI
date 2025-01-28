import os
import json
from flask import Flask, render_template, request, jsonify
from flask import Flask, request, render_template, redirect, url_for, flash
import ollama
from werkzeug.utils import secure_filename
from config import Config
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
from langchain.tools import Tool
from langchain.chains import RetrievalQA


#for semantic chunking

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

#for  data in MongoDB
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_mongodb import MongoDBAtlasVectorSearch


from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import AzureChatOpenAI


PROMPT_TEMPLATE = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an     answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.
    You can suggest few external resources but do not provide the answer directly.
    Answer the following question based on the context:\n\nContext: {context}\n------------------\nQuestion: {question}
    """

# Define the custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an     answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.
    You can suggest few external resources but do not provide the answer directly.
    Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {question}
    """
)


# Initialize the Flask application
app = Flask(__name__,static_folder='static')
app.config['DEBUG'] = True
app.config['TESTING'] = False


#import config file
app.config.from_object(Config)
DB_NAME=app.config['DB_NAME']
COLLECTION_NAME=app.config['COLLECTION_NAME']

#open ai variables
openai_client = OpenAI(api_key=app.config['OPEN_AI_KEY'])
os.environ["OPENAI_API_KEY"]=app.config['OPEN_AI_KEY']


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



@app.route('/ask/ollama', methods=['POST'])
def ask_ollama():
    query_text = request.json.get('message')
    prefilter=request.json.get('selectedOption')
    isAgentic=request.json.get('isAgentic')
    
    llm=app.config['OLLAMA_MODEL']
    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    if query_text:
        try:
            if(prefilter=="dummy"):
                response = ollama.chat(model=llm, messages=[{'role': 'user', 'content':query_text },]) #role can be  'user', 'assistant', 'system' or 'tool'
                if response.done:
                 
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
              
                mongo_collection=client[DB_NAME][COLLECTION_NAME]
                
                vector_store = MongoDBAtlasVectorSearch(
                    collection=mongo_collection,
                    embedding=embedding_model,
                    index_name=vector_index,
                    relevance_score_fn="cosine",
                    text_key="page_content")

                if isAgentic is True:
                    retriever =vector_store.as_retriever(
                         search_kwargs={
                                 'k': 10,
                         'pre_filter': { 'object_ref': prefilter} 
                     }
                    )
                    qa_chain = RetrievalQA.from_chain_type(
                        retriever=retriever,
                        llm=model,
                        chain_type_kwargs={"prompt": custom_prompt,"verbose": True},
                        return_source_documents=True
                        
                    )
                    response = qa_chain.invoke(query_text)
                    combine_chain = qa_chain.combine_documents_chain
                    prompt_template = combine_chain.llm_chain.prompt

                   # Fetch the prompt variables
                    context = "\n\n".join([doc.page_content for doc in response["source_documents"]])
                   

                    # Construct the generated prompt
                    generated_prompt = prompt_template.format(context=context, question=query_text)
                    return jsonify({"response":response['result'],"prompt":generated_prompt})
               

              

                retrieved_docs = vector_store.similarity_search_with_score(query=query_text, k=10,pre_filter={"object_ref":prefilter})
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in retrieved_docs])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt_text = prompt_template.format(context=context_text, question=query_text) 
                  


                response=model.invoke(prompt_text)
                

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
    if isAgentic:
        print("isAgentic is true")
    llm = AzureChatOpenAI(
    azure_endpoint=app.config['AZURE_GPT35_ENDPOINT'],
    openai_api_key=app.config['AZURE_GPT35_KEY'],
    openai_api_version=app.config['AZURE_GPT35_API_VERSION'],   
    temperature=0.7,
    max_tokens=4096
    )

    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    # Ensure user_input is not empty
    if query_text:
        try:
            if(prefilter=="dummy"):
            # Call OpenAI's new chat-based API (using the correct model and chat completions)
                response = llm.invoke([
                        {"role": "system", "content": "you are a helpful chatbot"},
                        {"role": "user", "content": query_text},
                    ]
                )
               
                # print(response)
                return jsonify({"response":response.content})
            
            
            else:
                metadata=prefilter.split("/")
                if(metadata[1]=='mxbai-embed-large'):
                    embeddingModel=metadata[1]
                    vector_index=app.config['MX_VECTOR']
                else:
                    embeddingModel="nomic-embed-text"
                    vector_index=app.config['NOM_VECTOR']
                
                # print(embeddingModel+" is the embedding model and vector index is "+vector_index) 
                
                embedding_model = OllamaEmbeddings(model=embeddingModel)
                mongo_collection=client[DB_NAME][COLLECTION_NAME]
                vector_store = MongoDBAtlasVectorSearch(
                    collection=mongo_collection,
                    embedding=embedding_model,
                    index_name=vector_index,
                    relevance_score_fn="cosine",
                    text_key="page_content")
                documents=vector_store.similarity_search_with_score(query=query_text, k=5,pre_filter={"object_ref":prefilter})
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in documents])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(context=context_text, question=query_text)
                response = llm.invoke(input=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                    ])
                # print(jsonify({"response": response.choices[0].message.content,"prompt":prompt}))
                return jsonify({"response": response.content,"prompt":prompt})
            
            

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            client.close()

    return jsonify({"error": "No message provided"}), 400



@app.route('/ask/gpt-4.0', methods=['POST'])
def ask_gpt4o():
    query_text = request.json.get('message')
    prefilter=request.json.get('selectedOption')
    isAgentic=request.json.get('isAgentic')
    if isAgentic:
        print("isAgentic is true")
    client = MongoClient(app.config['MONGODB_URI'], server_api=ServerApi('1'))
    llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.7,
    max_tokens=4096
    )

    # Ensure user_input is not empty
    if query_text:
        try:
            if(prefilter=="dummy"):
            # Call OpenAI's new chat-based API (using the correct model and chat completions)

                response = llm.invoke([
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content":query_text}
                ])

                print(response)
               
                # print(response)
                return jsonify({"response":response.content})
            
            
            else:
                print(prefilter)
                metadata=prefilter.split("/")
                if(metadata[1]=='mxbai-embed-large'):
                    embeddingModel=metadata[1]
                    vector_index=app.config['MX_VECTOR']
                else:
                    embeddingModel="nomic-embed-text"
                    vector_index=app.config['NOM_VECTOR']
                
                # print(embeddingModel+" is the embedding model and vector index is "+vector_index) 
                
                embedding_model = OllamaEmbeddings(model=embeddingModel)
                mongo_collection=client[DB_NAME][COLLECTION_NAME]
                vector_store = MongoDBAtlasVectorSearch(
                    collection=mongo_collection,
                    embedding=embedding_model,
                    index_name=vector_index,
                    relevance_score_fn="cosine",
                    text_key="page_content")
                documents=vector_store.similarity_search_with_score(query=query_text, k=5,pre_filter={"object_ref":prefilter})
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in documents])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(context=context_text, question=query_text)
                response = llm.invoke(input=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                    ])
                # print(response)
                # print(jsonify({"response": response.choices[0].message.content,"prompt":prompt}))
                return jsonify({"response": response.content,"prompt":prompt})
            
            

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        finally:
            client.close()

    return jsonify({"error": "No message provided"}), 400



if __name__ == '__main__':
    app.run(debug=True)