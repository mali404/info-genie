import os

import pandas as pd
import numpy as np
import textwrap

from transformers import BertTokenizer

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import VLLM, Ollama
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

from langchain.callbacks import StdOutCallbackHandler


import chromadb

class Inference_Call():
    
    def __init__(self, encoder_llm = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.encoder_llm = encoder_llm


    def load_qa_chain(self, retriever, llm, prompt):
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever, # here we are using the vectorstore as a retriever
            chain_type="stuff",
            return_source_documents=True, # including source documents in output
            chain_type_kwargs={'prompt': prompt, "callbacks" : [StdOutCallbackHandler()], 'document_separator':'\n\nOther context: '}
            )
    
    # Prettifying the response
    def get_response(self, query, chain):
        response = chain({'query': query})

        # Extracting only the LLM response
        wrapped_result = textwrap.fill(response['result'], width=100)
        response_text = "Advisor: " + wrapped_result

        return response_text


    

if __name__ == '__main__':
    inference = Inference_Call()
    llm0 = Ollama(model="mistral:v0.2", temperature=0)
    """llm1 = VLLM(model="mistralai/Mistral-7B-Instruct-v0.2",
                trust_remote_code=True, # mandatory for hf models 
                max_new_tokens=4096, 
                top_k=10, 
                top_p=0.95, 
                temperature=1.0, # 
                #tensor_parallel_size=4,# for distributed inference 
                gpu_memory_utilization=0.75, 
                vllm_kwargs = {"enable_lora":True, 'callbacks':[StdOutCallbackHandler()]}           
            )"""
    
    template1 = """<s> [INST] You are an academic advisor for Illinois Institute of Technology. \
    You have to answer the user's queries only on the basis context provided to you. \
    You have to use only relevant parts of the context to answer the user's queries. \
    The users are mostly students who want your advise on academic affairs. \
    If you don't know the answer, just say you don't know. Don't try to make up an answer.
    [/INST] </s>
    [INST] Question:{question}
    Context:{context}
    Answer:[/INST]
    """

    prompt = PromptTemplate.from_template(template1)

    home_directory = os.path.expanduser('~')
    new_directory = os.path.join(home_directory, 'chatbot')
    scrape_dir = os.path.join(new_directory, 'scrapped_data')
    DB_DIR_bge_large = os.path.join(new_directory, 'vdb_persist_dir')
    
    #model bge_large
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    embedding_function_bge_large = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        #cache_folder='/home/ec2-user/ITMT597/misc/models',
        encode_kwargs=encode_kwargs 
    )

    # initializing bge_large
    client_settings_bge_large = chromadb.config.Settings(
        is_persistent=True,
        persist_directory = DB_DIR_bge_large,
        anonymized_telemetry=False,
    )
    db_bge_large_combined_all = Chroma(
        collection_name="combined_all",
        persist_directory=DB_DIR_bge_large,
        client_settings=client_settings_bge_large,
        embedding_function=embedding_function_bge_large,
    )

    retriever1 = db_bge_large_combined_all.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "include_metadata": False}
    )

    chain0 = inference.load_qa_chain(retriever1, llm0, prompt)

    inference.get_response('I am an first year masters in data analytics and management. Which courses should I take?', chain0)