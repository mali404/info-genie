from inference import Inference_Call
from retriever_ops.process_and_populate import TextProcessor
from web_scrapping import Url_Crawl, MultiCrawler
from web_scrapping.webscrapping import WebScraper

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import VLLM, Ollama
from langchain import PromptTemplate


import chromadb

__all__ = ['Inference_Call', 'Url_Crawl', 'MultiCrawler', 'WebScraper']
def main():    
    ##__init__ constructor ##
    base_url= 'https://www.northwestern.edu/'
    if base_url:
        crawler = MultiCrawler([(base_url, 3)])   #, ('https://bulletin.iit.edu/', 3) ]) #testing the top level domain url_crawling
        #crawler = MultiCrawler([os.getenv('WEBSITE_URL'), 3])
        crawler.crawl()
        crawler.process_urls()
        crawler.sort_all_urls()
        crawler.store_urls()

        ## WebScrapping.py ##
        home_directory = os.path.expanduser('~')
        # Create the path to the new directory
        new_directory = os.path.join(home_directory, 'chatbot')
        #new_directory = r'C:\Study\Conferences\hackathon\files'
        os.chdir(new_directory)

        scraper = WebScraper(file='urls_combined.csv')
    else:
        scraper = WebScraper()        
    scraper.scrape()  

    ## process_and_populate.py ##
    tp = TextProcessor()
    tp.process_text()
    tp.create_embeddings()

    ## main.py ##
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
    #scrape_dir = os.path.join(new_directory, 'scrapped_data')
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
        #client_settings=client_settings_bge_large,
        embedding_function=embedding_function_bge_large,
    )

    retriever1 = db_bge_large_combined_all.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "include_metadata": False}
    )

    chain0 = inference.load_qa_chain(retriever1, llm0, prompt)

    inference.get_response('What is the purpose of MLH?', chain0)

if __name__ == '__main__':
    main()
