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
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import VLLM, Ollama
from langchain import PromptTemplate


import chromadb

__all__ = ['Inference_Call', 'Url_Crawl', 'MultiCrawler', 'WebScraper']

pre_has_run = False

def pre():
    
    ##__init__ constructor ##
    base_url = os.getenv('WEBSITE_URL')
    #base_url= 'https://www.northwestern.edu/'
    if base_url:
        crawler = MultiCrawler([(base_url, int(os.getenv('URL_DEPTH')))])   #, ('https://bulletin.iit.edu/', 3) ]) #testing the top level domain url_crawling
        #crawler = MultiCrawler([os.getenv('WEBSITE_URL'), 3])
        crawler.crawl()
        crawler.process_urls()
        crawler.sort_all_urls()
        crawler.store_urls()

        ## WebScrapping.py ##
        current_directory = os.getcwd()
        # Create the path to the new directory
        new_directory = os.path.join(current_directory, 'chatbot')
        os.chdir(new_directory)

        scraper = WebScraper(file='urls_combined.csv')
    else:
        scraper = WebScraper()        
    scraper.scrape()  

    ## process_and_populate.py ##
    tp = TextProcessor()
    tp.process_text()
    tp.create_embeddings()


def infer():    
    
    inference = Inference_Call()
    
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

    current_directory = os.getcwd()
    new_directory = os.path.join(current_directory, 'chatbot')
    #scrape_dir = os.path.join(new_directory, 'scrapped_data')
    DB_DIR_bge_large = os.path.join(new_directory, 'vdb_persist_dir')

    #model bge_large
    #model_name = "BAAI/bge-large-en-v1.5"
    model_name = os.getenv("RETRIEVER_MODEL")
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    embedding_function_bge_large = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
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
    #llm0 = Ollama(model="mistral:v0.2", temperature=0)
    llm0 = Ollama(model=os.getenv("LLM_MODEL"), temperature=0)

    chain0 = inference.load_qa_chain(retriever1, llm0, prompt)
  

    #st.image("/home/ec2-user/ITMT597/misc/files/CoC_horiz_red_white_2019.png", width=500)
    st.title("Info Genie")
    st.markdown("### Ask me anything about your organization")

    # Display past conversations
    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        # Render each past conversation
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="https://i0.wp.com/leadershipfreak.blog/wp-content/uploads/2013/06/genie-lamp.jpg?resize=450%2C349"):
                    st.markdown(msg["content"])

    # React to user input
    if prompt := st.chat_input("Message to advisor"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Pass the user input as the query to the get_response function
        response = inference.get_response(prompt, chain0)
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="https://www.ncaa.com/sites/default/files/images/logos/schools/bgl/iit.svg"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    if 'pre_run' not in st.session_state:
        pre()
        st.session_state.pre_run = True

    infer()
