import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

current_directory = os.getcwd()
new_directory = os.path.join(current_directory, 'chatbot')
scrape_dir = os.path.join(new_directory, 'scrapped_data')

        

class TextProcessor:
    def __init__(self, 
                 tokenizer_model='BAAI/bge-large-en-v1.5', 
                 chunk_size=480, 
                 chunk_overlap=90, 
                 threshold=35,
                 file_path=f'{scrape_dir}/combined_all.txt'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.bge_len,
            separators = ['\n\n', '\n', ' ', '']
        )
        self.threshold = threshold

    def bge_len(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def merge_rows(self, df):
        new_text = []
        new_tokens = []
        i = 0
        while i < len(df):
            merged_text, merged_tokens, i = self.merge_until_threshold(df, i, '', 0, self.threshold)
            new_text.append(merged_text)
            new_tokens.append(merged_tokens)
        new_df = pd.DataFrame({'text': new_text, 'NumTokens': new_tokens})
        return new_df

    def merge_until_threshold(self, df, i, current_text, current_tokens, threshold):
        if i < len(df) and current_tokens < threshold:
            current_text += '\n' + df.at[i, 'text'] if current_text else df.at[i, 'text']
            current_tokens += df.at[i, 'NumTokens']
            return self.merge_until_threshold(df, i+1, current_text, current_tokens, threshold)
        else:
            return current_text, current_tokens, i

    def process_text(self):
        with open(self.file_path, 'r') as file:
            data = file.read()
        combined_data = data.split('sos: ')
        self.chunks_combined_data_redone = []
        for doc in tqdm(combined_data):
            chunks = self.text_splitter.split_text(doc)
            for i, chunk in enumerate(chunks):
                self.chunks_combined_data_redone.append(chunk)
        #self.chunks_combined_data_redone = [str(element) for element in self.chunks_combined_data_redone]
        NumTokens = [self.bge_len(text) for text in self.chunks_combined_data_redone]
        df = pd.DataFrame({'text': self.chunks_combined_data_redone, 'NumTokens': NumTokens})
        df_merged = self.merge_rows(df)
        self.chunks_combined_all_merged = df_merged.text.tolist()
        
    def create_embeddings(self):
        persist_directory = os.path.join(new_directory, 'vdb_persist_dir')
        model_name = "BAAI/bge-large-en-v1.5"
        #model_name = self.retriever_model

        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs 
        )
        ##creating vector embeddings and storing them into separate collections based on split criteria
        Chroma.from_texts(texts=self.chunks_combined_all_merged, 
                          embedding=embedding_function, 
                          collection_name='combined_all', 
                          persist_directory=persist_directory)
        
if __name__ == '__main__':    
    tp = TextProcessor()
    tp.process_text()
    tp.create_embeddings()
