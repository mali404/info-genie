import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, tokenizer_model='BAAI/bge-large-en-v1.5', chunk_size=480, chunk_overlap=90, threshold=35):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
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

    def process_text(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        combined_data = data.split('sos: ')
        chunks_combined_data_redone = []
        for doc in tqdm(combined_data):
            chunks = self.text_splitter.split_text(doc)
            for i, chunk in enumerate(chunks):
                chunks_combined_data_redone.append(chunk)
        df = pd.DataFrame({'text': chunks_combined_data_redone, 'NumTokens': self.bge_len(chunks_combined_data_redone)})
        df_merged = self.merge_rows(df)
        chunks_combined_all_merged = df_merged.text.tolist()
        return chunks_combined_all_merged