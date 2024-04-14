import streamlit as st
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceBgeEmbeddings
from pathlib import Path
import subprocess
import os

# Models data
models = [
    {"model_name": "qwen:0.5b-chat-v1.5-q4_0", "memory": 23, "precision": "f16", "model_type": "qwen", "accuracy": 38.62},
    {"model_name": "codellama:7b", "memory": 28, "precision": "bf16", "model_type": "llama", "accuracy": 39.81},
    {"model_name": "gemma:2b", "memory": 8, "precision": "f16", "model_type": "gemma", "accuracy": 46.4},
    {"model_name": "llama2:7b-chat", "memory": 8, "precision": "bf16", "model_type": "llama", "accuracy": 50.74},
    {"model_name": "llama2:13b", "memory": 64, "precision": "bf16", "model_type": "llama", "accuracy": 53.47},
    {"model_name": "mistral:7b-instruct", "memory": 28, "precision": "bf16", "model_type": "mistral", "accuracy": 54.96},
    {"model_name": "qwen:7b", "memory": 41, "precision": "bf16", "model_type": "qwen", "accuracy": 59.19},
    {"model_name": "codellama:70b-instruct", "memory": 73, "precision": "bf16", "model_type": "llama", "accuracy": 59.98},
    {"model_name": "phi", "memory": 80, "precision": "f16", "model_type": "phi-2", "accuracy": 61.33},
    {"model_name": "mistral:7b-instruct-v0.2-fp16", "memory": 64, "precision": "bf16", "model_type": "mistral", "accuracy": 65.71},
    {"model_name": "llama2:70b-chat", "memory": 64, "precision": "f16", "model_type": "llama", "accuracy": 67.87},
    {"model_name": "mixtral:8x7b", "memory": 64, "precision": "bf16", "model_type": "mixtral", "accuracy": 68.47},
    {"model_name": "mixtral:8x7b-instruct-v0.1-q4_0", "memory": 90, "precision": "bf16", "model_type": "mixtral", "accuracy": 72.7},
    {"model_name": "qwen:72b", "memory": 144, "precision": "f16", "model_type": "qwen", "accuracy": 73.6},
]


retriever_models = [
    "Salesforce/SFR-Embedding-Mistral",
    "GritLM/GritLM-7B",
    "intfloat/e5-mistral-7b-instruct",
    "GritLM/GritLM-8x7B",
    "WhereIsAI/UAE-Large-V1",
    "mixedbread-ai/mxbai-embed-large-v1",
    "avsolatorio/GIST-large-Embedding-v0",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-large",
    "llmrails/ember-v1"
]

def interpret_precision(precision_str):
    precision_mapping = {"bfloat16": 16, "float16": 16}
    return precision_mapping.get(precision_str.lower(), 0)

def find_suitable_models(gpu_memory):
    gpu_precisions = [gpu_memory / divisor for divisor in [1, 4, 8]]
    suitable_models = []
    for model in models:
        model_precision_value = interpret_precision(model["precision"])
        if any(gpu_precision >= model_precision_value for gpu_precision in gpu_precisions):
            suitable_models.append(model)
    suitable_models_sorted = sorted(suitable_models, key=lambda x: x["accuracy"], reverse=True)
    return suitable_models_sorted[:5] if len(suitable_models_sorted) > 5 else suitable_models_sorted

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Get the home directory of the current user
        home_dir = Path.home()
        save_path = home_dir / "Desktop" / "Research_Project" / "Uploaded_Documents"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_path = os.path.join(save_path, uploaded_file.name)
        with open(full_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return full_path
    return None

def launch_rag_solution(retriever_model, llm_model, website_url, document_path, url_depth):
    os.environ['RETRIEVER_MODEL'] = retriever_model
    os.environ['LLM_MODEL'] = llm_model
    os.environ['WEBSITE_URL'] = website_url
    os.environ['DOCUMENT_PATH'] = document_path
    os.environ['URL_DEPTH'] = str(url_depth) 

    subprocess.run(["streamlit", "run", "main.py"])

def main():
    st.title("Info Genie RAG Workbench 🖥️🧞")

    # GPU Memory Input
    gpu_memory = st.number_input("Enter GPU Memory Size in GB:", min_value=1)

    # Finding suitable models based on GPU memory
    suitable_models = find_suitable_models(gpu_memory)
    model_names = [model["model_name"] for model in suitable_models]
    
    if model_names:
        selected_model = st.radio("Select a Model:", model_names)
    else:
        st.error("No suitable models found. Consider getting a more powerful GPU.")

    
    selected_retriever = st.selectbox("Select a Retriever Model:", retriever_models)

    website_url = st.text_input("Enter Website URL (leave blank if uploading a document):", "")

    uploaded_file = st.file_uploader("Or Upload Document (PDF or DOCX):", type=['pdf', 'docx'])
    document_path = save_uploaded_file(uploaded_file) if uploaded_file is not None else ""

    url_depth = st.number_input("Enter URL Depth (1 to 99):", min_value=1, max_value=99, value=1)

    if st.button("Launch RAG Solution"):
        if website_url and document_path:
            st.error("Please either enter a URL or upload a document, not both.")
        elif not website_url and not document_path:
            st.error("Please either enter a URL or upload a document to proceed.")
        else:
            launch_rag_solution(selected_retriever, selected_model, website_url, document_path, url_depth)
            st.success("RAG Solution launched successfully!")

if __name__ == "__main__":
    main()
