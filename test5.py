import streamlit as st
import tiktoken
from loguru import logger
import os
import pandas as pd
from langchain.schema import Document
import faiss
import pickle
import zipfile

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 환경변수 설정
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "ATOZ"

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 미리 정의된 CSV 파일 경로
PRELOADED_CSV = "test.csv"
VECTORSTORE_FILE = "vectorstore.pkl"
VECTORSTORE_PARTS_PATTERN = "vectorstore.pkl.part"

def split_file(file_path, chunk_size=100 * 1024 * 1024):
    file_size = os.path.getsize(file_path)
    file_base_name = os.path.basename(file_path)
    
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while chunk_num * chunk_size < file_size:
            chunk_file_name = f"{file_base_name}.part{chunk_num:04d}.zip"
            with zipfile.ZipFile(chunk_file_name, 'w', zipfile.ZIP_DEFLATED) as chunk_zip:
                data = f.read(chunk_size)
                chunk_zip.writestr(file_base_name, data)
            chunk_num += 1

def merge_files(output_file, parts_pattern):
    part_files = sorted([f for f in os.listdir() if f.startswith(parts_pattern)])
    
    with open(output_file, 'wb') as output_f:
        for part_file in part_files:
            with zipfile.ZipFile(part_file, 'r') as part_zip:
                file_name = part_zip.namelist()[0]
                part_data = part_zip.read(file_name)
                output_f.write(part_data)

def configure_git_lfs():
    os.system('git config --global lfs.concurrenttransfers 3')

def main():
    st.set_page_config(page_title="기업문화 A to Z")
    st.title("무엇이든 물어보세요 A to Z")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        model_selection = st.selectbox(
            "Choose the language model",
            ("gpt-4o", "gpt-4-turbo-preview", "gpt-3.5-turbo"),
            key="model_selection"
        )

    # 애플리케이션 시작 시 임베딩 프로세스 실행
    if st.session_state.processComplete is None:
        with st.spinner("데이터를 처리하고 있습니다. 이 작업은 수 분이 소요되니 커피 한잔 하고 오세요. ☕️"):
            configure_git_lfs()
            if not os.path.exists(VECTORSTORE_FILE):
                merge_files(VECTORSTORE_FILE, VECTORSTORE_PARTS_PATTERN)
            vectorstore = load_vectorstore(VECTORSTORE_FILE)
            if not vectorstore:
                files_text = get_text([PRELOADED_CSV])
                text_chunks = get_text_chunks(files_text)
                vectorstore = get_vectorstore(text_chunks)
                save_vectorstore(vectorstore, VECTORSTORE_FILE)
                split_file(VECTORSTORE_FILE)
            openai_api_key = st.secrets["default"]["OPENAI_API_KEY"]
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key, st.session_state.model_selection)
            st.session_state.processComplete = True
        st.success("데이터 처리가 완료되었습니다!")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "안녕하세요, 기업문화 ChatBot Beta입니다. 😊"
                       "궁금한 것을 질문해 주세요. ❓"
                       "  \n아직은 걸음마 단계이니 잘 부탁드려요. 👣"
                       "저는 앞으로 더 성장할 거에요. 🌱"
        }]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("의료비, 건강검진 등 복리후생 관련 질문이 가능합니다."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("관련된 문서를 찾는 중입니다..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("기존 유사 Reference"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    documents = []
    for index, row in df.iterrows():
        content = ' '.join(row.values.astype(str))
        document = Document(page_content=content, metadata={'source': file_path, 'row': index})
        documents.append(document)

    return documents

def get_text(file_paths):
    doc_list = []
    for file_path in file_paths:
        doc_list.extend(load_csv(file_path))
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def save_vectorstore(vectorstore, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def get_conversation_chain(vectorstore, openai_api_key, model_selection):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
