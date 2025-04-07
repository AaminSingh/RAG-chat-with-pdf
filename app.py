import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


#SIdebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
       ##About
       This app is an LLM-powered built using:
      - [Streamlit](https://streamlit.io/)
      - [LangChain](https://python.langchain.com/)
      - [OpenAI](https://platform.openai.com/docs/models)LLM model
            
                ''') 
    add_vertical_space(5)
    st.write('Made with  dedication😊 [Team Debugger](https://youtube.com/@engineerprompt)')


def main():
  st.header("Chat with PDF 🗨️🗣️ ")

  load_dotenv()

  
  # uplaod a pdf file
  pdf = st.file_uploader("Upload a PDF file", type="pdf")
  #st.write(pdf.name)


  #st.write(pdf)
  if pdf is not None:
     pdf_reader = PdfReader(pdf)
     st.write(pdf_reader)
     
     text = ""
     for page in pdf_reader.pages:
         text += page.extract_text()    

         text_splitter = RecursiveCharacterTextSplitter(
             chunk_size = 1000,
             chunk_overlap = 200,
             length_function = len
         )
         chunks = text_splitter.split_text(text=text)

         #embeddings
         embeddings = OpenAIEmbeddings()
         VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
         store_name = pdf.name[:-4]


         with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(VectorStore, f)


         # st.write(chunks)


         # st.write(text)

 
if __name__ == '__main__':
     main()                       


     