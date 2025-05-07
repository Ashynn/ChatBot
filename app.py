# Check 1
import os
import concurrent.futures
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.ensemble import EnsembleRetriever

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = " ",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def process_pdf(pdf):
    from unstructured.partition.pdf import partition_pdf
    os.environ["PATH"] += ";C:\\Program Files\\Tesseract-OCR"
    output_path = ".\\content\\"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.seek(0)
        tmp_file.write(pdf.read())
        tmp_file_path = tmp_file.name
    try:
        chunks = partition_pdf(
            filename=tmp_file_path,
            strategy="hi_res",
            infer_table_structure=False,
            tesseract_path="C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            extract_image_block_types=["Image"],
            image_output_dir_path=output_path,
            extract_image_block_to_payload=True,
            chunking_strategy=None,
        )
        return [
            chunk.metadata.image_base64
            for chunk in chunks if chunk.category == "Image"
        ]
    finally:
        os.remove(tmp_file_path)

def get_pdf_images(pdf_docs):
    image_b64 = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_docs)
        for result in results:
            image_b64.extend(result)
    return image_b64

def get_image_summaries(image_b64):
    prompt_template ="""Analyze this scientific figure/graph. Describe:
                    1. Figure type and purpose
                    2. Key elements (axis labels, units, data trends)
                    3. Significant patterns/outliers
                    4. Scientific context and interpretation
                    Avoid speculation. Base analysis on visual evidence."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            [
            {"type": "text", "text" : prompt_template},
            {
                "type" :"image_url",
                "image_url" : {"url": "data:image/jpeg;base64,{image}"},
            },
            ],
        )
    ])
    model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.3)
    chain = prompt | model | StrOutputParser()
    image_summaries = [] ###############################
    token_limit_reached = False
    try:
        image_summaries = chain.batch(image_b64)
    except Exception as e:
        if "token limit" in str(e).lower() or "context length" in str(e).lower():
            token_limit_reached = True
            st.sidebar.warning("Token limit reached. Processing images in batches.")
        else:
            st.sidebar.error(f"Unexpected error: {e}")   
    # image_summaries = chain.batch([{"image": img} for img in image_b64])
    # image_summaries = chain.batch(image_b64)
    return image_summaries if not token_limit_reached else image_summaries[:len(image_summaries)]

def get_vectorstore(text_summaries,image_summaries):
    embeddings = OpenAIEmbeddings()
    text_vectorstore = FAISS.from_texts(texts = text_summaries, embedding=embeddings)
    image_vectorstore = FAISS.from_texts(texts = image_summaries, embedding=embeddings)
    return text_vectorstore, image_vectorstore

def get_conversation_chain(text_vectorstore, image_vectorstore, memory):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )
    text_retriever = text_vectorstore.as_retriever()
    image_retriever = image_vectorstore.as_retriever()
    combined_retriever = EnsembleRetriever(retrievers=[text_retriever, image_retriever], weights=[0.5, 0.5])
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # custom_prompt = ChatPromptTemplate.from_messages([
    #         MessagesPlaceholder(variable_name="chat_history"),
    #         ("user","{question}")
    # ])
#     qa_template = ChatPromptTemplate.from_messages([
#         (
#             "system",
#                    "You are an expert research analyst specializing in summarizing and explaining scientific publications in detail. Your responses must be extremely helpful for understanding the research document as a whole. " 
#         "For every paper provided, ensure your answer includes the following elements:\n\n"
#         "1. **Comprehensive Summary:** Provide a clear and concise overview that covers the paper's purpose, methodology, results, and conclusions.\n"
#         "2. **Mathematical Insight:** Explain any mathematical equations or expressions, detailing their significance and how they contribute to the findings.\n"
#         "3. **Graphical Interpretation:** Analyze and interpret any graphs or figures, including key data trends, labels, and the overall message conveyed by the visuals.\n"
#         "4. **Critical Analysis:** Highlight the central contributions, discuss the implications, and note any limitations or potential future directions mentioned in the paper.\n\n"
#         "Your tone should be formal and precise yet accessible, striking a balance between academic rigor and readability. Avoid any speculation and base your responses strictly on the content provided. Always include appropriate citations or references to the document when necessary."
#         "Dont start with 'The paper is about' or 'Based on this paper' or 'Here is the summary'."        
#         ),
#         ("human", "Context:\n{context}\n\nQuestion: {question}")
# ,
#     ])
    qa_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research scientist specializing in the integration of advanced mathematical analysis and physical concepts. When summarizing and explaining research papers, your responses must reflect a deep, methodical understanding of both rigorous mathematical derivations and the fundamental physical principles underpinning the work. For every paper provided, ensure your answer includes the following elements:\n\n"
        "1. **Detailed Mathematical Analysis:** Elaborate on the key equations, expressions, and derivations found in the paper. Explain each symbol and step, discussing how these mathematical aspects contribute to the broader findings.\n\n"
        "2. **Physical Concepts and Principles:** Clearly unpack the physical theories, experimental designs, and underlying concepts. Relate the mathematical framework to observable phenomena and physical insights.\n\n"
        "3. **Comprehensive Scientific Breakdown:** Offer an integrated overview that covers the paper’s objectives, methodology, results, and conclusions, highlighting both the theoretical and empirical facets of the research.\n\n"
        "4. **Critical Evaluation:** Critically analyze the contributions of the work, discussing its implications in the context of current scientific understanding, noting limitations, and suggesting potential future directions. Your discussion should be strictly based on the content provided without unwarranted speculation.\n\n"
        "Maintain a formal, precise, and authoritative tone throughout your response. Avoid starting with phrases such as 'The paper is about' or 'Based on this paper,' ensuring that your reply flows naturally from your scientific analysis."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
    ])

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=combined_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_template},
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}",  message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}",  message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat bot for Scientific Papers", page_icon = ":books:")
    st.write(css, unsafe_allow_html=True) # Add the css to the page
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key= "chat_history", return_messages=True, output_key="answer")

    st.header("Chat bot for Scientific Papers :books:")
    user_question = st.text_input("Ask a question about your documents: ") # Search bar for the user to input
    if user_question:
        handle_user_input(user_question) # Call the function to handle the user input
    # st.write("Current memory buffer:", st.session_state.memory.buffer)

    st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True) # Add the user template to the page
    st.write(bot_template.replace("{{MSG}}", "Hello Seeker"), unsafe_allow_html=True) # Add the bot template to the page

    with st.sidebar:
        st.subheader("Upload your documents here")
        pdf_docs = st.file_uploader("Select the PDF files!", type=["pdf"], accept_multiple_files=True)

        if st.button("Analyze Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.text_chunks = get_text_chunks(raw_text)
                    st.session_state.pdf_docs = pdf_docs  # Store for later use
                    st.session_state.text_done = True
                    st.session_state.image_done = False
                    st.success("Text reading is done!")

        if st.session_state.get("text_done") and not st.session_state.get("image_done"):
            if st.button("Read Images"):
                with st.spinner("Reading the Images ..."):
                    images = get_pdf_images(st.session_state.pdf_docs)
                    st.session_state.image_summaries = get_image_summaries(images)
                    st.session_state.image_done = True
            if st.button("Skip Image Reading and Continue"):
                with st.spinner("Skipping image reading..."):
                    st.session_state.image_summaries = [""]
                    st.session_state.image_done = True

        # Final processing after both steps
        if st.session_state.get("text_done") and st.session_state.get("image_done"):
            with st.spinner("Finalizing processing..."):
                text_vectorstore, image_vectorstore = get_vectorstore(
                    st.session_state.text_chunks,
                    st.session_state.image_summaries
                )
                st.session_state.conversation = get_conversation_chain(
                    text_vectorstore, image_vectorstore, st.session_state.memory
                )
                st.success("Analysis complete! Ask your questions.")

if __name__ == "__main__":
    main()
