import os
import re
import tempfile
import pathlib
import streamlit as st
from dotenv import load_dotenv
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Load environment variables (Local .env & Streamlit Cloud Secrets)
load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG YouTube Chatbot", page_icon="🎥", layout="wide")
st.title("🎥🤖 RAG YouTube Chatbot + Media Downloader")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Enter a valid YouTube URL.\n2. Choose format/quality.\n3. Click **Process & Prepare**.\n4. Save locally or via browser, then chat below!")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def extract_video_id(url: str) -> str:
    """Extracts 11-character YouTube video ID."""
    match = re.search(r"(?:v=|\/shorts\/|\/)([0-9A-Za-z_-]{11})", url.strip())
    return match.group(1) if match else None

# Video URL Input
youtube_input = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)
with col1:
    download_format = st.radio("Select Format:", ["Video (MP4)", "Audio (MP3)"])

with col2:
    if download_format == "Video (MP4)":
        resolution = st.selectbox("Select Resolution:", ["Best Available", "720p", "360p"])
    else:
        resolution = "Audio Only"

if st.button("Process & Prepare"):
    video_id = extract_video_id(youtube_input)
    clean_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
    
    if not video_id:
        st.error("Please enter a valid YouTube video URL containing an 11-character video ID.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide an OpenAI API Key.")
    else:
        st.session_state.vector_store = None

        # 1. Download Handling
        with st.spinner("Downloading media..."):
            try:
                temp_dir = tempfile.mkdtemp()
                output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")

                ydl_opts = {
                    'outtmpl': output_template,
                    'quiet': True,
                    'no_warnings': True,
                    'format': 'bestvideo+bestaudio/best' if download_format == "Video (MP4)" else 'bestaudio/best',
                    'ignoreerrors': True,
                }

                if download_format == "Audio (MP3)":
                    ydl_opts['postprocessors'] = [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }]

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(clean_url, download=True)
                    
                downloaded_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if not f.endswith('.part')]
                
                if downloaded_files:
                    final_buffer_path = downloaded_files[0]
                    file_name = os.path.basename(final_buffer_path)
                    mime_type = "audio/mp3" if download_format == "Audio (MP3)" else "video/mp4"

                    with open(final_buffer_path, "rb") as file_bytes:
                        st.download_button(
                            label=f"💾 Save {file_name} via Browser",
                            data=file_bytes,
                            file_name=file_name,
                            mime=mime_type,
                            key="browser_download"
                        )
                    st.success("Media prepared for browser download!")
                else:
                    st.warning("⚠️ YouTube limited media download for this video on cloud servers.")

            except Exception as download_error:
                st.warning(f"Download warning: {str(download_error)}")

        # 2. Direct Transcript Extraction & RAG Pipeline
        with st.spinner("Extracting transcript and indexing..."):
            try:
                # Fetch transcript directly via YoutubeTranscriptApi
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'a.en'])
                full_text = " ".join([item['text'] for item in transcript_list])

                if not full_text.strip():
                    st.warning("⚠️ Transcript is empty.")
                else:
                    docs = [Document(page_content=full_text, metadata={"source": clean_url})]
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(docs)

                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    vector_store = FAISS.from_documents(chunks, embeddings)

                    st.session_state.vector_store = vector_store
                    st.success("Video indexed successfully! Ask your questions below.")

            except (TranscriptsDisabled, NoTranscriptFound):
                st.warning("⚠️ Captions/Subtitles are not enabled for this video.")
            except Exception as transcript_error:
                st.warning("⚠️ Could not load transcript due to cloud IP limitations.")

# 3. Interactive Chat Interface
if st.session_state.vector_store is not None:
    st.markdown("---")
    st.subheader("💬 Ask Questions About the Video")

    user_query = st.text_input("Your Question:")

    if user_query:
        with st.spinner("Generating answer..."):
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

            system_prompt = (
                "You are an assistant for question-answering tasks on YouTube transcript context.\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            qa_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, qa_chain)

            response = rag_chain.invoke({"input": user_query})
            st.write(response["answer"])