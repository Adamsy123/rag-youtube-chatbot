import os
import re
import tempfile
import pathlib
import streamlit as st
from dotenv import load_dotenv
import yt_dlp

# Load environment variables (Local .env & Streamlit Cloud Secrets)
load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# LangChain Imports
from langchain_community.document_loaders import YoutubeLoader
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

def extract_clean_url(url: str) -> str:
    """Extracts a valid YouTube URL string, supporting standard links and Shorts."""
    url = url.strip()
    match = re.search(r"(?:v=|\/shorts\/|\/)([0-9A-Za-z_-]{11})", url)
    if match:
        video_id = match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

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
    clean_url = extract_clean_url(youtube_input)
    
    if not clean_url or not re.search(r"[0-9A-Za-z_-]{11}", clean_url):
        st.error("Please enter a valid YouTube video URL containing an 11-character video ID.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide an OpenAI API Key.")
    else:
        st.session_state.vector_store = None

        # 1. Download & Local Storage Handling
        with st.spinner("Downloading and processing media with yt-dlp..."):
            try:
                temp_dir = tempfile.mkdtemp()
                downloads_folder = str(pathlib.Path.home() / "Downloads")

                if download_format == "Audio (MP3)":
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                        }],
                        'quiet': True,
                    }
                    mime_type = "audio/mp3"
                else:
                    format_str = 'bestvideo+bestaudio/best'
                    if resolution == "720p":
                        format_str = 'bestvideo[height<=720]+bestaudio/best'
                    elif resolution == "360p":
                        format_str = 'bestvideo[height<=360]+bestaudio/best'

                    ydl_opts = {
                        'format': format_str,
                        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                        'merge_output_format': 'mp4',
                        'quiet': True,
                    }
                    mime_type = "video/mp4"

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(clean_url, download=True)
                    video_title = info.get('title', 'media_file')
                    
                # Find the downloaded file in the temporary directory
                downloaded_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
                if not downloaded_files:
                    raise FileNotFoundError("Downloaded file was not created.")
                
                final_buffer_path = downloaded_files[0]
                file_name = os.path.basename(final_buffer_path)

                # Save copy to system Downloads folder if write permissions exist
                try:
                    local_path = os.path.join(downloads_folder, file_name)
                    import shutil
                    shutil.copyfile(final_buffer_path, local_path)
                    st.info(f"💾 File saved to server downloads: `{local_path}`")
                except Exception:
                    pass

                with open(final_buffer_path, "rb") as file_bytes:
                    st.download_button(
                        label=f"💾 Download {file_name} via Browser",
                        data=file_bytes,
                        file_name=file_name,
                        mime=mime_type,
                        key="browser_download"
                    )

            except Exception as download_error:
                st.warning(f"Download warning: {str(download_error)}")

        # 2. RAG Pipeline Processing
        with st.spinner("Extracting transcript and building vector store..."):
            try:
                loader = YoutubeLoader.from_youtube_url(clean_url, add_video_info=False)
                docs = loader.load()

                if not docs or not docs[0].page_content.strip():
                    st.warning("⚠️ No transcript/captions found for this video. The chat feature requires videos with subtitles.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(docs)

                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    vector_store = FAISS.from_documents(chunks, embeddings)

                    st.session_state.vector_store = vector_store
                    st.success("Video indexed successfully! Ask your questions below.")

            except Exception as transcript_error:
                st.warning("⚠️ Could not load transcript. The video may not have closed captions enabled.")

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