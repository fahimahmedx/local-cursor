import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from llama_index.core.node_parser import CodeSplitter
import chromadb
import ollama


def iter_file_paths(root_directory: str) -> List[Path]:
    """Returns a list of all file paths under the given root directory recursively.

    Directories are traversed in sorted order. Symlinked directories
    are not followed to avoid potential cycles.
    """
    file_paths = []
    ignore_directories = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
    }
    for current_dir, dir_names, file_names in os.walk(root_directory, followlinks=False):
        # Filter directories in-place to prevent os.walk from traversing them
        filtered_dir_names = []

        for dir in dir_names:
            if dir not in ignore_directories and not dir.startswith("."):
                filtered_dir_names.append(dir)

        dir_names[:] = filtered_dir_names # we use [:] to modify the list IN PLACE, which os.walk uses.

        dir_names.sort()
        file_names.sort()

        for file_name in file_names:
            candidate_path = Path(current_dir) / file_name # this concatenates the current directory with the file name
            if candidate_path.is_file() and candidate_path.suffix == ".py":
                file_paths.append(candidate_path)
    return file_paths

def chunk_python_code(text: str) -> List[str]:
    splitter = CodeSplitter.from_defaults(language="python")
    return splitter.split_text(text)

# creates file embeddings by chunking each file, and converting the chunks into embeddings
def chunk_files(root_directory: str) -> List[Dict[str, Any]]:
    chunks = []
    root_path = Path(root_directory).resolve()
    for file_path in iter_file_paths(str(root_path)):
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            parts = chunk_python_code(content)
            for idx, part in enumerate(parts):
                chunks.append({
                    "text": part,
                    "metadata": {
                        "source": str(file_path.relative_to(root_path)),
                        "abs_source": str(file_path),
                        "chunk_index": idx,
                    }
                })
        except Exception as e:
            print(f"[Error reading {file_path}]: {e}")
    return chunks

def codebase_hash(abs_path: str) -> str:
    return hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:12]

def get_client() -> chromadb.Client:
    # Persist to a stable local path so collections survive restarts
    base = os.path.expanduser("~/.codebase_indexing/chroma")
    os.makedirs(base, exist_ok=True)
    return chromadb.PersistentClient(path=base)

def get_collection(client: chromadb.Client, abs_root: str):
    name = f"docs_{codebase_hash(abs_root)}"
    return client.get_or_create_collection(name=name)

def index_codebase(root_directory: str) -> str:
    abs_root = str(Path(root_directory).resolve())
    client = get_client()
    collection = get_collection(client, abs_root)

    # If already indexed, skip. Use "Reindex" to force rebuild.
    count = 0
    try:
        count = collection.count()
    except Exception:
        pass

    if count > 0:
        return collection.name

    # rest of the function is to add the chunks to the vector database
    chunks = chunk_files(abs_root)
    if not chunks:
        return collection.name

    docs = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    ids = [f"{i}" for i in range(len(docs))]

    # vectorize the chunks
    resp = ollama.embed(model="mxbai-embed-large", input=docs)
    embeddings = resp["embeddings"]

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
    )
    return collection.name

# used if we want to switch to a different codebase.
def reindex_codebase(root_directory: str) -> str:
    abs_root = str(Path(root_directory).resolve())
    client = get_client()
    collection = get_collection(client, abs_root)
    # Delete whole collection, then recreate
    client.delete_collection(collection.name)
    client.create_collection(collection.name)
    return index_codebase(abs_root)

# retrieves the most relevant chunks / embeddings from the vector database
def retrieve_chunks(collection_name: str, query: str, k: int = 5):
    client = get_client()
    collection = client.get_collection(collection_name)

    # embed the query
    query_embeddings = ollama.embed(model="mxbai-embed-large", input=query)["embeddings"]
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=k
    )

    # for some reason the result is a nested list in the format of [["value1", "value2", ...]],
    # so we get the inner list.
    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    dists = results.get("distances", [[]])[0] if results.get("distances") else []
    return list(zip(docs, metas, dists))

def generate_answer(query: str, contexts: List[str]) -> str:
    prompt = (
        "You are a helpful codebase assistant. Use the provided context snippets "
        "to answer the user's question. If unsure, say so.\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Context snippets (may be partial):\n"
        + "\n\n".join([f"Snippet {i+1}:\n{c}" for i, c in enumerate(contexts)])
        + "\n\nAnswer:"
    )

    resp = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]

def parse_answer_with_thinking(answer: str) -> tuple[str, str]:
    """
    Parse an answer that may contain <think>...</think> tags.
    Returns (thinking_content, main_answer).
    If no thinking tags found, returns ("", original_answer).
    """
    import re
    
    # Look for <think>...</think> pattern
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, answer, re.DOTALL)
    
    if match:
        thinking_content = match.group(1).strip()
        # Remove the thinking tags and content from the main answer
        main_answer = re.sub(think_pattern, '', answer, flags=re.DOTALL).strip()
        return thinking_content, main_answer
    else:
        return "", answer

# Everything below this is the Streamlit UI, and was GPT generated.
# streamlit run app.py
st.set_page_config(page_title="Codebase Chat", layout="wide")

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "abs_root" not in st.session_state:
    st.session_state.abs_root = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("Codebase")
root_input = st.sidebar.text_input("Folder path", value="", placeholder="/path/to/your/codebase")
col1, col2 = st.sidebar.columns(2)
with col1:
    do_index = st.button("Index")
with col2:
    do_reindex = st.button("Reindex")

if do_index and root_input:
    abs_root = str(Path(root_input).resolve())
    if Path(abs_root).is_dir():
        name = index_codebase(abs_root)
        st.session_state.collection_name = name
        st.session_state.abs_root = abs_root
        st.sidebar.success(f"Indexed: {abs_root}")
    else:
        st.sidebar.error("Invalid folder path.")
if do_reindex and root_input:
    abs_root = str(Path(root_input).resolve())
    if Path(abs_root).is_dir():
        name = reindex_codebase(abs_root)
        st.session_state.collection_name = name
        st.session_state.abs_root = abs_root
        st.sidebar.success(f"Reindexed: {abs_root}")
    else:
        st.sidebar.error("Invalid folder path.")

st.title("Chat with your codebase")

if not st.session_state.collection_name:
    st.info("Enter a folder path in the sidebar and click Index to get started.")
else:
    # Show chat history
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask a question about the code...")
    if user_input:
        # Show user message
        st.session_state.messages.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant code snippets from vector database
        hits = retrieve_chunks(st.session_state.collection_name, user_input, k=5)
        contexts = [doc for (doc, meta, dist) in hits]

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                full_answer = generate_answer(user_input, contexts)
                thinking_content, main_answer = parse_answer_with_thinking(full_answer)
                
                # Show the main answer
                st.markdown(main_answer)
                
                # Show thinking in expander if present
                if thinking_content:
                    with st.expander("Thinking process"):
                        st.markdown(thinking_content)

                # Show sources
                with st.expander("Relevant code snippets"):
                    for i, (doc, meta, dist) in enumerate(hits, start=1):
                        src = meta.get("source") if isinstance(meta, dict) else meta
                        st.markdown(f"**{i}. {src}**  (distance: {dist:.4f})")
                        st.code(doc, language="python")

        st.session_state.messages.append(("assistant", main_answer))