import os
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
from llama_index.core.node_parser import CodeSplitter
import chromadb
import ollama
import subprocess
import platform


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

# rather than using generate_answer(), we use this so the user can SEE the thinking process as it happens.
# While I was building this app out, I realized the simple effect of streaming the thinking process
# makes the app feel less slow (giving a better user experience), because the user sees something is happening
# as opposed to it being frozen and showing the words "Thinking".
def generate_answer_streaming(query: str, contexts: List[str]):
    """
    Generator function that yields chunks of the response as they arrive.
    Returns tuples of (chunk_text, is_thinking, thinking_content, main_content)
    """
    prompt = (
        "You are a helpful codebase assistant. Use the provided context snippets "
        "to answer the user's question. If unsure, say so.\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Context snippets (may be partial):\n"
        + "\n\n".join([f"Snippet {i+1}:\n{c}" for i, c in enumerate(contexts)])
        + "\n\nAnswer:"
    )

    full_response = ""
    thinking_content = ""
    main_content = ""
    in_thinking = False
    
    try:
        stream = ollama.chat(
            model="deepseek-r1:1.5b", 
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        for chunk in stream:
            if chunk and 'message' in chunk and 'content' in chunk['message']:
                chunk_text = chunk['message']['content']
                full_response += chunk_text

                # In the logic below, we're seperating the response (that we're streaming to the user) into 2 parts: thinking and main content.
                # We're doing this by checking for <think> and </think> tags.
                # If we find <think>, then we know we're in a thinking block and can add the content AFTER <think> to the thinking content, and before <think> to the main content.
                # we the continue streaming content to the thinking_content, until we hit </think>.
                # If we find </think>, we know we're at the end of the thinking block, and all content after </think> is main content.
    
                if '<think>' in chunk_text:
                    in_thinking = True
                    # content before <think> is main content
                    before_think = chunk_text.split('<think>')[0]
                    if before_think:
                        main_content += before_think
                        yield (before_think, False, thinking_content, main_content)
                    
                    # thinking content starts here
                    after_think = chunk_text.split('<think>', 1)[1] if '<think>' in chunk_text else ""
                    # if the <think> and </think> are in the same chunk, we can complete the thinking block in one chunk
                    if '</think>' in after_think:
                        # complete thinking block in one chunk
                        think_part = after_think.split('</think>')[0]
                        thinking_content += think_part
                        yield (think_part, True, thinking_content, main_content)
                        
                        # main content after </think>
                        after_think_end = after_think.split('</think>', 1)[1]
                        main_content += after_think_end
                        if after_think_end:
                            yield (after_think_end, False, thinking_content, main_content)
                        in_thinking = False
                    else:
                        thinking_content += after_think
                        if after_think:
                            yield (after_think, True, thinking_content, main_content)
                        
                elif '</think>' in chunk_text and in_thinking:
                    # chunk contains end of thinking block
                    before_end = chunk_text.split('</think>')[0]
                    thinking_content += before_end
                    if before_end:
                        yield (before_end, True, thinking_content, main_content)
                    
                    # main content after </think>
                    after_end = chunk_text.split('</think>', 1)[1]
                    main_content += after_end
                    if after_end:
                        yield (after_end, False, thinking_content, main_content)
                    in_thinking = False
                    
                elif in_thinking:
                    # chunk is inside the thinking block
                    thinking_content += chunk_text
                    yield (chunk_text, True, thinking_content, main_content)
                    
                else:
                    # chunk is after the thinking block
                    main_content += chunk_text
                    yield (chunk_text, False, thinking_content, main_content)
                    
    except Exception as e:
        yield (f"Error during streaming: {e}", False, thinking_content, main_content)
        
    # return final parsed content. we use yield here because we're streaming the response.
    final_thinking, final_main = parse_answer_with_thinking(full_response)
    yield ("", False, final_thinking or thinking_content, final_main or main_content)

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

def select_folder_macos():
    """
    Opens a native macOS folder selection dialog using AppleScript.
    Returns the selected folder path or None if cancelled.
    """
    try:
        # AppleScript to open folder picker
        applescript = '''
        tell application "System Events"
            activate
            set folderPath to choose folder with prompt "Select your codebase folder:"
            return POSIX path of folderPath
        end tell
        '''
        
        # Run AppleScript via osascript command
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for user to select
        )
        
        if result.returncode == 0 and result.stdout:
            # Remove trailing newline and return the path
            return result.stdout.strip()
        return None
        
    except subprocess.TimeoutExpired:
        st.warning("Folder selection timed out.")
        return None
    except Exception as e:
        st.error(f"Error opening folder dialog: {e}")
        return None

# Everything below this is the Streamlit UI, and was GPT generated.
# streamlit run app.py
st.set_page_config(page_title="Codebase Chat", layout="wide")

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "abs_root" not in st.session_state:
    st.session_state.abs_root = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_folder" not in st.session_state:
    st.session_state.selected_folder = None

st.sidebar.title("Codebase")

# Check if we're on macOS
is_macos = platform.system() == "Darwin"

if is_macos:
    # Button to open native macOS folder picker
    if st.sidebar.button("Browse for Folder", use_container_width=True, help="Click to open native folder picker"):
        with st.spinner("Opening folder picker..."):
            selected = select_folder_macos()
            if selected:
                st.session_state.selected_folder = selected
                st.rerun()  # Rerun to update the UI with the new selection
else:
    st.sidebar.error("⚠️ This app currently only supports macOS folder selection.")

# Display currently selected folder
if st.session_state.selected_folder:
    st.sidebar.success(f"Selected: {st.session_state.selected_folder}")
else:
    st.sidebar.warning("No folder selected. Please choose a codebase.")

# Index and Reindex buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    do_index = st.button("Index", disabled=not st.session_state.selected_folder)
with col2:
    do_reindex = st.button("Reindex", disabled=not st.session_state.selected_folder)

if do_index and st.session_state.selected_folder:
    abs_root = str(Path(st.session_state.selected_folder).resolve())
    if Path(abs_root).is_dir():
        name = index_codebase(abs_root)
        st.session_state.collection_name = name
        st.session_state.abs_root = abs_root
        st.sidebar.success(f"Indexed: {abs_root}")
    else:
        st.sidebar.error("Invalid folder path.")
        
if do_reindex and st.session_state.selected_folder:
    abs_root = str(Path(st.session_state.selected_folder).resolve())
    if Path(abs_root).is_dir():
        name = reindex_codebase(abs_root)
        st.session_state.collection_name = name
        st.session_state.abs_root = abs_root
        st.sidebar.success(f"Reindexed: {abs_root}")
    else:
        st.sidebar.error("Invalid folder path.")

st.title("Chat with your codebase")

if not st.session_state.collection_name:
    st.info("Select a folder in the sidebar and click Index to get started.")
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

        # Generate answer with streaming
        with st.chat_message("assistant"):
            # show code context in dropdown menu
            with st.expander("Relevant code snippets"):
                for i, (doc, meta, dist) in enumerate(hits, start=1):
                    src = meta.get("source") if isinstance(meta, dict) else meta
                    st.markdown(f"**{i}. {src}**  (distance: {dist:.4f})")
                    st.code(doc, language="python")
            
            # Create containers for different parts of the response
            thinking_container = st.empty()
            main_answer_container = st.empty()
            
            thinking_content = ""
            main_answer = ""
            thinking_visible = False
            
            # Stream the response
            for chunk_text, is_thinking, full_thinking, full_main in generate_answer_streaming(user_input, contexts):
                if is_thinking:
                    thinking_content = full_thinking
                    if not thinking_visible:
                        thinking_visible = True
                    # Update thinking display in real-time
                    with thinking_container.container():
                        # expanded=True means the thinking process is shown by default
                        with st.expander("Thinking process", expanded=True):
                            st.markdown(thinking_content + "▋")  # Add cursor to show it's streaming
                else:
                    main_answer = full_main
                    # Update main answer display
                    if main_answer.strip():
                        main_answer_container.markdown(main_answer + "▋")  # Add cursor to show it's streaming
                
                # Small delay to make streaming visible (and not too fast)
                time.sleep(0.05)
            
            # Final update without the ▋ cursor
            if thinking_content:
                with thinking_container.container():
                    # the final thinking process is optional to view, so we don't expand the dropdown menu default
                    with st.expander("Thinking process"):
                        st.markdown(thinking_content)
            else:
                thinking_container.empty()
                
            if main_answer:
                main_answer_container.markdown(main_answer)
            else:
                main_answer_container.markdown("I couldn't generate a response. Please try again.")

        st.session_state.messages.append(("assistant", main_answer))