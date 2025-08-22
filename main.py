import os
from pathlib import Path
from typing import List
from llama_index.core.node_parser import CodeSplitter
import ollama
import chromadb
from pprint import pprint

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


def print_all_file_contents(root_directory: str) -> None:
    """Recursively print the contents of every file under root_directory.

    Files are printed with clear begin/end markers using paths relative to the
    provided root. Text is decoded as UTF-8 with replacement for undecodable
    bytes to handle mixed encodings gracefully.
    """
    root_path = Path(root_directory).resolve()
    for file_path in iter_file_paths(str(root_path)):
        relative_path = file_path.relative_to(root_path)
        print(f"----- BEGIN FILE: {relative_path} -----")
        try:
            with open(file_path, mode="r", encoding="utf-8", errors="replace") as file_handle:
                for line in file_handle:
                    print(line, end="") # don't add a newline at the end of the line
        except Exception as error:  # noqa: BLE001 - bubble context to output
            # ^ the comment above is a linter surpression
            print(f"[Error reading {relative_path}]: {error}")
        print()
        print(f"----- END FILE: {relative_path} -----")
        print()


def create_file_embeddings(root_directory: str) -> List[str]:
    """Embed all files under the given root directory using ollama.

    The embeddings are returned as a list of strings.
    """
    file_embeddings = []
    code_splitter = CodeSplitter.from_defaults(language="python")

    root_path = Path(root_directory).resolve()
    for file_path in iter_file_paths(str(root_path)):
        file_contents = ""

        try:
            with open(file_path, mode="r", encoding="utf-8", errors="replace") as file_handle:
                for line in file_handle:
                    file_contents += line

            file_embeddings.extend(code_splitter.split_text(file_contents))
        except Exception as error:  # noqa: BLE001 - bubble context to output
            # ^ the comment above is a linter surpression
            print(f"[Error reading {file_path}]: {error}")

    return file_embeddings


def store_file_embeddings(client: chromadb.Client, file_embeddings: List[str]) -> None:
    """Store the file embeddings in the vector embedding database."""
    collection = client.get_or_create_collection(name="docs")

    # store each document in a vector embedding database
    for i, d in enumerate(file_embeddings):
        response = ollama.embed(model="mxbai-embed-large", input=d)
        response_embeddings = response["embeddings"]

        collection.add(
            ids=[str(i)],
            embeddings=response_embeddings,
            documents=[d]
        )


def retrieve_relevant_chunks(client: chromadb.Client, query: str) -> List[str]:
    """Retrieve the most relevant chunks from the vector embedding database."""
    collection = client.get_collection(name="docs")
    # use the SAME embedding model as used for indexing to avoid dimension mismatch
    response = ollama.embed(model="mxbai-embed-large", input=query)
    response_embeddings = response["embeddings"]

    results = collection.query(
        query_embeddings=response_embeddings,
        n_results=3 # 3 chunks to retrieve
    )
    return results

if __name__ == "__main__":
    import sys

    target_root = sys.argv[1] if len(sys.argv) > 1 else "."
    file_embeddings = create_file_embeddings(target_root)

    client = chromadb.Client()

    store_file_embeddings(client, file_embeddings)

    relevant_chunks = retrieve_relevant_chunks(client, "What is the main function of the program?")
    
    pprint(relevant_chunks["documents"])




