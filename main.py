import os
from pathlib import Path
from typing import List


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
            if candidate_path.is_file():
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


if __name__ == "__main__":
    import sys

    target_root = sys.argv[1] if len(sys.argv) > 1 else "."
    print_all_file_contents(target_root)


