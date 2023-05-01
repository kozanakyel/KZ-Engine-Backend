import os
import shutil


def remove_directory(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'The path is REMOVED: {path}')
    else:
        print(f'The path is not exist. {path}')