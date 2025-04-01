# --- Print File Names to debug error "No such file or directory: {file or directory}" ---
import os 

def find_names(*args):
    for file_path in args:
        if os.path.exists(file_path):
            print(f"File path found in directory: {file_path}")
        else:
            print(f"File path not found in directory: {file_path}")

find_names('data/failure_reasons.csv')