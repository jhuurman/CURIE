import os
import json
import pdfplumber
import time
import argparse
import gc
import spacy
import nltk
from nltk.stem import PorterStemmer
from collections import Counter

# Ensure that the necessary NLTK resources are available.
nltk.download('punkt', quiet=True)
porter = PorterStemmer()

# Define directories and global constants.
PDF_DIRECTORIES = [
    '/home/jno/Curie_Test/No_Tags_Documents'
]
MAX_TEXT_LENGTH = 10**9  # Maximum cumulative characters to process per PDF.

def init_nlp():
    """
    Initializes a lightweight NLP model for tokenization using spacy.
    GPU support is attempted via spacy.prefer_gpu(), but the blank model is used
    to limit overhead by excluding heavy components.
    """
    global nlp
    try:
        spacy.prefer_gpu()  # Try to use GPU if available.
    except Exception:
        pass  # Fallback to CPU if GPU is not available.
    # Using a blank model to get just the tokenizer for efficiency.
    nlp = spacy.blank("en")
    nlp.max_length = MAX_TEXT_LENGTH

def process_pdf(file_path):
    """
    Processes a single PDF file and computes the frequency dictionary of stemmed words.
    
    The function reads each page's text and divides it into chunks (if necessary) to
    reduce memory spikes. It then tokenizes, filters out non-alphabet tokens and stopwords,
    stems tokens, and updates the frequency counter on-the-fly.
    
    Args:
        file_path (str): The path to the PDF file.
    
    Returns:
        tuple: (file_path, frequency dictionary) where the dictionary maps stemmed tokens to counts.
    """
    frequency_counter = Counter()
    text_length = 0
    chunk_size = 100000  # Process text in chunks of 100,000 characters for efficiency.
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if not page_text:
                    continue

                # Limit processing to the remaining allowed text length.
                remaining_length = MAX_TEXT_LENGTH - text_length
                if remaining_length <= 0:
                    break
                if len(page_text) > remaining_length:
                    page_text = page_text[:remaining_length]

                # Process long texts in manageable chunks.
                for i in range(0, len(page_text), chunk_size):
                    chunk = page_text[i:i+chunk_size]
                    try:
                        doc = nlp(chunk)
                    except Exception as e:
                        print(f"Error processing text chunk in {file_path}: {e}")
                        continue
                    
                    # Update frequency counter directly without building a large token list.
                    for token in doc:
                        if token.is_alpha and not token.is_stop:
                            # Lowercase and stem the token for normalization.
                            stemmed_token = porter.stem(token.text.lower())
                            frequency_counter[stemmed_token] += 1
                    # Remove the doc object to free memory.
                    del doc

                text_length += len(page_text)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (file_path, {})

    return (file_path, dict(frequency_counter))

def get_pdf_paths(directories, max_files=None):
    """
    Collects PDF file paths from specified directories.
    
    Args:
        directories (list): A list of directories to search.
        max_files (int, optional): Maximum number of PDFs to collect. Defaults to None.
    
    Returns:
        list: List of full file paths to PDFs.
    """
    pdf_paths = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"WARNING: Directory not found: {directory}")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
                    if max_files and len(pdf_paths) >= max_files:
                        return pdf_paths
    return pdf_paths

def process_all_pdfs_sequential(directories, max_files=None, batch_size=1,
                                partial_dir="/home/jno/Curie_Test/Post_Processing/partials",
                                start_index=0):
    """
    Processes PDFs sequentially and saves partial results in batches.
    
    This function iterates over all identified PDF paths (starting from the user-specified
    index), processes each using `process_pdf`, and periodically writes the results to a JSON
    file to limit memory usage.
    
    Args:
        directories (list): Directories to search for PDF files.
        max_files (int, optional): Maximum number of files to process.
        batch_size (int): Number of processed PDFs per partial output file.
        partial_dir (str): Directory to save partial JSON outputs.
        start_index (int): Zero-indexed starting position in the list of PDF paths.
    """
    pdf_paths = get_pdf_paths(directories, max_files)
    
    # Ensure valid start_index and slice the list of PDF paths.
    if start_index < 0 or start_index >= len(pdf_paths):
        print("WARNING: Start index out of range. Starting from the beginning.")
        start_index = 0
    else:
        print(f"INFO: Starting processing from file number {start_index + 1}.")
        
    pdf_paths = pdf_paths[start_index:]
    total_files = len(pdf_paths)
    print(f"INFO: {total_files} PDF files will be processed.")

    os.makedirs(partial_dir, exist_ok=True)
    batch_data = []
    
    for count, path in enumerate(pdf_paths, start=1):
        result = process_pdf(path)
        batch_data.append({
            "file_path": result[0],
            "keyword_frequency": result[1]
        })
        
        # Save output every batch_size PDFs or upon completion.
        if count % batch_size == 0 or count == total_files:
            print(f"INFO: Processed {count}/{total_files} files. Saving partial output.")
            partial_file = os.path.join(partial_dir, f"processed_partial_{start_index + count}.json")
            with open(partial_file, "w") as f:
                json.dump(batch_data, f, indent=4)
            batch_data = []  # Reset the batch to free memory.
            gc.collect()    # Trigger garbage collection less frequently.
    print("INFO: Processing complete.")

def main():
    """
    The main function that parses command-line arguments, initializes the NLP pipeline,
    and begins processing.
    """
    parser = argparse.ArgumentParser(
        description="Efficient PDF keyword frequency extraction with minimal memory footprint."
    )
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of PDFs to process.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of PDFs processed per batch.")
    parser.add_argument("--start-file", type=int, default=372,
                        help="The file number (1-indexed) to start processing from.")
    args, unknown = parser.parse_known_args()
    
    # Convert the provided start file number to a zero-indexed start index.
    start_index = max(args.start_file - 1, 0)
    
    init_nlp()  # Single initialization for the NLP model.
    
    start_time = time.time()
    process_all_pdfs_sequential(PDF_DIRECTORIES, args.max_files, args.batch_size, start_index=start_index)
    print(f"INFO: Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
