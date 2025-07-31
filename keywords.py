import os
import glob
import re
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker


def extract_keywords_from_pdf(pdf_path, spell, stop_words, ps):

    text = extract_text(pdf_path)
    
    words = re.findall(r'\b\w+\b', text)
    words = [word.lower() for word in words]
    
    keywords = []
    for word in words:
        if word in stop_words:
            continue
        stemmed = ps.stem(word)
        if stemmed not in spell:
            corrected = spell.correction(stemmed)
            if corrected is None:
                corrected = stemmed
        else:
            corrected = stemmed
        keywords.append(corrected)
    
    return keywords

def write_keywords_to_file(keywords, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword in keywords:
            f.write(f"{keyword}\n")

def process_all_pdfs():
    pdf_dir = os.path.expanduser('~/Curie_Test/Tagged_Documents')
    pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in directory: {pdf_dir}")
        return
    
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    spell = SpellChecker()
    
    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        output_file = f"{base_name}_keywords.txt"
        
        print(f"Processing: {pdf_file}")
        keywords = extract_keywords_from_pdf(pdf_file, spell, stop_words, ps)
        print(f"Extracted {len(keywords)} keywords.")
        
        write_keywords_to_file(keywords, output_file)
        print(f"Keywords have been written to: {output_file}\n")

if __name__ == "__main__":
    process_all_pdfs()
