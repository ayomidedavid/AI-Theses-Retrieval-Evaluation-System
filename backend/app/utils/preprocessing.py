import os
import re
from pypdf import PdfReader
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources silently if not installed
for resource, resource_type in [('stopwords', 'corpora'), ('wordnet', 'corpora'), ('punkt', 'tokenizers'), ('punkt_tab', 'tokenizers')]:
    try:
        nltk.data.find(f'{resource_type}/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Max characters to process per document (prevents hang on huge PDFs)
MAX_TEXT_CHARS = 50000


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['figure', 'et', 'al', 'section', 'table', 'chapter', 'however', 'therefore'])
        self.lemmatizer = WordNetLemmatizer()

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from first few pages of a PDF to avoid hangs on massive files."""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            # Only process first 10 pages for speed/metadata
            num_pages = min(len(reader.pages), 10)
            for i in range(num_pages):
                extracted = reader.pages[i].extract_text()
                if extracted:
                    text += extracted + "\n"
                if len(text) >= MAX_TEXT_CHARS:
                    break
            text = text[:MAX_TEXT_CHARS]
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return None
        return text

    def extract_year(self, text):
        """Attempts to extract a plausible publication year (1950 - 2030) from the document text."""
        if not text:
            return None
        import datetime
        matches = re.findall(r'\b(?:19|20)\d{2}\b', text)
        valid_years = [int(m) for m in matches if 1950 <= int(m) <= 2030]
        if valid_years:
            current_year = datetime.datetime.now().year
            valid_years = [y for y in valid_years if y <= current_year + 1]
            if valid_years:
                return max(valid_years)
        return None

    def clean_text(self, text):
        """Removes HTML, special chars, extra whitespace; lowercases."""
        soup = BeautifulSoup(text, "html.parser")
        clean_txt = soup.get_text()
        clean_txt = re.sub(r'[^a-zA-Z\s]', ' ', clean_txt)
        clean_txt = re.sub(r'\s+', ' ', clean_txt).strip().lower()
        return clean_txt

    def nlp_process(self, text):
        """Stop-word removal and Lemmatization using simple split (safe for large texts)."""
        # Use simple split instead of NLTK word_tokenize to avoid hanging on huge texts
        tokens = text.split()
        processed_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]
        return " ".join(processed_tokens)

    def process_document(self, pdf_path):
        """Full Pipeline: Read -> Clean -> NLP -> return processed text"""
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            return ""
        cleaned = self.clean_text(raw_text)
        return self.nlp_process(cleaned)
