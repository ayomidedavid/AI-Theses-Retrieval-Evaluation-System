import os
import glob
from PyPDF2 import PdfReader

PDF_DIR = "downloaded_pdfs"

import re

def get_safe_filename(title):
    """Clean text and remove characters that aren't allowed in Windows filenames"""
    # Remove any leading spaces
    title = title.lstrip()
    
    # Remove leading numbers
    title = re.sub(r'^[\d\s]+', '', title)
    
    # Remove leading small letter roman numerals (i, ii, iii, iv, v, vi, etc.) 
    # The \b ensures it's a standalone word and we don't accidentally chop letters off valid words
    title = re.sub(r'^[ivx]+\b\s*', '', title)
    
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    safe_title = "".join(c for c in title if c in valid_chars)
    
    # Strip any dangling spaces or digits that were exposed after cleanup
    safe_title = re.sub(r'^[\d\s]+', '', safe_title)
    
    return safe_title.strip()

def rename_pdfs_by_title():
    if not os.path.exists(PDF_DIR):
        print(f"Directory '{PDF_DIR}' not found.")
        return

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    print(f"Found {len(pdf_files)} PDFs. Scanning for titles...")
    
    renamed_count = 0
    
    for filepath in pdf_files:
        try:
            reader = PdfReader(filepath)
            # Try to extract title from the first page text
            text = reader.pages[0].extract_text()
            title = None
            
            if text:
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().upper() == "BY":
                        # Join all non-empty lines before 'BY'
                        title = " ".join([l.strip() for l in lines[:i] if l.strip()])
                        break
            
            # Fallback to metadata if 'BY' extraction failed
            if not title:
                meta = reader.metadata
                title = meta.title if meta else None
            
            # If still no valid title, skip it
            if not title or title.strip() == "" or "Microsoft Word" in title:
                print(f"[-] Skipping (Could not extract valid title): {os.path.basename(filepath)}")
                continue
                
            safe_title = get_safe_filename(title)
            
            if not safe_title:
                print(f"[-] Skipping (Title could not be converted to filename): {os.path.basename(filepath)}")
                continue

            new_filename = f"{safe_title}.pdf"
            new_filepath = os.path.join(PDF_DIR, new_filename)
            
            # Don't rename if it already has the correct name or the new name exists
            if filepath == new_filepath:
                continue
                
            if os.path.exists(new_filepath):
                print(f"[-] Name collision, skipping: {new_filename} already exists.")
                continue
                
            os.rename(filepath, new_filepath)
            print(f"[+] Renamed: '{os.path.basename(filepath)}' -> '{new_filename}'")
            renamed_count += 1
            
        except Exception as e:
            print(f"[!] Error processing {os.path.basename(filepath)}: {e}")

if __name__ == "__main__":
    rename_pdfs_by_title()
    print(f"\nDone! Successfully renamed {renamed_count} files.")
