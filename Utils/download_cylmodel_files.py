"""
Download cylindrical model files from TERN data repository
"""

import os
import requests
from urllib.parse import urljoin
from html.parser import HTMLParser
import sys

# Configuration
BASE_URL = "https://data.tern.org.au/rs/public/data/field_validation/vegetation_structure/14501596_3675889/20120504/models/cylmodel/"
OUTPUT_FOLDER = "downloaded_cylmodels"

class LinkExtractor(HTMLParser):
    """Extract links from HTML"""
    def __init__(self):
        super().__init__()
        self.links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href' and value.endswith('.txt'):
                    self.links.append(value)

def get_file_list(url):
    """Fetch the directory listing and extract txt file names"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        parser = LinkExtractor()
        parser.feed(response.text)
        
        return parser.links
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return []

def download_files(file_list, base_url, output_folder):
    """Download all files from the list"""
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    total_files = len(file_list)
    print(f"Found {total_files} files to download")
    print(f"Downloading to: {os.path.abspath(output_folder)}")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for idx, filename in enumerate(file_list, 1):
        file_url = urljoin(base_url, filename)
        file_path = os.path.join(output_folder, filename)
        
        try:
            print(f"[{idx}/{total_files}] Downloading {filename}...", end=" ", flush=True)
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content) / 1024  # Convert to KB
            print(f"OK ({file_size:.1f} KB)")
            successful += 1
            
        except Exception as e:
            print(f"FAILED ({str(e)})")
            failed += 1
    
    print("-" * 60)
    print(f"\nDownload Complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {total_files}")

def main():
    print("TERN Cylindrical Model File Downloader")
    print("=" * 60)
    print(f"Source: {BASE_URL}")
    print()
    
    # Get file list
    print("Fetching file list...")
    file_list = get_file_list(BASE_URL)
    
    if not file_list:
        print("No files found or error occurred")
        return
    
    # Download files
    download_files(file_list, BASE_URL, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
