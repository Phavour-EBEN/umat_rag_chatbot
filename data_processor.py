import json
import re
from typing import List, Dict

class DataProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        Initialize the data processor
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_chunks = []
    
    def load_scraped_data(self, filename='umat_data.json'):
        """Load scraped data from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Loaded {len(data)} pages from {filename}")
            return data
        except FileNotFoundError:
            print(f"✗ File {filename} not found!")
            return []
    
    def clean_text(self, text):
        """Additional cleaning for text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text.strip()
    
    def split_into_sentences(self, text):
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text, metadata):
        """
        Split text into overlapping chunks
        
        Args:
            text: The text to chunk
            metadata: Additional info (url, title) to attach to chunks
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk_size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source_url': metadata['url'],
                    'source_title': metadata['title'],
                    'chunk_id': len(chunks)
                })
                
                # Start new chunk with overlap
                # Keep last part of previous chunk for context
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source_url': metadata['url'],
                'source_title': metadata['title'],
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def process_all_data(self, scraped_data):
        """Process all scraped pages into chunks"""
        print("\n" + "="*50)
        print("PROCESSING DATA INTO CHUNKS")
        print("="*50)
        
        all_chunks = []
        
        for i, page in enumerate(scraped_data, 1):
            print(f"Processing page {i}/{len(scraped_data)}: {page['title']}")
            
            # Clean the content
            cleaned_text = self.clean_text(page['content'])
            
            # Create metadata
            metadata = {
                'url': page['url'],
                'title': page['title']
            }
            
            # Split into chunks
            chunks = self.create_chunks(cleaned_text, metadata)
            all_chunks.extend(chunks)
            
            print(f"  ✓ Created {len(chunks)} chunks")
        
        self.processed_chunks = all_chunks
        print(f"\n✓ Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_chunks(self, filename='processed_chunks.json'):
        """Save processed chunks to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.processed_chunks, f, indent=2, ensure_ascii=False)
        print(f"✓ Chunks saved to {filename}")
    
    def get_statistics(self):
        """Get statistics about processed chunks"""
        if not self.processed_chunks:
            return "No chunks processed yet!"
        
        total_chunks = len(self.processed_chunks)
        avg_length = sum(len(c['text']) for c in self.processed_chunks) / total_chunks
        unique_sources = len(set(c['source_url'] for c in self.processed_chunks))
        
        stats = {
            'total_chunks': total_chunks,
            'average_chunk_length': round(avg_length, 2),
            'unique_sources': unique_sources,
            'chunk_size_setting': self.chunk_size,
            'overlap_setting': self.chunk_overlap
        }
        
        return stats


# Usage Example
if __name__ == "__main__":
    # Initialize processor
    processor = DataProcessor(
        chunk_size=500,      # Adjust based on your needs
        chunk_overlap=50     # Keeps context between chunks
    )
    
    # Load scraped data
    scraped_data = processor.load_scraped_data('umat_data.json')
    
    if scraped_data:
        # Process into chunks
        chunks = processor.process_all_data(scraped_data)
        
        # Save processed chunks
        processor.save_chunks('processed_chunks.json')
        
        # Display statistics
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        stats = processor.get_statistics()
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Show sample chunks
        print("\n" + "="*50)
        print("SAMPLE CHUNKS")
        print("="*50)
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk {i}:")
            print(f"Source: {chunk['source_title']}")
            print(f"Text: {chunk['text'][:200]}...")
            print(f"Length: {len(chunk['text'])} characters")
