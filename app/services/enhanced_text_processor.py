"""
Enhanced text preprocessing for better semantic understanding
"""
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import email
from email.mime.text import MIMEText

class AdvancedTextProcessor:
    """Advanced text processing for context engine"""
    
    def __init__(self):
        # Common email signature patterns
        self.signature_patterns = [
            r"--\s*\n.*",  # Standard signature delimiter
            r"Best regards.*",
            r"Thanks.*\n.*",
            r"Sent from my.*",
            r"Get Outlook for.*",
            r"This email was sent from.*"
        ]
        
        # Forwarded/reply patterns
        self.forward_patterns = [
            r"---------- Forwarded message.*",
            r"From:.*\nTo:.*\nSubject:.*",
            r"On.*wrote:",
            r">.*\n",  # Quoted lines
            r"^\s*>+.*$"  # Multi-level quotes
        ]
        
    def clean_email_content(self, email_text: str, subject: str = "") -> Dict[str, str]:
        """Extract clean content from email"""
        
        # Remove HTML tags
        if '<html>' in email_text.lower() or '<div>' in email_text.lower():
            soup = BeautifulSoup(email_text, 'html.parser')
            email_text = soup.get_text()
        
        # Normalize whitespace
        email_text = re.sub(r'\s+', ' ', email_text.strip())
        
        # Split into original vs quoted content
        original_content = self._extract_original_content(email_text)
        quoted_content = self._extract_quoted_content(email_text)
        
        # Remove signatures from original content
        clean_original = self._remove_signatures(original_content)
        
        return {
            "original": clean_original,
            "quoted": quoted_content,
            "subject_clean": self._clean_subject(subject),
            "content_type": self._classify_email_type(clean_original, subject)
        }
    
    def _extract_original_content(self, text: str) -> str:
        """Extract only the original message content"""
        lines = text.split('\n')
        original_lines = []
        
        for line in lines:
            # Stop at forwarded message indicators
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.forward_patterns):
                break
            # Skip quoted lines (starting with >)
            if not line.strip().startswith('>'):
                original_lines.append(line)
        
        return '\n'.join(original_lines).strip()
    
    def _extract_quoted_content(self, text: str) -> str:
        """Extract quoted/forwarded content"""
        quoted_lines = []
        lines = text.split('\n')
        
        for line in lines:
            if line.strip().startswith('>'):
                # Remove quote markers and clean
                clean_line = re.sub(r'^>+\s*', '', line).strip()
                if clean_line:
                    quoted_lines.append(clean_line)
        
        return '\n'.join(quoted_lines).strip()
    
    def _remove_signatures(self, text: str) -> str:
        """Remove email signatures"""
        for pattern in self.signature_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()
    
    def _clean_subject(self, subject: str) -> str:
        """Clean email subject line"""
        # Remove Re:, Fwd:, etc.
        subject = re.sub(r'^(Re|Fwd|FW):\s*', '', subject, flags=re.IGNORECASE)
        # Remove excessive punctuation
        subject = re.sub(r'[!]{2,}', '!', subject)
        subject = re.sub(r'[?]{2,}', '?', subject)
        return subject.strip()
    
    def _classify_email_type(self, content: str, subject: str) -> str:
        """Classify email type for better processing"""
        content_lower = content.lower()
        subject_lower = subject.lower()
        
        # Job-related
        if any(term in content_lower + subject_lower for term in 
               ['job', 'position', 'interview', 'candidate', 'hire', 'opportunity']):
            return "job_related"
        
        # Newsletter/automated
        if any(term in content_lower for term in 
               ['unsubscribe', 'newsletter', 'digest', 'automated']):
            return "newsletter"
        
        # Personal communication
        if any(term in content_lower for term in 
               ['thanks', 'please', 'could you', 'would you']):
            return "personal"
        
        # Technical/work
        if any(term in content_lower for term in 
               ['code', 'api', 'bug', 'feature', 'development']):
            return "technical"
        
        return "general"

class SemanticChunker:
    """Intelligent chunking that preserves semantic boundaries"""
    
    def __init__(self, max_chunk_size: int = 512, overlap_size: int = 128):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_email(self, email_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from email"""
        chunks = []
        
        # Separate handling for different content types
        original_content = email_data["original"]
        subject = email_data["subject_clean"]
        content_type = email_data["content_type"]
        
        if not original_content.strip():
            return []
        
        # For short emails, keep as single chunk
        if len(original_content) <= self.max_chunk_size:
            chunks.append({
                "text": f"Subject: {subject}\n\n{original_content}",
                "chunk_type": "complete_email",
                "content_type": content_type,
                "has_subject": True
            })
        else:
            # Split by sentences for better semantic boundaries
            sentences = self._split_into_sentences(original_content)
            
            # Create chunks that respect sentence boundaries
            chunks.extend(self._create_sentence_aware_chunks(
                sentences, subject, content_type
            ))
        
        return chunks
    
    def chunk_notion_page(self, page_content: str, title: str) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from Notion pages"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in page_content.split('\n\n') if p.strip()]
        
        current_chunk = f"Title: {title}\n\n"
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds limit, save current chunk
            if len(current_chunk + paragraph) > self.max_chunk_size:
                if len(current_chunk.strip()) > len(f"Title: {title}\n\n"):
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_type": "notion_section",
                        "content_type": "knowledge_base",
                        "has_title": True
                    })
                
                # Start new chunk with overlap
                current_chunk = f"Title: {title}\n\n{paragraph}\n\n"
            else:
                current_chunk += f"{paragraph}\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_type": "notion_section",
                "content_type": "knowledge_base",
                "has_title": True
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (could be enhanced with spaCy/NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_sentence_aware_chunks(self, sentences: List[str], subject: str, content_type: str) -> List[Dict[str, Any]]:
        """Create chunks that respect sentence boundaries"""
        chunks = []
        current_chunk = f"Subject: {subject}\n\n"
        
        for sentence in sentences:
            # Check if adding this sentence exceeds the limit
            if len(current_chunk + sentence) > self.max_chunk_size:
                # Save current chunk if it has content
                if len(current_chunk.strip()) > len(f"Subject: {subject}\n\n"):
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_type": "email_section",
                        "content_type": content_type,
                        "has_subject": True
                    })
                
                # Start new chunk with subject (for context)
                current_chunk = f"Subject: {subject}\n\n{sentence} "
            else:
                current_chunk += f"{sentence} "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_type": "email_section", 
                "content_type": content_type,
                "has_subject": True
            })
        
        return chunks