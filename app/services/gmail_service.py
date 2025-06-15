import os
import base64
from datetime import datetime, timezone
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json
import re
from bs4 import BeautifulSoup

class GmailService:
    def __init__(self):
        self.SCOPES = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/calendar.readonly'
        ]
        self.service = self._authenticate()
    
    def _authenticate(self):
        """Handle OAuth2 authentication"""
        creds = None
        token_path = 'config/token.json'
        credentials_path = 'config/credentials.json'
        
        # Load existing token
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        
        # If no valid credentials, run OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        return build('gmail', 'v1', credentials=creds)
    
    def get_recent_emails(self, max_results=50, days_back=30):
        """Get recent emails with content"""
        try:
            # Query for recent emails
            query = f"newer_than:{days_back}d"
            results = self.service.users().messages().list(
                userId='me', 
                q=query, 
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            emails = []
            for message in messages:
                email_data = self._get_email_content(message['id'])
                if email_data:
                    emails.append(email_data)
            
            return emails
            
        except Exception as error:
            print(f'An error occurred: {error}')
            return []
    
    def _get_email_content(self, message_id):
        """Extract email content and metadata"""
        try:
            message = self.service.users().messages().get(
                userId='me', 
                id=message_id, 
                format='full'
            ).execute()
            
            # Extract headers
            headers = {}
            for header in message['payload'].get('headers', []):
                headers[header['name'].lower()] = header['value']
            
            # Extract body content
            body = self._extract_body(message['payload'])
            
            # Clean and format
            return {
                'id': message_id,
                'subject': headers.get('subject', 'No Subject'),
                'sender': headers.get('from', 'Unknown'),
                'date': headers.get('date', ''),
                'body': self._clean_text(body),
                'thread_id': message.get('threadId', ''),
                'timestamp': self._parse_date(headers.get('date', ''))
            }
            
        except Exception as error:
            print(f'Error getting email {message_id}: {error}')
            return None
    
    def _extract_body(self, payload):
        """Extract text content from email payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    break
                elif part['mimeType'] == 'text/html':
                    data = part['body']['data']
                    html_body = base64.urlsafe_b64decode(data).decode('utf-8')
                    # Proper HTML parsing with BeautifulSoup
                    soup = BeautifulSoup(html_body, 'html.parser')
                    body = soup.get_text(separator=' ', strip=True)
        else:
            if payload['body'].get('data'):
                raw_body = base64.urlsafe_b64decode(
                    payload['body']['data']
                ).decode('utf-8')
                
                # Check if it's HTML content
                if payload.get('mimeType') == 'text/html':
                    soup = BeautifulSoup(raw_body, 'html.parser')
                    body = soup.get_text(separator=' ', strip=True)
                else:
                    body = raw_body
        
        return body
    
    def _clean_text(self, text):
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common email artifacts
        text = re.sub(r'On .* wrote:', '', text)
        text = re.sub(r'From:.*?Subject:', 'Subject:', text, flags=re.DOTALL)
        
        return text.strip()
    
    def _parse_date(self, date_str):
        """Parse email date to timestamp"""
        try:
            # This is a simplified parser - email dates can be complex
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            return dt.timestamp()
        except:
            return datetime.now().timestamp()
    
    def email_to_chunks(self, email_data, chunk_size=2048):
        """Convert email to document chunks for embedding"""
        
        # Combine subject and body for full context
        full_content = f"Subject: {email_data['subject']}\n\n{email_data['body']}"
        
        # For emails, we usually want the full message as context
        # But split if too long
        chunks = []
        
        if len(full_content) <= chunk_size:
            chunks.append({
                'content': full_content,
                'source': 'gmail',
                'metadata': {
                    'email_id': email_data['id'],
                    'sender': email_data['sender'],
                    'subject': email_data['subject'],
                    'timestamp': email_data['timestamp'],
                    'content_type': 'email'
                }
            })
        else:
            # Split long emails into chunks but preserve metadata
            words = full_content.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > chunk_size and current_chunk:
                    chunks.append({
                        'content': ' '.join(current_chunk),
                        'source': 'gmail',
                        'metadata': {
                            'email_id': email_data['id'],
                            'sender': email_data['sender'],
                            'subject': email_data['subject'],
                            'timestamp': email_data['timestamp'],
                            'content_type': 'email_chunk'
                        }
                    })
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(word)
                current_length += len(word) + 1
            
            # Add remaining chunk
            if current_chunk:
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'source': 'gmail',
                    'metadata': {
                        'email_id': email_data['id'],
                        'sender': email_data['sender'],
                        'subject': email_data['subject'],
                        'timestamp': email_data['timestamp'],
                        'content_type': 'email_chunk'
                    }
                })
        
        return chunks