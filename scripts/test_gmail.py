#!/usr/bin/env python3
"""
Quick test script to verify Gmail integration works
Run this to test OAuth and email fetching
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.gmail_service import GmailService

def test_gmail_connection():
    print("Testing Gmail connection...")
    
    try:
        gmail = GmailService()
        print("✅ OAuth authentication successful")
        
        print("Fetching recent emails...")
        emails = gmail.get_recent_emails(max_results=5)
        
        print(f"✅ Retrieved {len(emails)} emails")
        
        for i, email in enumerate(emails[:3]):
            print(f"\n--- Email {i+1} ---")
            print(f"Subject: {email['subject'][:60]}...")
            print(f"From: {email['sender']}")
            print(f"Body preview: {email['body'][:100]}...")
        
        # Test chunking with DocumentProcessor
        if emails:
            print("\nTesting email-to-document conversion...")
            from app.services.document_processor import DocumentProcessor
            processor = DocumentProcessor()

            # Convert first email to document format
            email = emails[0]
            document = {
                "id": email['id'],
                "title": email['subject'],
                "content": f"From: {email['sender']}\nSubject: {email['subject']}\n\n{email['body']}",
                "created_time": email.get('date'),
                "last_edited_time": email.get('date')
            }

            chunks = processor.process_document(document)
            print(f"✅ First email converted to {len(chunks)} chunks using DocumentProcessor")

            if chunks:
                print(f"Sample chunk format:")
                print(f"  ID: {chunks[0]['id']}")
                print(f"  Text preview: {chunks[0]['text'][:100]}...")
                print(f"  Metadata keys: {list(chunks[0]['metadata'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_gmail_connection()
    if success:
        print("\n🎉 Gmail integration is working!")
    else:
        print("\n💥 Gmail integration needs debugging")