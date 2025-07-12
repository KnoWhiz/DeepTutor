#!/usr/bin/env python3
"""
Test script for the Azure OpenAI PDF Chatbot

This script tests the basic functionality of the chatbot without the interactive session.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai_file_api_test import AzureOpenAIPDFChatbot

def test_chatbot():
    """Test the chatbot with a simple question."""
    
    # PDF file path
    pdf_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/Multiplexed_single_photon_source_arXiv__resubmit_.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return
    
    print("🧪 Testing Azure OpenAI PDF Chatbot...")
    print(f"📚 PDF file: {os.path.basename(pdf_path)}")
    
    try:
        # Create chatbot instance
        chatbot = AzureOpenAIPDFChatbot()
        
        # Upload PDF and setup
        print("📤 Uploading PDF...")
        file_ids = chatbot.upload_pdf_files([pdf_path])
        
        print("🔍 Creating vector store...")
        chatbot.create_vector_store(file_ids)
        
        print("🤖 Creating assistant...")
        chatbot.create_assistant()
        
        print("💬 Creating thread...")
        chatbot.create_thread()
        
        print("✅ Setup complete!")
        
        # Test with a simple question
        test_question = "What is this paper about? Please provide a brief summary."
        print(f"\n❓ Test question: {test_question}")
        
        print("🤖 Getting response...")
        response = chatbot.send_message_and_get_response(test_question)
        
        print(f"\n📝 Response:\n{response}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        chatbot.cleanup()
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chatbot() 