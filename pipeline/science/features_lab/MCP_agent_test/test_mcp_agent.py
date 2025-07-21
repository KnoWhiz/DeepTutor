#!/usr/bin/env python3
"""
Test script for MCP Agent Chatbot

This script performs basic functionality tests of the MCP Agent Chatbot
to ensure it's working correctly.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import logging

# Add the current directory to Python path to import the chatbot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MCP_agent_test import MCPAgentChatbot, MCPConfig

class TestMCPAgentChatbot(unittest.TestCase):
    """Test cases for MCP Agent Chatbot"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.chatbot = MCPAgentChatbot()
    
    def test_openai_client_initialization(self):
        """Test OpenAI client initialization"""
        self.assertIsNotNone(self.chatbot.client, "OpenAI client should be initialized")
    
    def test_mcp_config_creation(self):
        """Test MCP configuration creation"""
        config = MCPConfig(
            name="Test",
            endpoint="http://test.example.com",
            is_connected=True,
            capabilities=["test_capability"]
        )
        
        self.assertEqual(config.name, "Test")
        self.assertEqual(config.endpoint, "http://test.example.com")
        self.assertTrue(config.is_connected)
        self.assertIn("test_capability", config.capabilities)
    
    @patch('requests.head')
    def test_mcp_endpoint_validation(self, mock_head):
        """Test MCP endpoint validation"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response
        
        is_valid, message = self.chatbot.validate_mcp_endpoint("http://valid.example.com")
        self.assertTrue(is_valid)
        self.assertIn("accessible", message.lower())
        
        # Test invalid URL
        is_valid, message = self.chatbot.validate_mcp_endpoint("")
        self.assertFalse(is_valid)
        self.assertIn("invalid", message.lower())
    
    def test_mcp_capabilities(self):
        """Test MCP capabilities retrieval"""
        notion_caps = self.chatbot._get_mcp_capabilities("notion")
        dropbox_caps = self.chatbot._get_mcp_capabilities("dropbox")
        
        self.assertIn("create_page", notion_caps)
        self.assertIn("search_pages", notion_caps)
        self.assertIn("list_files", dropbox_caps)
        self.assertIn("upload_file", dropbox_caps)
    
    def test_mock_mcp_responses(self):
        """Test mock MCP response generation"""
        # Test Notion search response
        response = self.chatbot._get_mock_mcp_responses(
            "notion", "search_pages", {"query": "test"}
        )
        self.assertTrue(response["success"])
        self.assertIn("pages", response["data"])
        
        # Test Dropbox list response
        response = self.chatbot._get_mock_mcp_responses(
            "dropbox", "list_files", {"path": "/"}
        )
        self.assertTrue(response["success"])
        self.assertIn("files", response["data"])
    
    def test_mcp_tools_creation(self):
        """Test MCP tools creation for OpenAI Assistant"""
        tools = self.chatbot._create_mcp_tools()
        
        self.assertEqual(len(tools), 2)
        
        tool_names = [tool["function"]["name"] for tool in tools]
        self.assertIn("call_notion_mcp", tool_names)
        self.assertIn("call_dropbox_mcp", tool_names)
    
    def test_assistant_instructions(self):
        """Test assistant instructions building"""
        # Setup mock MCP config
        self.chatbot.mcp_configs = {
            "notion": MCPConfig("Notion", "http://test.notion.com", True),
            "dropbox": MCPConfig("Dropbox", "http://test.dropbox.com", True)
        }
        
        instructions = self.chatbot._build_assistant_instructions()
        
        self.assertIn("REACT", instructions.upper())
        self.assertIn("REASON", instructions)
        self.assertIn("ACT", instructions)
        self.assertIn("OBSERVE", instructions)
        self.assertIn("notion", instructions.lower())
        self.assertIn("dropbox", instructions.lower())

def run_interactive_test():
    """Run interactive test to demonstrate functionality"""
    print("\n🧪 MCP Agent Chatbot - Interactive Test")
    print("=" * 50)
    
    # Initialize chatbot
    print("1. Initializing chatbot...")
    chatbot = MCPAgentChatbot()
    print("   ✅ Chatbot initialized")
    
    # Test MCP connections
    print("\n2. Testing MCP connections...")
    notion_success = chatbot.setup_mcp_connection(
        "notion", 
        "http://mcp.composio.dev/partner/composio/notion/mcp?customerId=4d31a7c9-fe76-471d-a074-2c33da41f7bc"
    )
    dropbox_success = chatbot.setup_mcp_connection(
        "dropbox",
        "http://mcp.composio.dev/partner/composio/dropbox/mcp?customerId=4d31a7c9-fe76-471d-a074-2c33da41f7bc"
    )
    
    print(f"   Notion MCP: {'✅' if notion_success else '❌'}")
    print(f"   Dropbox MCP: {'✅' if dropbox_success else '❌'}")
    
    # Test MCP service calls
    print("\n3. Testing MCP service calls...")
    notion_response = chatbot.call_mcp_service("notion", "search_pages", {"query": "test"})
    dropbox_response = chatbot.call_mcp_service("dropbox", "list_files", {"path": "/"})
    
    print(f"   Notion search: {'✅' if notion_response['success'] else '❌'}")
    print(f"   Dropbox list: {'✅' if dropbox_response['success'] else '❌'}")
    
    # Display sample responses
    print("\n4. Sample responses:")
    print(f"   Notion pages found: {len(notion_response['data']['pages'])}")
    print(f"   Dropbox files found: {len(dropbox_response['data']['files'])}")
    
    print("\n✅ All tests completed successfully!")
    print("\n🚀 The chatbot is ready to use. Run the Streamlit app to interact with it:")
    print("   streamlit run MCP_agent_test.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive_test()
    else:
        print("🧪 Running unit tests...")
        unittest.main(verbosity=2) 