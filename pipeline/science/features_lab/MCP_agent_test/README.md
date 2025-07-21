# 🤖 MCP Agent Chatbot

An advanced AI chatbot built with Streamlit and OpenAI Assistant API, featuring Model Context Protocol (MCP) integration for Notion and Dropbox. Implements ReAct (Reasoning and Acting) agentic workflow for intelligent data retrieval and action execution.

## ✨ Features

- **🔗 MCP Integration**: Connect to Notion and Dropbox via MCP servers
- **🧠 ReAct Workflow**: Implements Reasoning → Acting → Observing cycles
- **🤖 OpenAI Assistant API**: Uses the latest Assistant API with tool calling
- **💬 Interactive Chat Interface**: Streamlit-based conversational UI
- **📊 Real-time Status**: Live connection and operation status
- **🛠️ Tool Calling**: Automatic MCP service selection based on user queries
- **📝 Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ and the `deeptutor` conda environment activated:
   ```bash
   conda activate deeptutor
   ```

2. **Required Packages**: The following packages should already be installed in your environment:
   - `streamlit`
   - `openai`
   - `requests`
   - Standard Python libraries (os, json, time, logging, etc.)

3. **OpenAI API Key**: The application uses the provided OpenAI API key embedded in the code

### Installation & Running

1. **Navigate to the project directory**:
   ```bash
   cd pipeline/science/features_lab/MCP_agent_test/
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run MCP_agent_test.py
   ```

3. **Open the application**: Your browser should automatically open to `http://localhost:8501`

## 📖 How to Use

### Step 1: Configure MCP Connections

1. **Open the sidebar** (if not already visible, click the arrow in the top-left)
2. **Configure Notion MCP**:
   - Enter your Notion MCP endpoint or leave blank to use the default
   - Click "Connect Notion MCP"
   - Wait for the green "✅ Notion MCP connected!" message

3. **Configure Dropbox MCP**:
   - Enter your Dropbox MCP endpoint or leave blank to use the default
   - Click "Connect Dropbox MCP"
   - Wait for the green "✅ Dropbox MCP connected!" message

**Default endpoints** (used if you leave the input fields blank or if your endpoint is invalid):
- **Notion**: `http://mcp.composio.dev/partner/composio/notion/mcp?customerId=4d31a7c9-fe76-471d-a074-2c33da41f7bc`
- **Dropbox**: `http://mcp.composio.dev/partner/composio/dropbox/mcp?customerId=4d31a7c9-fe76-471d-a074-2c33da41f7bc`

### Step 2: Initialize the Assistant

1. **Ensure at least one MCP service is connected** (green status)
2. **Click "🚀 Initialize Assistant"**
3. **Wait for initialization** (this creates the OpenAI Assistant and conversation thread)
4. **Confirm readiness** with the "✅ Assistant ready!" message

### Step 3: Start Chatting

Once the assistant is ready, you can start asking questions! The chatbot will:

1. **🧠 Reason** about your request
2. **⚡ Act** by calling appropriate MCP services
3. **👀 Observe** the results
4. **🔄 Repeat** if necessary until your query is fully addressed

## 💡 Example Queries

### Notion Examples
```
🔍 Search and Discovery:
- "Search for pages about machine learning"
- "Find all pages related to project planning"
- "Show me my recent meeting notes"

📝 Content Creation:
- "Create a new page called 'Project Plan'"
- "Create a database for tracking tasks"
- "Add a note about today's meeting to my project page"

📊 Database Operations:
- "Show me all tasks in my database"
- "Query my database for completed items"
- "Update the status of task XYZ"
```

### Dropbox Examples
```
📁 File Management:
- "List all files in my documents folder"
- "Show me files modified in the last week"
- "Create a new folder called 'Project Files'"

⬆️ Upload & Download:
- "Upload the file 'report.pdf' to my work folder"
- "Download my presentation from the shared folder"

🔍 Search Operations:
- "Search for files containing 'budget'"
- "Find all PDF files in my account"
- "Look for files shared with me recently"

🔗 Sharing:
- "Share my presentation with read-only access"
- "Get the share link for my document"
```

### Cross-Platform Examples
```
🔄 Integration Workflows:
- "Save my Notion meeting notes to Dropbox as a PDF"
- "Create a Notion page with content from my Dropbox file"
- "Backup my Notion database to Dropbox"
- "Import file list from Dropbox into a Notion database"
```

## 🎯 ReAct Workflow in Action

The chatbot follows a structured ReAct pattern:

### Example: "Create a new page for project planning"

1. **🧠 REASON**: 
   - User wants to create content
   - Notion is appropriate for page creation
   - Need to use `create_page` action

2. **⚡ ACT**: 
   - Call Notion MCP with `create_page` action
   - Pass page title and initial content

3. **👀 OBSERVE**: 
   - Check if page creation was successful
   - Verify page ID and URL returned

4. **💬 RESPOND**: 
   - Confirm page creation
   - Provide page URL and next steps

## 🔧 Advanced Configuration

### Custom MCP Endpoints

If you have your own MCP servers, you can configure custom endpoints:

1. **Enter your endpoint URL** in the sidebar
2. **Ensure the endpoint supports the MCP protocol**
3. **The system will validate connectivity** before establishing the connection

### Environment Variables

You can override the OpenAI API key by setting the environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 📊 Status Indicators

The sidebar shows real-time status for:

- **🟢/🔴 MCP Connections**: Connected/Disconnected status for each service
- **🟢/🔴 Assistant**: Ready/Not Ready status
- **🟢/🔴 Thread**: Active/Inactive conversation thread

## 🐛 Troubleshooting

### Common Issues

1. **"Assistant not initialized"**:
   - Ensure at least one MCP service is connected
   - Click "🚀 Initialize Assistant" again
   - Check the console for error messages

2. **MCP Connection Failed**:
   - Verify the endpoint URL is correct
   - Check your internet connection
   - Try using the default endpoints

3. **OpenAI API Errors**:
   - Verify your API key is valid
   - Check your OpenAI account has sufficient credits
   - Ensure you have access to GPT-4 models

4. **Slow Responses**:
   - This is normal for the first few interactions
   - Subsequent responses should be faster
   - Check your internet connection speed

### Logging

The application logs detailed information to:
- **Console**: Real-time status and errors
- **Log File**: `mcp_chatbot.log` in the same directory

To view logs in real-time:
```bash
tail -f mcp_chatbot.log
```

## 🔒 Security Notes

- **API Keys**: The OpenAI API key is embedded in the code for demonstration
- **MCP Endpoints**: Validate endpoints before connecting
- **Data Privacy**: All data flows through the configured MCP servers
- **Logging**: Sensitive information is not logged

## 🛠️ Development Notes

### Architecture

- **Frontend**: Streamlit web interface
- **AI Engine**: OpenAI Assistant API with GPT-4
- **Integration**: MCP (Model Context Protocol) for external services
- **Workflow**: ReAct pattern for reasoning and action

### Code Structure

```
MCP_agent_test.py
├── MCPConfig (dataclass)          # MCP configuration
├── MCPAgentChatbot (main class)   # Core chatbot logic
│   ├── OpenAI client setup
│   ├── MCP connection management
│   ├── Assistant creation & tools
│   ├── ReAct workflow execution
│   └── Message processing
└── main() (Streamlit app)         # UI and user interaction
```

### Extending the Chatbot

To add new MCP services:

1. **Add service configuration** in `_get_mcp_capabilities()`
2. **Create tool definitions** in `_create_mcp_tools()`
3. **Implement service calls** in `call_mcp_service()`
4. **Add UI elements** in the sidebar

## 📚 Additional Resources

- [OpenAI Assistant API Documentation](https://platform.openai.com/docs/assistants/overview)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Model Context Protocol (MCP)](https://github.com/anthropics/mcp)
- [ReAct: Reasoning and Acting with Language Models](https://arxiv.org/abs/2210.03629)

## 🤝 Support

For issues or questions:

1. **Check the logs** for detailed error information
2. **Review the troubleshooting section** above
3. **Verify all prerequisites** are met
4. **Restart the application** if needed

---

**Built with ❤️ by the DeepTutor Team** | *Advancing AI Education and Tools* 