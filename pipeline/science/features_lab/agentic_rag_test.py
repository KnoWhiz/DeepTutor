from typing import Literal, List, Optional
import os
import sys
import logging
import glob
from pathlib import Path

# Add the project root to the Python path so the pipeline module can be found
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent.parent
sys.path.append(str(project_root))
print(f"Added {project_root} to Python path")

# Now import the modules
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
# Import util functions from pipeline to use ApiHandler and get_llm
from pipeline.science.pipeline.utils import get_llm
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler
from pipeline.science.pipeline.embeddings import get_embedding_models
from pipeline.science.features_lab.visualize_graph_test import visualize_graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_rag_test")

# Define the tool for context retrieval
@tool
def retrieve_web_context(query: str):
    """Search for relevant documents from web sources."""
    try:
        # Example URL configuration
        urls = [
            "https://docs.python.org/3/tutorial/index.html",
            "https://realpython.com/python-basics/",
            "https://www.learnpython.org/"
        ]
        # Load documents
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(docs)

        # Use embeddings from the pipeline
        config = load_config()
        # The get_embedding_models function returns the model directly, not a dictionary
        embedding_model = get_embedding_models('default', config['llm'])

        # Create VectorStore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="python_docs",
            embedding=embedding_model,
        )
        retriever = vectorstore.as_retriever()
        results = retriever.invoke(query)
        
        if not results:
            return "No relevant information found from web sources."
            
        formatted_results = []
        for doc in results:
            source = doc.metadata.get('source', 'Unknown web source')
            formatted_results.append({
                "content": doc.page_content,
                "source": source,
                "type": "web"
            })
            
        return {"results": formatted_results, "source_type": "web"}
    except Exception as e:
        logger.error(f"Error retrieving web context: {e}")
        return f"Error retrieving web context: {e}"

def discover_pdf_files(directory: str) -> List[str]:
    """
    Find all PDF files in the specified directory and its subdirectories.
    
    Args:
        directory: Path to directory to search
        
    Returns:
        List of paths to PDF files
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return []
    
    # Use glob to find all .pdf files in the directory and subdirectories
    pdf_pattern = os.path.join(directory, "**", "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files

@tool
def retrieve_pdf_context(query: str, pdf_files: List[str] = None):
    """Search for relevant information from PDF documents.
    
    Args:
        query: The search query
        pdf_files: Optional list of PDF file paths. If not provided, will use default paths.
    """
    try:
        # # If no PDF files provided, use example default paths
        # if not pdf_files:
        #     # Example PDF file paths - in a real application, these could be loaded from a directory
        #     pdf_files = [
        #         "/Users/bingranyou/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/Multiplexed_single_photon_source_arXiv__resubmit_.pdf",
        #         "/Users/bingranyou/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/PhysRevLett.130.213601.pdf",
        #         "/Users/bingranyou/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/PRXQuantum.5.020308.pdf"
        #     ]

        # Filter for existing files
        existing_files = [f for f in pdf_files if Path(f).exists()]
        
        if not existing_files:
            return "No PDF files found. Please provide valid PDF file paths."
        
        all_docs = []
        # Load documents from PDFs
        for pdf_path in existing_files:
            try:
                loader = PyMuPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {pdf_path}")
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {e}")
        
        if not all_docs:
            return "Failed to extract content from PDF files."
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(all_docs)
        
        # Use embeddings from the pipeline
        config = load_config()
        embedding_model = get_embedding_models('default', config['llm'])
        
        # Create VectorStore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="pdf_docs",
            embedding=embedding_model,
        )
        retriever = vectorstore.as_retriever()
        results = retriever.invoke(query)
        
        if not results:
            return "No relevant information found in PDF documents."
            
        formatted_results = []
        for doc in results:
            source = doc.metadata.get('source', 'Unknown PDF')
            page = doc.metadata.get('page', 'Unknown page')
            formatted_results.append({
                "content": doc.page_content,
                "source": source,
                "page": page,
                "type": "pdf"
            })
            
        return {"results": formatted_results, "source_type": "pdf"}
    except Exception as e:
        logger.error(f"Error retrieving PDF context: {e}")
        return f"Error retrieving PDF context: {e}"

@tool
def search_all_sources(query: str, pdf_files: List[str] = None):
    """
    Search both web sources and PDF documents for relevant information.
    
    Args:
        query: The search query
        pdf_files: Optional list of PDF file paths
    """
    try:
        # Get results from web sources
        web_results = retrieve_web_context.invoke(query)
        
        # Get results from PDF documents
        pdf_results = retrieve_pdf_context.invoke(query, pdf_files)
        
        # Combine results
        results = {
            "web_sources": web_results if isinstance(web_results, dict) else {"error": web_results},
            "pdf_sources": pdf_results if isinstance(pdf_results, dict) else {"error": pdf_results}
        }
        
        return results
    except Exception as e:
        logger.error(f"Error searching all sources: {e}")
        return f"Error searching all sources: {e}"

def agentic_rag(pdf_directory: Optional[str] = None, query: str = None):
    """
    Run an agentic RAG workflow that combines web and PDF content.
    
    Args:
        pdf_directory: Optional directory to search for PDF files
        query: Optional query to use instead of the default one
    """
    try:
        # Get PDF files if directory is provided
        pdf_files = []
        if pdf_directory:
            pdf_files = discover_pdf_files(pdf_directory)
        
        # Configure tools
        tools = [retrieve_web_context, retrieve_pdf_context, search_all_sources]
        tool_node = ToolNode(tools)

        # Load config to get LLM parameters
        config = load_config()
        llm_params = config['llm']
        
        # Use get_llm to get the advanced model (gpt-4o), with stream=False for tool calling
        model = get_llm('advanced', llm_params, stream=False)
        
        # Bind tools to the model
        model = model.bind_tools(tools)

        # Function to decide whether to continue or stop the workflow
        def should_continue(state: MessagesState) -> Literal["tools", END]:
            messages = state['messages']
            last_message = messages[-1]
            # If the LLM makes a tool call, go to the "tools" node
            if last_message.tool_calls:
                return "tools"
            # Otherwise, finish the workflow
            return END

        # Function that invokes the model
        def call_model(state: MessagesState):
            messages = state['messages']
            response = model.invoke(messages)
            return {"messages": [response]}  # Returns as a list to add to the state

        # Define the workflow with LangGraph
        workflow = StateGraph(MessagesState)

        # Add nodes to the graph
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # Connect nodes
        workflow.add_edge(START, "agent")  # Initial entry
        workflow.add_conditional_edges("agent", should_continue)  # Decision after the "agent" node
        workflow.add_edge("tools", "agent")  # Cycle between tools and agent

        # Configure memory to persist the state
        checkpointer = MemorySaver()

        # Compile the graph into a LangChain Runnable application
        app = workflow.compile(checkpointer=checkpointer)

        # Visualize the graph
        visualize_graph(app, "graph_diagram.mmd")

        # Prepare input for the workflow
        if not query:
            query = "Please explain Python data structures (lists, dictionaries, sets) with examples, and include information from both web sources and any available PDF documents."
        
        # Create initial message with instructions for the agent
        initial_message = f"""
{query}

As an agentic RAG system, you have access to multiple information sources:
1. Web sources through the retrieve_web_context tool
2. PDF documents through the retrieve_pdf_context tool
3. A combined search across all sources through the search_all_sources tool

To provide the most comprehensive answer:
- Consider which sources would be most relevant to the query
- You can use both web and PDF sources together
- Specify which source information came from in your response
- Feel free to use the search_all_sources tool for a unified search"""

        # If PDF files are available, add them to the message
        if pdf_files:
            pdf_list = "\n".join([f"- {os.path.basename(pdf)}" for pdf in pdf_files])
            initial_message += f"\n\nThe following PDF files are available:\n{pdf_list}"
            # Pass PDF files to the search_all_sources tool
            initial_message += f"\n\nYou can use these PDF files in your search by passing them to the retrieve_pdf_context or search_all_sources tools."
        
        # Execute the workflow
        logger.info("Starting workflow execution...")
        final_state = app.invoke(
            {"messages": [HumanMessage(content=initial_message)]},
            config={"configurable": {"thread_id": 42}}
        )

        # Show the final response
        print("Final response:\n\n")
        print(final_state["messages"][-1].content)
        
        return final_state["messages"][-1].content
    except Exception as e:
        logger.error(f"Error in agentic_rag: {e}")
        raise

if __name__ == "__main__":
    try:
        # Check if API keys are set
        from dotenv import load_dotenv
        load_dotenv()
        
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # Run the test with a specified PDF directory if it exists
        pdf_dir = "/Users/bingranyou/Library/Mobile Documents/com~apple~CloudDocs/Downloads/temp/"
        if os.path.exists(pdf_dir):
            result = agentic_rag(pdf_directory=pdf_dir, query="What is the main idea of the multiplexed single photon source?")
        else:
            # result = agentic_rag()
            raise ValueError(f"PDF directory {pdf_dir} does not exist")
    except Exception as e:
        logger.error(f"Test failed: {e}")
