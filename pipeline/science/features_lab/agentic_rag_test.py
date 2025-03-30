from typing import Literal, List, Optional, Dict, Any
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

# Create a simple wrapper to handle callback contexts
class SafeToolInvoker:
    """Wrapper to safely invoke tools with proper error handling for callbacks."""
    
    @staticmethod
    def invoke_tool(tool, **kwargs):
        """Safely invoke a tool with proper error handling."""
        # Get the tool name safely
        def get_tool_name(tool):
            # Try different attributes to get the tool name
            if hasattr(tool, '__name__'):
                return tool.__name__
            elif hasattr(tool, 'name'):
                return tool.name
            # For StructuredTool objects
            elif hasattr(tool, 'func') and hasattr(tool.func, '__name__'):
                return tool.func.__name__
            else:
                return str(tool).split()[0]  # Fallback to the first part of the string representation
        
        try:
            # For LangChain BaseTool objects that need 'input' parameter
            if hasattr(tool, 'invoke'):
                # Check if it's a BaseTool from LangChain that expects an 'input' parameter
                import inspect
                sig = inspect.signature(tool.invoke)
                if 'input' in sig.parameters:
                    # Convert kwargs to a single input dict for BaseTool
                    return tool.invoke(input=kwargs)
                else:
                    # Use kwargs directly if the invoke method doesn't expect 'input'
                    return tool.invoke(**kwargs)
            # Fall back to direct call if needed
            return tool(**kwargs)
        except AttributeError as e:
            # Check if this is a callback-related error and handle it
            if "parent_run_id" in str(e) or "raise_error" in str(e) or "ignore_agent" in str(e):
                logger.warning(f"Ignoring callback-related error: {e}")
                # Call the function directly as a workaround
                return tool.__call__(**kwargs)
            else:
                # Re-raise other AttributeError issues
                raise
        except Exception as e:
            tool_name = get_tool_name(tool)
            logger.error(f"Error invoking tool {tool_name}: {e}")
            # Return a structured error that won't cause additional problems
            return {"error": f"Error in {tool_name}: {str(e)}"}

# Define the tool for context retrieval
@tool
def retrieve_web_context(query: str):
    """Search for relevant documents from web sources."""
    try:
        # Example URL configuration for general and scientific sources
        general_urls = [
            "https://docs.python.org/3/tutorial/index.html",
            "https://realpython.com/python-basics/",
            "https://www.learnpython.org/"
        ]
        
        scientific_urls = [
            "https://en.wikipedia.org/wiki/Quantum_optics",
            "https://en.wikipedia.org/wiki/Single-photon_source",
            "https://en.wikipedia.org/wiki/Multiplexing",
            "https://en.wikipedia.org/wiki/Quantum_information",
            "https://physics.nist.gov/Divisions/Div844/publications.html"  # NIST Quantum Physics publications
        ]
        
        # Determine if this is a scientific query
        scientific_keywords = ["quantum", "photon", "optics", "physics", "multiplexing", "temporal", "ion"]
        is_scientific_query = any(keyword in query.lower() for keyword in scientific_keywords)
        
        # Choose appropriate URLs based on query type
        urls = scientific_urls if is_scientific_query else general_urls
        
        # Load documents
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
            return {"results": [], "source_type": "web", "message": "No relevant information found from web sources."}
            
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
        return {"error": f"Error retrieving web context: {e}", "results": [], "source_type": "web"}

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
def grade_document_relevance(document_content: str, query: str):
    """
    Grade the relevance of a document to the query on a scale of 0-10.
    Follows the Corrective-RAG approach of evaluating document relevance.
    
    Args:
        document_content: The content of the document
        query: The search query
    
    Returns:
        Dictionary with relevance score and reasoning
    """
    try:
        # Load config to get LLM parameters
        config = load_config()
        llm_params = config['llm']
        
        # Use get_llm to get the model for grading
        model = get_llm('advanced', llm_params, stream=False)
        
        # Create the prompt for grading
        grading_prompt = f"""
        You are an expert scientific document evaluator specializing in quantum physics. Your task is to evaluate how relevant a document is to a technical scientific query.
        
        Query: {query}
        
        Document Content:
        {document_content}
        
        Please evaluate the relevance of this document to the query on a scale of 0-10, where:
        - 0: Completely irrelevant to the scientific topic
        - 3: Mentions related topics but not directly relevant to the specific query
        - 5: Contains some relevant technical information about the query
        - 7: Directly discusses the scientific topic but may lack depth
        - 10: Highly relevant, contains detailed technical information directly addressing the scientific query
        
        For technical/scientific content, consider:
        1. Presence of relevant technical terminology
        2. Discussion of related experimental setups
        3. Mentions of relevant physical processes
        4. Quantitative data or equations related to the topic
        5. Technical descriptions of methods or approaches
        
        Respond with a JSON object with the following structure:
        {{
            "relevance_score": <score>,
            "reasoning": "<your detailed reasoning for the score>",
            "key_technical_terms": ["<list of technical terms found>"]
        }}
        """
        
        # Get the grading result
        response = model.invoke(grading_prompt)
        
        # Extract JSON from the response
        import json
        import re
        
        # Handle possible code block formatting in the response
        content = response.content
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
        
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            # Fallback with a default structure
            return {
                "relevance_score": 0,
                "reasoning": "Failed to parse grading result, assuming irrelevant",
                "key_technical_terms": []
            }
    except Exception as e:
        logger.error(f"Error grading document relevance: {e}")
        return {
            "relevance_score": 0,
            "reasoning": f"Error in grading: {str(e)}",
            "key_technical_terms": []
        }

@tool
def retrieve_pdf_context(query: str, pdf_files: List[str] = None):
    """Search for relevant information from PDF documents.
    
    Args:
        query: The search query
        pdf_files: Optional list of PDF file paths. If not provided, will use default paths.
    """
    try:
        # Handle the case when pdf_files is None or empty
        if not pdf_files:
            return {"error": "No PDF files provided for search", "results": [], "source_type": "pdf"}
        
        # Filter for existing files
        existing_files = [f for f in pdf_files if Path(f).exists()]
        
        if not existing_files:
            return {"error": "No PDF files found. Please provide valid PDF file paths.", "results": [], "source_type": "pdf"}
        
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
            return {"error": "Failed to extract content from PDF files.", "results": [], "source_type": "pdf"}
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
            return {"results": [], "source_type": "pdf", "message": "No relevant information found in PDF documents."}
            
        # Following CRAG approach: grade and filter documents based on relevance
        formatted_results = []
        for doc in results:
            source = doc.metadata.get('source', 'Unknown PDF')
            page = doc.metadata.get('page', 'Unknown page')
            
            # Grade document relevance using the safe invoker
            grade_result = SafeToolInvoker.invoke_tool(
                grade_document_relevance, 
                document_content=doc.page_content, 
                query=query
            )
            
            if isinstance(grade_result, dict):
                relevance_score = grade_result.get("relevance_score", 0)
                reasoning = grade_result.get("reasoning", "No reasoning provided")
                key_technical_terms = grade_result.get("key_technical_terms", [])
            else:
                relevance_score = 0
                reasoning = "Error in relevance grading"
                key_technical_terms = []
            
            # Include relevance information in results
            formatted_results.append({
                "content": doc.page_content,
                "source": source,
                "page": page,
                "type": "pdf",
                "relevance_score": relevance_score,
                "relevance_reasoning": reasoning,
                "key_technical_terms": key_technical_terms
            })
        
        # Sort results by relevance score (highest first)
        formatted_results = sorted(formatted_results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return {"results": formatted_results, "source_type": "pdf"}
    except Exception as e:
        logger.error(f"Error retrieving PDF context: {e}")
        return {"error": f"Error retrieving PDF context: {e}", "results": [], "source_type": "pdf"}

@tool
def reformulate_query(original_query: str, search_context: str = ""):
    """
    Reformulate a query to make it more effective for retrieval.
    
    Args:
        original_query: The original search query
        search_context: Optional additional context about search results so far
        
    Returns:
        A reformulated query
    """
    try:
        # Load config to get LLM parameters
        config = load_config()
        llm_params = config['llm']
        
        # Use get_llm to get the model for reformulation
        model = get_llm('advanced', llm_params, stream=False)
        
        # Create the prompt for query reformulation
        reformulation_prompt = f"""
        You are an expert in scientific information retrieval and quantum physics. Your task is to reformulate a search query to make it more effective at finding technical information.
        
        Original Query: {original_query}
        
        {f"Search Context So Far: {search_context}" if search_context else ""}
        
        Please reformulate the query to:
        1. Be more specific and targeted for technical/scientific content
        2. Include key technical terms that would appear in scientific papers on this topic
        3. Break down complex technical questions into searchable components
        4. Focus on essential scientific concepts and terminology
        5. Include alternative technical terms for the same concepts (e.g., "temporal multiplexing", "time multiplexing")
        6. Add specific related physical processes or components that might be discussed in papers on this topic
        
        Provide only the reformulated query without any explanation or additional text.
        """
        
        # Get the reformulated query
        response = model.invoke(reformulation_prompt)
        
        # Return the reformulated query
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error reformulating query: {e}")
        return original_query  # Return the original query if reformulation fails

@tool
def search_all_sources(query: str, pdf_files: List[str] = None):
    """
    Search both web sources and PDF documents for relevant information.
    
    Args:
        query: The search query
        pdf_files: Optional list of PDF file paths
    """
    try:
        # Get results from web sources using the safe invoker
        web_results = SafeToolInvoker.invoke_tool(retrieve_web_context, query=query)
        
        # Get results from PDF documents if pdf_files is provided
        if pdf_files:
            # Convert string representation to list if needed
            if isinstance(pdf_files, str):
                if pdf_files.startswith("[") and pdf_files.endswith("]"):
                    # This looks like a string representation of a list, try to parse it
                    import ast
                    try:
                        pdf_files = ast.literal_eval(pdf_files)
                    except (SyntaxError, ValueError):
                        # If parsing fails, assume it's a single file path
                        pdf_files = [pdf_files]
                else:
                    # Assume it's a single file path
                    pdf_files = [pdf_files]
            
            # Now we should have a proper list of file paths
            pdf_results = SafeToolInvoker.invoke_tool(
                retrieve_pdf_context, 
                query=query, 
                pdf_files=pdf_files
            )
        else:
            pdf_results = {"error": "No PDF files provided for search", "results": [], "source_type": "pdf"}
        
        # Combine results
        results = {
            "web_sources": web_results if isinstance(web_results, dict) else {"error": web_results, "results": []},
            "pdf_sources": pdf_results if isinstance(pdf_results, dict) else {"error": pdf_results, "results": []}
        }
        
        # Check if any source has relevant information
        has_relevant_web = isinstance(web_results, dict) and len(web_results.get("results", [])) > 0
        has_relevant_pdf = isinstance(pdf_results, dict) and len(pdf_results.get("results", [])) > 0
        
        # If no relevant information from either source, try reformulating the query
        if not has_relevant_web and not has_relevant_pdf:
            reformulated_query = SafeToolInvoker.invoke_tool(
                reformulate_query, 
                original_query=query
            )
            
            if isinstance(reformulated_query, str) and reformulated_query != query:
                results["reformulated_query"] = reformulated_query
                results["suggestion"] = f"No relevant information found. Consider trying the reformulated query: '{reformulated_query}'"
        
        return results
    except Exception as e:
        logger.error(f"Error searching all sources: {e}")
        return {"error": f"Error searching all sources: {e}", "web_sources": {}, "pdf_sources": {}}

@tool
def scientific_search(query: str, specific_domain: str = "quantum_physics"):
    """
    Perform a specialized search for scientific/technical information in a specific domain.
    This tool is optimized for finding detailed technical information in specific fields.
    
    Args:
        query: The technical/scientific search query
        specific_domain: The scientific domain to focus on (default: quantum_physics)
    
    Returns:
        Formatted search results with technical information
    """
    try:
        # This would ideally connect to scientific databases or specialized search services
        # For now, we'll use our existing tools but with enhanced parameters
        
        # First, reformulate the query for scientific search
        domain_enhanced_query = f"Technical scientific details of {query} in {specific_domain} research papers"
        
        # Load config to get LLM parameters
        config = load_config()
        llm_params = config['llm']
        
        # Use get_llm to get the model for specialized search
        model = get_llm('advanced', llm_params, stream=False)
        
        # Expand the query with technical terms
        expansion_prompt = f"""
        You are an expert in {specific_domain}. Please expand the following technical query with relevant scientific terminology,
        equations, and key concepts that would help in retrieving specialized information from research papers.
        
        Query: {query}
        
        Provide only the expanded search terms without explanation. Include multiple variations of technical terminology.
        """
        
        expanded_query = model.invoke(expansion_prompt).content.strip()
        
        # Use the expanded query for web search
        web_results = SafeToolInvoker.invoke_tool(retrieve_web_context, query=expanded_query)
        
        # Add domain-specific interpretation of results
        if isinstance(web_results, dict) and web_results.get("results"):
            interpretation_prompt = f"""
            You are a scientific research assistant specializing in {specific_domain}. 
            Analyze these search results about "{query}" and extract the most technically relevant information.
            Focus on extracting exact scientific details, measurements, techniques, and theoretical frameworks.
            
            Search results:
            {web_results.get("results")}
            
            Provide a concise, technically precise extraction of the most relevant scientific information.
            """
            
            scientific_interpretation = model.invoke(interpretation_prompt).content.strip()
            
            # Add the scientific interpretation to the results
            if isinstance(web_results, dict):
                web_results["scientific_interpretation"] = scientific_interpretation
        
        return web_results
    except Exception as e:
        logger.error(f"Error in scientific search: {e}")
        return {"error": f"Error in scientific search: {e}", "results": [], "source_type": "scientific"}

@tool
def iterative_search(query: str, pdf_files: List[str] = None, rounds: int = 3):
    """
    Perform iterative, multi-round search with query refinement for difficult technical topics.
    This tool automatically conducts multiple rounds of search with feedback-based refinement.
    
    Args:
        query: The initial search query
        pdf_files: Optional list of PDF file paths
        rounds: Number of search iterations to perform (default: 3)
    """
    try:
        # Load config to get LLM parameters
        config = load_config()
        llm_params = config['llm']
        
        # Use get_llm to get the model for specialized search
        model = get_llm('advanced', llm_params, stream=False)
        
        # Initialize storage for all retrieved information
        all_results = []
        all_queries = [query]
        current_query = query
        
        # Conduct multiple rounds of searching
        for i in range(rounds):
            logger.info(f"Iterative search round {i+1}/{rounds} with query: {current_query}")
            
            # First try scientific search
            science_results = SafeToolInvoker.invoke_tool(
                scientific_search, 
                query=current_query
            )
            
            # Then try general search
            general_results = SafeToolInvoker.invoke_tool(
                search_all_sources, 
                query=current_query, 
                pdf_files=pdf_files
            )
            
            # Combine results
            round_results = {
                "round": i+1,
                "query": current_query,
                "scientific_results": science_results,
                "general_results": general_results
            }
            
            # Add to all results
            all_results.append(round_results)
            
            # Skip feedback step on final round
            if i == rounds - 1:
                break
                
            # Generate feedback on search quality and create a new query
            result_summary = f"""
            Scientific search results: {science_results.get('results', [])}
            Web sources: {general_results.get('web_sources', {}).get('results', [])}
            PDF sources: {general_results.get('pdf_sources', {}).get('results', [])}
            """
            
            feedback_prompt = f"""
            You are an expert scientific research assistant helping with an iterative search process.
            
            Original query: {query}
            
            Current round ({i+1}/{rounds}) query: {current_query}
            
            Search results summary:
            {result_summary}
            
            Please analyze these search results and:
            1. Identify what important technical information is still missing
            2. Identify what aspects of the query could be improved to find more specific information
            3. Suggest a completely new reformulation of the query that would help find the missing information
            
            Provide your response in JSON format:
            {{
                "information_gaps": ["list specific technical details that are still missing"],
                "query_limitations": ["issues with the current query"],
                "new_query": "a completely reformulated query to find missing information"
            }}
            """
            
            feedback_response = model.invoke(feedback_prompt)
            
            # Extract the new query from feedback
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'({.*})', feedback_response.content, re.DOTALL)
            if json_match:
                try:
                    feedback_json = json.loads(json_match.group(1))
                    new_query = feedback_json.get('new_query')
                    if new_query:
                        current_query = new_query
                        all_queries.append(new_query)
                    else:
                        # If no new query found, just reuse with small variation
                        current_query = f"Technical details of {query} with focus on implementation and mechanisms"
                        all_queries.append(current_query)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    current_query = f"Alternative perspective on {query} with focus on technical implementation"
                    all_queries.append(current_query)
            else:
                # Fallback if no JSON found
                current_query = f"Different formulation of {query} focusing on scientific principles"
                all_queries.append(current_query)
        
        # Compile final results 
        final_results = {
            "all_queries": all_queries,
            "all_results": all_results,
            "rounds_completed": rounds
        }
        
        return final_results
    except Exception as e:
        logger.error(f"Error in iterative search: {e}")
        return {"error": f"Error in iterative search: {e}", "results": []}

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
            if not pdf_files:
                logger.warning(f"No PDF files found in directory: {pdf_directory}")
        
        # Configure tools with CRAG approach
        # Create tool handlers to avoid direct function calls (which cause deprecation warnings)
        from langchain.tools.base import Tool
        
        tools = [
            retrieve_web_context,
            retrieve_pdf_context, 
            search_all_sources, 
            grade_document_relevance,
            reformulate_query,
            scientific_search,
            iterative_search
        ]
        
        # Create the tool node with proper configuration
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

As an advanced agentic RAG system implementing Corrective-RAG (CRAG) for scientific content, you have access to multiple information sources:
1. Web sources through the retrieve_web_context tool
2. PDF documents through the retrieve_pdf_context tool
3. A combined search across all sources through the search_all_sources tool
4. Document relevance grading through the grade_document_relevance tool
5. Query reformulation through the reformulate_query tool
6. Specialized scientific search through the scientific_search tool
7. Multi-round iterative search through the iterative_search tool

For highly technical scientific topics like quantum physics and single photon sources:
- Use iterative_search as your primary approach - it will automatically conduct multiple rounds of search with different queries
- For single, focused searches on technical content, use scientific_search
- Use multiple reformulations for technical topics
- Examine the key_technical_terms identified in scientific documents

To provide the most comprehensive answer for technical scientific queries:
- For technical scientific content, prioritize using iterative_search first
- Use query reformulation aggressively for technical topics, trying multiple reformulations
- Consider which sources would be most relevant to the technical query
- Use both web and PDF sources together
- Pay attention to the key_technical_terms identified in the documents
- Focus on documents with high relevance scores (when using PDF sources)
- If documents initially returned aren't technical enough, try alternative search strategies including specialized scientific_search
- Specify which source information came from in your response
- Use the search_all_sources tool for an efficient unified search

For quantum physics topics specifically:
- Focus on identifying technical terminology in the documents
- Look for discussions of experimental setups and methodologies
- Pay attention to quantitative data and mathematical formulations
- Extract information about physical processes and mechanisms

Following the enhanced scientific Corrective-RAG approach:
- Grade document relevance to ensure quality scientific information
- Use technical term extraction to identify key concepts
- If initial documents aren't technically detailed enough, try more specialized search approaches
- Synthesize information from multiple technical sources
- When explaining technical concepts, maintain scientific accuracy and depth

IMPORTANT:
- When using retrieve_pdf_context or search_all_sources tools, you MUST provide the pdf_files parameter
- The pdf_files parameter must be a list of file paths, even if there's only one file
- For scientific queries about quantum physics, use the scientific_search tool with specific_domain="quantum_physics"
- For complex technical queries where a single search isn't sufficient, use iterative_search with rounds=3
"""

        # If PDF files are available, add them to the message
        if pdf_files:
            pdf_list = "\n".join([f"- {os.path.basename(pdf)}" for pdf in pdf_files])
            initial_message += f"\n\nThe following PDF files are available:\n{pdf_list}"
            
            # Add the file paths in a format that can be directly used by the agent
            initial_message += f"\n\nHere are the full paths that you can use directly with the tools:\n```python\npdf_files = {pdf_files}\n```"
            initial_message += "\n\nWhen using these PDF files with tools, make sure to pass them as a list, exactly as shown above."
        else:
            initial_message += "\n\nNOTE: No PDF files are available for this query. Please use web sources only."
        
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
        
        if not azure_api_key or not azure_endpoint:
            raise ValueError("Azure OpenAI API key and endpoint are required")
        
        # Get the user's query from command line arguments or use default
        import argparse
        parser = argparse.ArgumentParser(description='Run Agentic RAG with CRAG approach')
        parser.add_argument('--query', type=str, default="What is the main idea of the multiplexed single photon source?", 
                          help='The query to search for')
        parser.add_argument('--pdf_dir', type=str, 
                          default="/Users/bingranyou/Library/Mobile Documents/com~apple~CloudDocs/Downloads/temp/",
                          help='Directory containing PDF files')
        args = parser.parse_args()
        
        # Run the test with a specified PDF directory if it exists
        pdf_dir = args.pdf_dir
        query = args.query
        
        if os.path.exists(pdf_dir):
            # Find PDF files first to make sure they exist
            pdf_files = discover_pdf_files(pdf_dir)
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
                try:
                    result = agentic_rag(
                        pdf_directory=pdf_dir, 
                        query=query
                    )
                    print(f"\nSuccess! Response generated for query: '{query}'")
                except Exception as e:
                    logger.error(f"Error in agentic_rag: {e}")
                    print(f"\nError running agentic_rag: {e}")
            else:
                logger.error(f"No PDF files found in {pdf_dir}")
                print(f"\nNo PDF files found in directory: {pdf_dir}")
                print("Running query using web sources only...")
                try:
                    result = agentic_rag(query=query)
                    print(f"\nSuccess! Response generated for query: '{query}' using web sources only")
                except Exception as e:
                    logger.error(f"Error in agentic_rag with web sources only: {e}")
                    print(f"\nError running agentic_rag with web sources only: {e}")
        else:
            logger.error(f"PDF directory {pdf_dir} does not exist")
            print(f"\nPDF directory does not exist: {pdf_dir}")
            print("Running query using web sources only...")
            try:
                result = agentic_rag(query=query)
                print(f"\nSuccess! Response generated for query: '{query}' using web sources only")
            except Exception as e:
                logger.error(f"Error in agentic_rag with web sources only: {e}")
                print(f"\nError running agentic_rag with web sources only: {e}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\nTest failed with error: {e}")
        
    print("\nDone!")
