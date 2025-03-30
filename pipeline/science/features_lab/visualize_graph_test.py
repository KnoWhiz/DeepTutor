import os
import sys
import logging
from pathlib import Path
import tempfile

# Add the project root to the Python path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent.parent
sys.path.append(str(project_root))
print(f"Added {project_root} to Python path")

# Import langgraph modules
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualize_graph_test")

# Define state class for the graph
class State(TypedDict):
    count: int
    message: str

# Define node functions
def start_node(state: State) -> State:
    """Initial node that starts the processing"""
    return {"count": state.get("count", 0) + 1, "message": "Started processing"}

def process_a(state: State) -> State:
    """Process A node"""
    return {"count": state.get("count", 0) + 1, "message": "Processed by A"}

def process_b(state: State) -> State:
    """Process B node"""
    return {"count": state.get("count", 0) + 1, "message": "Processed by B"}

def process_c(state: State) -> State:
    """Process C node - final step"""
    return {"count": state.get("count", 0) + 1, "message": "Finished processing"}

def check_count(state: State) -> str:
    """Conditional routing based on count"""
    if state["count"] > 2:
        return "process_c"  # Skip process_b and go straight to C if count > 2
    return "process_b"      # Otherwise follow normal path to B

# Function to build a simple graph for testing
def build_simple_graph():
    """Build a simple graph with a linear flow and conditional branches"""
    # Initialize the graph
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("start", start_node)
    builder.add_node("process_a", process_a)
    builder.add_node("process_b", process_b)
    builder.add_node("process_c", process_c)
    
    # Add regular edges
    builder.add_edge("start", "process_a")
    
    # Add conditional edge from process_a
    builder.add_conditional_edges(
        "process_a",
        check_count,
        {
            "process_b": "process_b",
            "process_c": "process_c"
        }
    )
    
    # Add remaining edges
    builder.add_edge("process_b", "process_c")
    
    # Set entry and end points
    builder.set_entry_point("start")
    builder.set_finish_point("process_c")
    
    # Compile the graph
    return builder.compile()

# Custom visualization functions
def visualize_graph_structure(graph):
    """
    Visualize the graph structure in a simple text format
    """
    # Get the graph object
    g = graph.get_graph()
    
    # Get nodes and edges
    nodes = g.nodes
    edges = g.edges
    
    print("\nGraph Structure:\n-----------------")
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")
    
    print("\nNodes:")
    for node in sorted(nodes):
        print(f"  - {node}")
    
    print("\nEdges:")
    try:
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                source, target = edge
                print(f"  - {source} -> {target}")
            else:
                print(f"  - {edge} (format: {type(edge)})")
    except Exception as e:
        print(f"Error processing edges: {e}")
        print(f"Edge format: {type(edges)}")
        if hasattr(edges, '__iter__'):
            for i, edge in enumerate(edges):
                print(f"  Edge {i}: {edge} (type: {type(edge)})")

def generate_mermaid_diagram(graph):
    """
    Generate a Mermaid diagram code for the graph
    """
    g = graph.get_graph()
    nodes = g.nodes
    edges = g.edges
    
    mermaid_code = ["graph TD;"]
    
    # Add nodes
    for node in sorted(nodes):
        node_id = node.replace(".", "_").replace(" ", "_")
        if node == "START" or node == "__start__":
            mermaid_code.append(f'    {node_id}["{node}"]:::startClass;')
        elif node == "END" or node == "__end__":
            mermaid_code.append(f'    {node_id}["{node}"]:::endClass;')
        else:
            mermaid_code.append(f'    {node_id}["{node}"]:::nodeClass;')
    
    # Add edges - properly handle Edge class
    try:
        conditional_sources = set()
        for edge in edges:
            # Handle Edge class format
            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                source = edge.source
                target = edge.target
                source_id = str(source).replace(".", "_").replace(" ", "_")
                target_id = str(target).replace(".", "_").replace(" ", "_")
                
                # Keep track of conditional sources
                if hasattr(edge, 'conditional') and edge.conditional:
                    conditional_sources.add(source)
                    edge_style = " -.-> "
                    
                    # Try to add a label if possible
                    if source == "process_a" and target == "process_b":
                        # For our specific test case
                        label = "count <= 2"
                        mermaid_code.append(f'    {source_id}{edge_style}|"{label}"| {target_id};')
                    elif source == "process_a" and target == "process_c":
                        # For our specific test case
                        label = "count > 2"
                        mermaid_code.append(f'    {source_id}{edge_style}|"{label}"| {target_id};')
                    else:
                        mermaid_code.append(f'    {source_id}{edge_style}{target_id};')
                else:
                    edge_style = " --> "
                    mermaid_code.append(f'    {source_id}{edge_style}{target_id};')
    except Exception as e:
        mermaid_code.append(f'    // Error processing edges: {e}')
    
    # Add style definitions
    mermaid_code.append('    classDef startClass fill:#ffdfba;')
    mermaid_code.append('    classDef endClass fill:#baffc9;')
    mermaid_code.append('    classDef nodeClass fill:#fad7de;')
    
    return "\n".join(mermaid_code)

def save_mermaid_to_file(mermaid_code, output_path):
    """
    Save Mermaid code to a file
    """
    with open(output_path, 'w') as f:
        f.write(mermaid_code)
    return output_path

# Main test function to demonstrate visualization capabilities
def test_graph_visualization():
    try:
        # Create a simple graph
        logger.info("Building simple graph...")
        graph = build_simple_graph()
        
        # Run the graph to see what happens - first with count=0
        logger.info("Running the graph with count=0...")
        final_state_1 = graph.invoke({"count": 0, "message": "Initial state"})
        print(f"Final state (count=0): {final_state_1}")
        print(f"Path taken: start -> process_a -> process_b -> process_c (Normal path)")
        
        # Run again with count=3 to take different path
        logger.info("Running the graph with count=3...")
        final_state_2 = graph.invoke({"count": 3, "message": "Initial state with higher count"})
        print(f"Final state (count=3): {final_state_2}")
        print(f"Path taken: start -> process_a -> process_c (Skipped process_b)")
        
        # 1. Visualize the graph structure
        logger.info("Visualizing graph structure...")
        visualize_graph_structure(graph)
        
        # 2. Generate and save Mermaid diagram code
        logger.info("Generating Mermaid diagram...")
        mermaid_code = generate_mermaid_diagram(graph)
        print("\nMermaid Diagram Code:")
        print(mermaid_code)
        
        # 3. Save the Mermaid code to a file
        mermaid_file_path = os.path.join(tempfile.gettempdir(), "graph_diagram.mmd")
        save_mermaid_to_file(mermaid_code, mermaid_file_path)
        logger.info(f"Mermaid diagram code saved to: {mermaid_file_path}")
        
        # 4. Display the content of the saved Mermaid file
        print("\nContents of the Mermaid file:")
        with open(mermaid_file_path, 'r') as f:
            print(f.read())
        
        # Save instructions for visualization
        instructions = f"""
To visualize this graph:

1. Use the Mermaid diagram saved at: {mermaid_file_path}
2. Copy the contents of this file and paste into one of:
   - Mermaid Live Editor: https://mermaid.live/
   - VS Code with Mermaid extension
   - Any Markdown renderer that supports Mermaid (like GitHub)
   - Or use the Mermaid CLI tool if installed

3. For programmatic visualization in Python projects:
   - Install additional packages: pip install graphviz pyppeteer
   - Check LangGraph documentation for updated visualization methods
        """
        
        print(instructions)
        
        return "Graph visualization test completed successfully"
        
    except Exception as e:
        logger.error(f"Error in test_graph_visualization: {e}")
        raise

if __name__ == "__main__":
    try:
        result = test_graph_visualization()
        print(result)
    except Exception as e:
        logger.error(f"Test failed: {e}")
