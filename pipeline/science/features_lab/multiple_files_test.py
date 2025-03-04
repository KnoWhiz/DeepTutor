import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os

st.title("📄 Multi-PDF Viewer")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Create a file selector in the sidebar
    st.sidebar.header("📑 Select PDF to View")
    selected_pdf_name = st.sidebar.selectbox(
        "Choose a PDF",
        [file.name for file in uploaded_files]
    )
    
    # Find the selected PDF file object
    selected_pdf = next(file for file in uploaded_files if file.name == selected_pdf_name)
    
    # Save the selected file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(selected_pdf.read())
        temp_pdf_path = temp_file.name

    try:
        # Open the PDF and get total pages
        doc = fitz.open(temp_pdf_path)
        total_pages = len(doc)
        
        # Add page navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        
        # Initialize current page in session state if not exists
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
            
        with col1:
            if st.button("← Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
            # Add a slider for quick page navigation
            new_page = st.slider("Jump to page", 1, total_pages, st.session_state.current_page)
            if new_page != st.session_state.current_page:
                st.session_state.current_page = new_page
                
        with col3:
            if st.button("Next →") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1

        # Display the current page
        current_page = doc[st.session_state.current_page - 1]
        img = current_page.get_pixmap()
        
        # Create a container for the PDF view with fixed height
        pdf_container = st.container()
        with pdf_container:
            st.image(img.tobytes(), caption=f"Page {st.session_state.current_page}", use_container_width=True)
            
        # Option to download the current PDF
        with open(temp_pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Current PDF",
                data=f,
                file_name=selected_pdf_name,
                mime="application/pdf"
            )
            
        # Clean up
        doc.close()
        os.unlink(temp_pdf_path)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")