# GraphRAG Per-Page Chunking Implementation

## Overview

This implementation modifies GraphRAG's embedding process to use per-page chunking instead of fixed chunk sizes. Each page in the document is now treated as its own chunk, allowing for more natural content boundaries and better preservation of page-level context.

## Problem Statement

Previously, GraphRAG used a fixed chunk size (1200 characters) regardless of the actual content length of each page. This approach could:
- Split important content across chunk boundaries
- Create chunks that don't align with natural page boundaries
- Lose page-level context and structure

## Solution

The implementation changes the input file structure for GraphRAG processing:

### 1. Modified File Structure
Instead of saving one large text file, the system now saves each page as a separate file:
```
GraphRAG/input/
├── abc12345_page_001.txt  # Page 1 content
├── abc12345_page_002.txt  # Page 2 content
├── abc12345_page_003.txt  # Page 3 content
└── ...
```

### 2. Updated Configuration
Modified `graphrag_settings.yaml`:
```yaml
input:
  file_pattern: ".*_page_\\d+\\.txt$"  # Match per-page files

chunks:
  size: 0  # Disable fixed chunking since each file is already a page
  overlap: 0  # No overlap needed for per-page chunks
```

### 3. Enhanced File Processing
Updated `save_file_txt_locally()` function in `doc_processor.py`:
- Splits markdown content by page separators (`---`)
- Saves each page as a separate `.txt` file
- Maintains page numbering and metadata
- Creates enhanced mapping file with chunking strategy information

## Files Modified

### 1. `pipeline/science/pipeline/doc_processor.py`
**Function**: `save_file_txt_locally()`
**Changes**:
- Split markdown content into pages using `---` separators
- Save each page as a separate file with naming pattern: `{hash}_page_{num:03d}.txt`
- Enhanced mapping file to track page structure and chunking strategy
- Fallback handling for documents without clear page separators

### 2. `pipeline/science/pipeline/graphrag_settings.yaml`
**Changes**:
- Updated `file_pattern` to match per-page files: `".*_page_\\d+\\.txt$"`
- Set `chunks.size` to 0 to disable fixed chunking
- Set `chunks.overlap` to 0 since each file is already a complete page

## Benefits

### 1. Natural Content Boundaries
- Each chunk corresponds to a natural page boundary
- Preserves page-level context and structure
- Maintains document flow and organization

### 2. Improved Retrieval Quality
- GraphRAG can now work with complete page content
- Better preservation of semantic relationships within pages
- More accurate entity extraction and knowledge graph building

### 3. Flexible Content Handling
- Short pages remain as single chunks
- Long pages are processed as complete units
- No artificial splitting of related content

### 4. Enhanced Metadata
- Page-level tracking and mapping
- Better source attribution in responses
- Improved debugging and analysis capabilities

## Usage

The implementation is automatically applied when using `ChatMode.ADVANCED` (GraphRAG mode). No changes are needed in the calling code:

```python
# The existing code continues to work as before
if _mode == ChatMode.ADVANCED:
    save_file_txt_locally(file_path, filename=file_id[:8], embedding_folder=embedding_folder)
    async for chunk in generate_GraphRAG_embedding(embedding_folder, time_tracking):
        yield chunk
```

## Testing

A comprehensive test script is provided: `pipeline/science/features_lab/test_graphrag_per_page_chunking.py`

**Test Coverage**:
- PDF processing with multiple pages
- Markdown processing with page separators
- File structure validation
- Mapping file verification

**Run Tests**:
```bash
cd pipeline/science/features_lab
python test_graphrag_per_page_chunking.py
```

## File Structure Example

After processing a document, the structure will be:

```
embeddings/
├── GraphRAG/
│   ├── input/
│   │   ├── abc12345_page_001.txt
│   │   ├── abc12345_page_002.txt
│   │   └── abc12345_page_003.txt
│   ├── filename_mapping.json
│   └── ...
└── markdown/
    └── abc12345.md
```

**Mapping File Content**:
```json
{
  "abc12345": {
    "original_name": "document.pdf",
    "total_pages": 3,
    "chunking_strategy": "per_page"
  }
}
```

## Backward Compatibility

The implementation maintains backward compatibility:
- Existing GraphRAG embeddings continue to work
- No changes required in query/response code
- Gradual migration as new documents are processed

## Future Enhancements

1. **Dynamic Page Detection**: Improve page boundary detection for various document formats
2. **Content-Aware Splitting**: Implement intelligent splitting for very long pages
3. **Metadata Enhancement**: Add more detailed page-level metadata
4. **Performance Optimization**: Optimize file I/O for large documents

## Monitoring and Debugging

The implementation includes comprehensive logging:
- Page count and file creation tracking
- Content length statistics per page
- Mapping file updates
- Error handling and fallback mechanisms

## Conclusion

This implementation successfully transforms GraphRAG's chunking strategy from fixed-size chunks to per-page chunks, providing better content preservation and more natural document processing while maintaining full compatibility with existing GraphRAG functionality. 