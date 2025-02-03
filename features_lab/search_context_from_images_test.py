import os
import json
import sys

def extract_image_context(folder_dir):
    """
    Scan the given folder for image files and a markdown file.
    For each image file, search the markdown file for lines that mention the image filename.
    Save the results as a JSON file in the same folder.
    
    Parameters:
        folder_dir (str): The path to the folder containing image files and one markdown file.
    
    Output:
        A JSON file named 'image_context.json' is written to the folder.
        The JSON contains a dictionary mapping image filenames to a list of context strings (lines from the markdown file).
    """
    # Define the image file extensions we care about.
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg')

    # List all image files in the folder (case-insensitive match).
    image_files = [f for f in os.listdir(folder_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        print("No image files found in the folder.")
        return

    # Find the markdown file (assuming there's only one .md file).
    md_files = [f for f in os.listdir(folder_dir) if f.lower().endswith('.md')]
    if not md_files:
        print("No markdown file found in the folder.")
        return
    # Use the first markdown file found.
    md_file = md_files[0]
    md_path = os.path.join(folder_dir, md_file)
    
    # Read the content of the markdown file.
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Split the markdown content into lines.
    md_lines = md_content.splitlines()
    
    # Create a dictionary to store image filename vs. list of context lines.
    image_context = {}
    for image in image_files:
        # Find all lines in the markdown file that mention the image filename.
        contexts = [line.strip() for line in md_lines if image in line]
        image_context[image] = contexts
    
    # Write the dictionary to a JSON file in the same folder.
    output_path = os.path.join(folder_dir, 'image_context.json')
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(image_context, outfile, indent=2)
    
    print(f"Image context data saved to: {output_path}")

if __name__ == "__main__":
    # Check if a folder path has been provided as a command-line argument.
    folder_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/embedded_content/16005aaa19145334b5605c6bf61661a0/markdown/"
    extract_image_context(folder_dir)