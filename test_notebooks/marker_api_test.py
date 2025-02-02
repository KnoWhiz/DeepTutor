import os
import time
import requests
import base64
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
API_KEY = os.getenv("MARKER_API_KEY")
# print(API_KEY)
INPUT_FILE = "/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/science.1189075.pdf"  # Replace with your input file path
API_URL = "https://www.datalab.to/api/v1/marker"

# --- Submit the file ---
with open(INPUT_FILE, "rb") as f:
    form_data = {
        # The file parameter: (filename, file-object, mimetype)
        "file": (INPUT_FILE, f, "application/pdf"),
        # Optional parameters:
        "langs": (None, "English"),
        "force_ocr": (None, False),
        "paginate": (None, False),
        "output_format": (None, "markdown"),
        "use_llm": (None, False),
        "strip_existing_ocr": (None, False),
        "disable_image_extraction": (None, False),
    }
    headers = {"X-Api-Key": API_KEY}
    response = requests.post(API_URL, files=form_data, headers=headers)

# Check initial response and get the URL to poll for results
data = response.json()
if not data.get("success"):
    raise Exception(f"Request failed immediately: {data.get('error')}")

request_check_url = data.get("request_check_url")
print("Submitted request. Polling for results at:")
print(request_check_url)

# --- Polling until processing is complete ---
max_polls = 300  # maximum number of polls
poll_interval = 2  # seconds between polls

result = None
for i in range(max_polls):
    time.sleep(poll_interval)
    poll_response = requests.get(request_check_url, headers=headers)
    result = poll_response.json()
    status = result.get("status")
    print(f"Poll {i+1}: status = {status}")
    if status == "complete":
        break
else:
    raise Exception("The request did not complete within the expected time.")

# --- Process the final result ---
if result.get("success"):
    # Save the returned markdown
    markdown = result.get("markdown", "")
    with open("output.md", "w", encoding="utf-8") as md_file:
        md_file.write(markdown)
    print("Markdown saved to output.md")

    # Save images if any (images is a dict with filename keys and base64 encoded data)
    images = result.get("images", {})
    if images:
        for filename, b64data in images.items():
            try:
                image_data = base64.b64decode(b64data)
                with open(filename, "wb") as img_file:
                    img_file.write(image_data)
                print(f"Saved image: {filename}")
            except Exception as e:
                print(f"Error saving image {filename}: {e}")
    else:
        print("No images were returned with the result.")
else:
    print("The request completed with an error:")
    print(result.get("error"))