import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings

# Create output directory
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Initialize converter and process PDF
converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("/Users/bingran_you/Library/Mobile Documents/com~apple~CloudDocs/Downloads/papers/science.1189075.pdf")
text, _, images = text_from_rendered(rendered)

# Save images
if images:
    print(f"Saving {len(images)} images to {output_dir}")
    for img_name, img in images.items():
        try:
            # Create a valid filename from the image name
            safe_filename = "".join(c for c in img_name if c.isalnum() or c in ('-', '_', '.'))
            output_path = os.path.join(output_dir, f"{safe_filename}.{settings.OUTPUT_IMAGE_FORMAT}")
            
            # Save the image
            img.save(output_path)
            print(f"Saved image: {output_path}")
        except Exception as e:
            print(f"Error saving image {img_name}: {str(e)}")
else:
    print("No images found in the PDF")