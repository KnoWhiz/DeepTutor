import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
load_dotenv()

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import unittest

class TestImageUnderstanding(unittest.TestCase):
    def setUp(self):
        # Initialize Azure OpenAI client
        self.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = os.getenv('AZURE_OPENAI_KEY')
        self.deployment_name = 'gpt-4o'
        self.api_version = '2024-06-01'
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}openai/deployments/{self.deployment_name}"
        )
        
    def test_image_understanding(self):
        # Test image URL - replace with your actual test image URL
        image_url = "https://knowhiztutorrag.blob.core.windows.net/knowhiztutorrag/file_appendix/3671da1e844b53ffbdccac7bc8c57341/images/_page_1_Figure_1.jpeg"
        
        try:
            # Create messages for the vision model
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant capable of analyzing images. Please describe what you see in detail, including main subjects/objects, colors and composition, any text visible in the image, and overall context or setting."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this picture:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=2000
            )
            
            # Assertions to verify the response
            self.assertIsNotNone(response)
            self.assertTrue(hasattr(response.choices[0].message, 'content'))
            self.assertTrue(len(response.choices[0].message.content) > 0)
            
            # Print the response for manual verification
            print("\nImage Analysis Result:")
            print(response.choices[0].message.content)
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main()
