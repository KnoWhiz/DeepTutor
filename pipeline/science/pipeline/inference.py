import os
import openai
import logging
from dotenv import load_dotenv
from typing import Optional
from pipeline.science.pipeline.config import load_config
load_dotenv()

logger = logging.getLogger("tutorpipeline.science.inference")

def deepseek_inference(
    prompt: str,
    system_message: str = "You are a deep thinking researcher reading a paper. If you don't know the answer, say you don't know.",
    stream: bool = False,
    temperature: float = 0.4,
    top_p: float = 0.1,
    max_tokens: int = 3000,
    model: str = "DeepSeek-R1-Distill-Llama-70B"
) -> Optional[str]:
    """
    Get completion from the DeepSeek model with optional streaming support.

    Args:
        prompt: The user's input prompt
        system_message: The system message to set the AI's behavior
        stream: Whether to stream the output or not
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate

    Returns:
        The generated text if streaming is False, None if streaming is True
    """
    config = load_config()
    max_tokens = config["inference_token_limit"]
    if model == "DeepSeek-R1-Distill-Llama-70B":
        model = "DeepSeek-R1-Distill-Llama-70B"
        base_url = "https://api.sambanova.ai/v1"
        max_tokens *= 3
    elif model == "DeepSeek-R1":
        model = "DeepSeek-R1"
        base_url = "https://preview.snova.ai/v1"
        max_tokens *= 1
    else:
        model = "DeepSeek-R1-Distill-Llama-70B"
        base_url = "https://api.sambanova.ai/v1"
        max_tokens *= 1

    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url=base_url
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream
        )

        if stream:
            # Process the streaming response
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # Add a newline at the end
            return None
        else:
            # Return the complete response
            return response.choices[0].message.content

    except openai.APIError as e:
        logger.exception(f"API Error: {str(e)}")
        return None
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Example with streaming
    print("Streaming response:")
    deepseek_inference("what is 1+1?", stream=True)

    print("\nNon-streaming response:")
    response = deepseek_inference("what is 1+1?", stream=False)
    if response:
        print(response)