import os
from dotenv import load_dotenv
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def main():
    # Set your SambaNova Cloud API key
    os.environ["SAMBANOVA_API_KEY"] = os.getenv("SAMBANOVA_API_KEY")

    # Instantiate the ChatSambaNovaCloud model with desired parameters
    llm = ChatSambaNovaCloud(
        model="Meta-Llama-3.3-70B-Instruct",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.01,
    )

    # === Example 1: Direct invocation with predefined messages ===
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    response = llm.invoke(messages)
    print("Translation 1:", response.content)

    # === Example 2: Using ChatPromptTemplate for dynamic input ===
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )

    # Create a chain by combining the prompt with the LLM
    chain = prompt | llm

    response2 = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    print("Translation 2:", response2.content)

if __name__ == "__main__":
    main()