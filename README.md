# KnoWhizTutor

## Installation

Create environment and install required packages

```bash
conda create --name knowhiztutor python=3.12
conda activate knowhiztutor
pip install -r requirements.txt
```

## Set OPENAI_API_KEY

If use OpenAI API

```bash
cd KnoWhizTutor
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > .env
```

If use Azure OpenAI API
```bash
cd KnoWhizTutor
# Should replace xxx to a real Azure openai api key and endpoint url
echo "AZURE_OPENAI_API_KEY=xxx" > .env
echo "AZURE_OPENAI_ENDPOINT=xxx" > .env
```

## Run Native

Run the streamlit app via

```bash
streamlit run tutor.py
```