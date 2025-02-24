# DeepTutor

## Webapp Demo

Please feel free to try out our deployed webapp here and feel free to leave any comments!

https://deeptutor.streamlit.app/

## Installation

Create environment and install required packages

```bash
conda create --name deeptutor python=3.12
conda activate deeptutor
pip install -r requirements.txt
# Install AgentChat and OpenAI client from Extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
# Install AutoGen Studio for no-code GUI
pip install -U "autogenstudio"
```

## Set OPENAI_API_KEY

If use OpenAI API

```bash
cd DeepTutor
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > .env
```

If use Azure OpenAI API
```bash
cd DeepTutor
# Should replace xxx to a real Azure openai api key and endpoint url
echo "AZURE_OPENAI_API_KEY=xxx" > .env
echo "AZURE_OPENAI_ENDPOINT=xxx" > .env
```

## Run Native

Run the streamlit app via

```bash
python -m streamlit run tutor.py
```

## Commone errors

1. According to [PyMuPDF Documentation](https://pymupdf.readthedocs.io/en/latest/installation.html#option-2-install-from-binaries) you need to download a wheel file that is specific to your platform (e.g windows, mac, linux). The wheel files can be found on [PyMuPDF files](https://pypi.org/project/PyMuPDF/#files).

2. Make sure to check the correct version of your python running on your system ```python -V```

3. Once downloaded place it at the root directory of your project.

4. Then run ```pip install PyMuPDF-<...>.whl``` replace ```PyMuPDF-<...>.whl``` with the name of the wheel file you have downloaded in (1) above.

5. Now import fitz should be available in your module.
