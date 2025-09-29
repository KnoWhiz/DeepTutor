import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    get_llm,
    count_tokens,
    replace_latex_formulas,
    Question
)
from pipeline.science.pipeline.doc_processor import (
    process_pdf_file
)
from pipeline.science.pipeline.content_translator import (
    detect_language
)
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.images_understanding import (
    aggregate_image_contexts_to_urls, 
    create_image_context_embeddings_db, 
    analyze_image
)
from pipeline.science.pipeline.rag_agent import get_rag_context
from pipeline.science.pipeline.inference import stream_response_with_tags
# from pipeline.science.pipeline.claude_code_sdk import get_claude_code_response, get_claude_code_response_async
from dotenv import load_dotenv
load_dotenv()
import logging
logger = logging.getLogger("tutorpipeline.science.get_response")


async def get_multiple_files_summary(file_path_list, embedding_folder_list, chat_session=None, stream=False):
    """
    Generate a summary for multiple files by combining previews of each file.
    
    Args:
        file_path_list: List of file paths to generate a summary for
        embedding_folder_list: List of embedding folders
        chat_session: The current chat session for tracking responses
        stream: Whether to stream the response
        
    Returns:
        An async generator yielding the summary if stream=True, otherwise a string
    """
    # Load config and LLM
    config = load_config()
    llm = get_llm('advanced', config['llm'])

    # If the number of files is more than summary_file_limit, just reply "Hi, I'm DeepTutor. What can I help you with?"
    if len(file_path_list) > config["summary_file_limit"]:
        if stream:
            async def process_stream_async():
                yield "<response>\n\n"
                yield "Hi, I'm DeepTutor. What can I help you with?"
                yield "\n\n</response>"
            return process_stream_async()
        else:
            return "<response>\n\nHi, I'm DeepTutor. What can I help you with?\n\n</response>"
    
    # Log the list of files being processed
    logger.info(f"Processing multiple files for summary: {file_path_list}")
    logger.info(f"Using embedding folders: {embedding_folder_list}")
    
    # Extract first 3000 tokens from each file
    file_previews = []
    for i, file_path in enumerate(file_path_list):
        try:
            logger.info(f"Processing file {i+1}/{len(file_path_list)}: {file_path}")
            # Process the PDF file properly
            document, doc = process_pdf_file(file_path)
            logger.info(f"File {os.path.basename(file_path)} processed, document has {len(document)} pages")
            
            # Extract text from the document
            file_content = ""
            for page_doc in document:
                if hasattr(page_doc, 'page_content') and page_doc.page_content:
                    file_content += page_doc.page_content.strip() + "\n"
            
            # # Calculate token limit per file - maximum 3000 tokens per file but adjust for file count
            # token_limit = min(3000, 10000 // len(file_path_list))
            token_limit = int(config["basic_token_limit"] * 0.1 // len(file_path_list))
            
            # Get total tokens in the content
            try:
                total_tokens = count_tokens(file_content)
                logger.info(f"File {file_path} has {total_tokens} tokens, limiting to {token_limit}")
                
                # Log the first chunk of content (up to 200 chars)
                first_content_preview = file_content[:200].replace("\n", " ") + "..."
                logger.info(f"First content chunk for {os.path.basename(file_path)}: {first_content_preview}")
                
                # Truncate to token limit
                if total_tokens > token_limit:
                    # Take approximately the first X tokens (by characters, not exact)
                    char_limit = int(len(file_content) * (token_limit / total_tokens))
                    truncated_content = file_content[:char_limit]
                    truncated_content += "\n\n[Content truncated due to length...]"
                    logger.info(f"File {os.path.basename(file_path)} truncated from {len(file_content)} to {len(truncated_content)} characters")
                else:
                    truncated_content = file_content
                    logger.info(f"File {os.path.basename(file_path)} content used in full ({len(file_content)} characters)")
            except Exception as e:
                logger.exception(f"Error calculating tokens for {file_path}: {str(e)}")
                # Fallback to character-based approximation (roughly 4 chars per token)
                char_limit = token_limit * 4
                if len(file_content) > char_limit:
                    truncated_content = file_content[:char_limit]
                    truncated_content += "\n\n[Content truncated due to length...]"
                    logger.info(f"Using character-based fallback: truncated {os.path.basename(file_path)} to {char_limit} characters")
                else:
                    truncated_content = file_content
                    logger.info(f"Using character-based fallback: using full content of {os.path.basename(file_path)}")
            
            file_previews.append((os.path.basename(file_path), truncated_content))
            logger.info(f"Extracted preview from {os.path.basename(file_path)}: {len(truncated_content)} characters")
        except Exception as e:
            logger.exception(f"Error extracting content from {file_path}: {str(e)}")
            file_name = os.path.basename(file_path)
            file_previews.append((file_name, f"Error extracting content: {str(e)}"))
    
    # Format the prompt with file previews
    prompt_parts = []
    for i, (file_name, preview) in enumerate(file_previews):
        prompt_parts.append(f"\n\n### DOCUMENT {i+1}: {file_name}\n\nPreview Content:\n```\n{preview}\n```\n")
    
    formatted_previews = "\n".join(prompt_parts)
    logger.info(f"Created formatted previews for {len(file_previews)} files, total length: {len(formatted_previews)} characters")
    
    # Create proper ChatPromptTemplate with system and user messages
    system_prompt = """You are an expert academic tutor helping a student understand multiple documents. 
The student has loaded multiple PDF files and needs a comprehensive summary that explains what each document is about.

Please provide a comprehensive summary that:
1. Introduces each document with its title (derived from content if possible) and main topic
2. Summarizes the key content and main findings of each document
3. Identifies relationships or connections between the documents (they appear to be related scientific papers)
4. Highlights the most important concepts across all documents
5. Uses markdown formatting for clear organization with sections and subsections
6. Makes appropriate use of bold, bullet points, and other formatting to improve readability
7. Highest title level is 3, and the title should be concise and informative.

Format your summary with a friendly welcome message at the beginning and a closing "Ask me anything" message at the end."""

    user_prompt = """Here are the files with previews of their content:
{formatted_previews}"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    
    logger.info(f"Generated summary prompt with length: {len(system_prompt + user_prompt)} characters")
    logger.info(f"Generating summary for multiple files: {[os.path.basename(fp) for fp in file_path_list]}")
    
    if stream:
        # Stream response for real-time feedback - remove thinking part
        logger.info("Using streaming mode for summary generation")
        chain = prompt_template | llm
        answer = chain.stream({"formatted_previews": formatted_previews})

        async def process_stream_async():
            yield "<response>\n\n"
            for chunk in answer:
                # Convert AIMessageChunk to string
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)
            yield "\n\n</response>"
            logger.info("Completed streaming summary generation")

        return process_stream_async()
    else:
        # Return complete response at once - remove thinking part
        logger.info("Using non-streaming mode for summary generation")
        chain = prompt_template | llm
        response = chain.invoke({"formatted_previews": formatted_previews})
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Generated summary with length: {len(response_text)} characters")
        return f"<response>\n\n{response_text}\n\n</response>"


async def get_response(chat_session: ChatSession, file_path_list, question: Question, chat_history, embedding_folder_list, deep_thinking = True, stream=True):
    config = load_config()
    user_input = question.text
    user_input_string = str(user_input + "\n\n" + question.special_context)
    
    # Check if this is a summary request for multiple files. If so, return a generator from get_multiple_files_summary
    if len(file_path_list) > 1 and user_input == config["summary_wording"]:
        logger.info("Handling multiple files summary request.")
        return await get_multiple_files_summary(file_path_list, embedding_folder_list, chat_session, stream)
    
    # Handle Lite mode first
    if chat_session.mode == ChatMode.LITE or chat_session.mode == ChatMode.BASIC or chat_session.mode == ChatMode.ADVANCED:
        config = load_config()
        token_limit = config["inference_token_limit"]
        map_symbol_to_index = config["map_symbol_to_index"]
        # Get the first 3 keys from map_symbol_to_index for examples in the prompt
        first_keys = list(map_symbol_to_index.keys())[:3]
        example_keys = ", or ".join(first_keys)
        logger.info(f"embedding_folder_list in get_response: {embedding_folder_list}")
        await get_rag_context(chat_session=chat_session,
                            file_path_list=file_path_list,
                            question=question,
                            chat_history=chat_history,
                            embedding_folder_list=embedding_folder_list,
                            deep_thinking=deep_thinking,
                            stream=stream,
                            context="")
        formatted_context_string = chat_session.formatted_context
        # Create proper ChatPromptTemplate with system and user messages
        system_prompt = """You are a deep thinking tutor helping a student reading a paper.

MATH RENDERING — HARD RULES (must follow):
- Wrap ALL math (include important numbers) in $...$ (inline) or $$...$$ (display). Never write bare math.
- Do NOT use \( \) or \[ \]; only $...$ or $$...$$.
- Do NOT put math in backticks. Backticks are for code only.
- Balance every $ and $$ pair.
- In display math, keep the entire expression inside a single $$...$$ block.
- For units and symbols, use LaTeX: e.g., $10\,\mathrm{{MHz}}$, $\mu$, $\Omega$, $\mathbf{{x}}$, $x_i$.

RESPONSE GUIDELINES:
0. **TL;DR:** Start with 1–2 sentences that directly answer the question.
1. Provide concise, accurate answers directly addressing the question.
2. Use clear, precise language with appropriate technical terminology.
3. Format key concepts with **bold**.
4. Maintain a professional, academic tone.
5. Break down complex information into structured, logical segments.
6. When explaining technical concepts, include relevant examples or applications.
7. State limitations/uncertainty clearly.
8. Use bullet points or numbered lists for sequences.
9. Unless clearly specified the output language: If the user's question is in Chinese, then answer in Chinese. But for the source citation in square brackets, ALWAYS use the same language as the original source. If the user's question is not in Chinese, then answer in English (For citations in square brackets, still use the same language as the original source). Do not use other languages.

SOURCING MODES
Case 1 (Answerable from context chunks):
  - Use only the context. For *each sentence* in the response, cite the most relevant chunk key(s) in the format "[<1>]" or "[<1>][<3>]" at the end of the sentence.
  - Immediately after each citation key, append one sentence from the source (IMPORTANT: italic, in quotes) inside square brackets, e.g., ["_...source sentence..._"]. IMPORTANT: Use the same language as the original source!
  - Use markdown emphasis for readability.

Case 2 (Not answerable from context):
  - State clearly that you are using your own knowledge.
  - Keep the same math and formatting rules.

SELF-CHECK BEFORE SENDING (must pass all):
- [Math-1] No visible math outside $...$/$$...$$.
- [Math-2] All $ and $$ are balanced.
- [Math-3] No \(\), \[\], or backticked math; no mixed currency $ mistaken for math.
- [Source-1] In Case 1, every sentence ends with correct [<k>] citations + the required one-sentence italic source extract.
- [Tone-1] **TL;DR** present; academic tone maintained.

────────────────────────────────────────────────
GOOD EXAMPLES (follow exactly)
────────────────────────────────────────────────

GOOD A — Inline math, Case 1 with citations
User Q: "What is the relation between energy and frequency for a photon?"
Context Chunks:
  [<1>]: "Planck's relation states E = ħω for a single photon."
  [<2>]: "Angular frequency ω relates to frequency f by ω = 2πf."

Assistant (Case 1):
**TL;DR:** The photon's energy is proportional to its angular frequency via $E=\hbar\omega$. [<1>] ["_Planck's relation states E = ħω for a single photon._"]
**Planck relation.** The energy of a photon is $E=\hbar\omega$. [<1>] ["_Planck’s relation states E = ħω for a single photon._"]  
**Frequency form.** Using $\omega=2\pi f$, we also have $E=h f$ with $h=2\pi\hbar$. [<2>][<1>] ["_Angular frequency ω relates to frequency f by ω = 2πf._"]["_Planck's relation states E = ħω for a single photon._"]

GOOD B — Display math, multi-step, Case 2 (own knowledge)
User Q: "Show the variance of a Bernoulli($p$) variable."
Assistant (Case 2):
**TL;DR:** For $X\sim\mathrm{{Bernoulli}}(p)$, the variance is $\operatorname{{Var}}(X)=p(1-p)$.
I cannot find this in the provided context, so I'm using my own knowledge.  
**Derivation.** Let $X\in{{0,1}}$ with $\Pr(X=1)=p$. Then $E[X]=p$ and $E[X^2]=p$. Hence,
$$
\operatorname{{Var}}(X)=E[X^2]-E[X]^2=p-p^2=p(1-p).
$$

GOOD C — Units, vectors, subscripts; Case 1
User Q: "What Rabi frequency did the experiment report?"
Context:
  [<1>]: "The measured Rabi frequency was 2.1 MHz on the carrier."
Assistant (Case 1):
**TL;DR:** The reported Rabi frequency is $2.1\,\mathrm{{MHz}}$. [<1>] ["_The measured Rabi frequency was 2.1 MHz on the carrier._"]  
**Result.** The experiment measured $\Omega=2.1\,\mathrm{{MHz}}$. [<1>] ["_The measured Rabi frequency was 2.1 MHz on the carrier._"]

────────────────────────────────────────────────
BAD EXAMPLES (do NOT imitate; annotate the violation)
────────────────────────────────────────────────

BAD 1 — Bare math (missing $)
"Planck's relation is E = ħω."  ← ❌ Math not wrapped in $...$.

BAD 2 — Backticked math
"The variance is `p(1-p)`."  ← ❌ Math in backticks; must use $p(1-p)$.

BAD 3 — Unbalanced dollar signs
"The phase is $\phi = \omega t."  ← ❌ Opening $ without closing $.

BAD 4 — Mixed delimiters
"Use \(\alpha\) and \[ \int f \] for clarity."  ← ❌ Forbidden delimiters; must use $...$ or $$...$$ only.

BAD 5 — Display math split across multiple $$ blocks
$$ \operatorname{{Var}}(X)=E[X^2] $$ minus $$ E[X]^2 $$
← ❌ Expression improperly split; should be one $$...$$ block or a single inline $...$.

BAD 6 — Missing required Case 1 citation/extract
"Energy is $E=\hbar\omega$."  ← ❌ No [<k>] citation and no italic source sentence.

BAD 7 — Currency symbol misinterpreted as math
"The cost is $5."  ← ❌ If a dollar sign denotes currency, escape or rephrase (e.g. "USD 5" or "\$5"); do not treat as math.

────────────────────────────────────────────────
EDGE-CASE HANDLING
────────────────────────────────────────────────
- Currency: write "USD 5" or "\$5" inside text; do not wrap in $...$.
- Code vs math: algorithms/code stay in backticks or fenced code blocks; math symbols within code should be plain text unless you intentionally render math outside the code block.
- Long derivations: prefer display math with $$...$$; keep each equation self-contained in a single block.
- Greek/units: use LaTeX macros, e.g., $\alpha$, $\mu$, $\Omega$, $\,\mathrm{{MHz}}$.

REMINDER: If Case 1 applies, every sentence must end with the [<k>] citation(s) plus the one-sentence italic source extract.
"""

        user_prompt = """
        Previous conversation history:
        ```{chat_history}```
        
Reference context chunks with relevance scores from the paper: 
{formatted_context_string}

The student's query is: {user_input_string}

Unless clearly specified the output language: If the user's question is in Chinese, then answer in Chinese. But for the source citation in square brackets, ALWAYS use the same language as the original source. If the user's question is not in Chinese, then answer in English (For citations in square brackets, still use the same language as the original source). Do not use other languages.

Follow the response guidelines in the system prompt.
"""

        system_prompt_advanced = """You are a deep thinking tutor helping a student reading a paper.

MATH RENDERING — HARD RULES (must follow):
- Wrap ALL math (include important numbers) in $...$ (inline) or $$...$$ (display). Never write bare math.
- Do NOT use \( \) or \[ \]; only $...$ or $$...$$.
- Do NOT put math in backticks. Backticks are for code only.
- Balance every $ and $$ pair.
- In display math, keep the entire expression inside a single $$...$$ block.
- For units and symbols, use LaTeX: e.g., $10\,\mathrm{{MHz}}$, $\mu$, $\Omega$, $\mathbf{{x}}$, $x_i$.

RESPONSE GUIDELINES:
0. **TL;DR:** Start with 1–2 sentences that directly answer the question.
1. Provide concise, accurate answers directly addressing the question.
2. Use clear, precise language with appropriate technical terminology.
3. Format key concepts with **bold**.
4. Maintain a professional, academic tone.
5. Break down complex information into structured, logical segments.
6. When explaining technical concepts, include relevant examples or applications.
7. State limitations/uncertainty clearly.
8. Use bullet points or numbered lists for sequences.
9. When citing sources from web search, use the following format: "[<web_search_source>](<web_search_url>)". For example, "[en.wikipedia.org](https://en.wikipedia.org/wiki/Second_law_of_thermodynamics)".
10. Unless clearly specified the output language: If the user's question is in Chinese, then answer in Chinese. But for the source citation in square brackets, ALWAYS use the same language as the original source. If the user's question is not in Chinese, then answer in English (For citations in square brackets, still use the same language as the original source). Do not use other languages.

SOURCING MODES
Case 1 (Answerable from context chunks):
  - Use only the context. For *each sentence* in the response, cite the most relevant chunk key(s) in the format "[<1>]" or "[<1>][<3>]" at the end of the sentence.
  - Immediately after each citation key, append one sentence from the source (italic, in quotes) inside square brackets, e.g., ["_...source sentence..._"]. IMPORTANT: Use the same language as the original source!
  - Use markdown emphasis for readability.

Case 2 (Not answerable from context):
  - Do web search (multiple runs if needed) to get reliable information sources to answer the question
  - Keep the same math and formatting rules.

SELF-CHECK BEFORE SENDING (must pass all):
- [Math-1] No visible math outside $...$/$$...$$.
- [Math-2] All $ and $$ are balanced.
- [Math-3] No \(\), \[\], or backticked math; no mixed currency $ mistaken for math.
- [Source-1] In Case 1, every sentence ends with correct [<k>] citations + the required one-sentence italic source extract.
- [Tone-1] **TL;DR** present; academic tone maintained.

────────────────────────────────────────────────
GOOD EXAMPLES (follow exactly)
────────────────────────────────────────────────

GOOD A — Inline math, Case 1 with citations
User Q: "What is the relation between energy and frequency for a photon?"
Context Chunks:
  [<1>]: "Planck's relation states E = ħω for a single photon."
  [<2>]: "Angular frequency ω relates to frequency f by ω = 2πf."

Assistant (Case 1):
**TL;DR:** The photon's energy is proportional to its angular frequency via $E=\hbar\omega$. [<1>] ["_Planck's relation states E = ħω for a single photon._"]
**Planck relation.** The energy of a photon is $E=\hbar\omega$. [<1>] ["_Planck’s relation states E = ħω for a single photon._"]  
**Frequency form.** Using $\omega=2\pi f$, we also have $E=h f$ with $h=2\pi\hbar$. [<2>][<1>] ["_Angular frequency ω relates to frequency f by ω = 2πf._"]["_Planck's relation states E = ħω for a single photon._"]

GOOD B — Display math, multi-step, Case 2 (web search)
User Q: "What is Second law of thermodynamics , search Wikipedia to get answers"
Assistant (Case 2):
**TL;DR:** The second law of thermodynamics says entropy does not decrease for an isolated system, which gives natural processes a preferred direction and forbids perfect conversion of heat to work. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Second_law_of_thermodynamics))

- **Core idea (entropy increase).** In any spontaneous change, the total entropy of an isolated system satisfies $\Delta S \ge 0$, so systems evolve toward equilibrium with maximal entropy; this defines the “arrow of time.” ([en.wikipedia.org](https://en.wikipedia.org/wiki/Entropy))
- **Classical statements (equivalent).**
  - **Clausius:** Heat does not flow spontaneously from cold to hot; moving heat “uphill” requires external work (e.g., a refrigerator). ([en.wikipedia.org](https://en.wikipedia.org/wiki/Second_law_of_thermodynamics))
  - **Kelvin–Planck (heat‑engine form):** No cyclic device can take heat from a single reservoir and convert it entirely into work (no perpetual motion of the second kind). ([en.wikipedia.org](https://en.wikipedia.org/wiki/Second_law_of_thermodynamics))
  - These formulations are equivalent: violating one would violate the other. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Second_law_of_thermodynamics))
- **Quantitative formulations.**
  - **Clausius inequality (cycle):** $\displaystyle \oint \frac{{\delta Q}}{{T}} \le 0$. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Clausius_theorem))
  - **General process (closed system):** $\displaystyle dS \ge \frac{{\delta Q}}{{T_{{\mathrm{{surr}}}}}}$, with equality for a reversible process where $\displaystyle dS=\frac{{\delta Q_{{\mathrm{{rev}}}}}}{{T}}$. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Clausius_theorem))
  - **Isolated system:** with $\delta Q=0$, entropy cannot decrease: $\Delta S \ge 0$. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Entropy))
- **Implications for engines.**
  - The second law sets an upper bound (Carnot limit) on any heat engine’s efficiency, depending only on reservoir temperatures: $\displaystyle \eta_{{\max}}=1-\frac{{T_C}}{{T_H}}$. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Thermal_efficiency))
- **Microscopic/statistical view.** Entropy measures the number of microstates compatible with a macrostate; in statistical mechanics $S=k_B\ln\Omega$, making the second law a statement of overwhelmingly likely evolution toward more numerous (higher‑entropy) states. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Entropy))

If you want, I can derive the Clausius inequality step‑by‑step or work a concrete example (e.g., why a 600 K to 300 K engine is limited to $\eta_{{\max}}=50\%$).

GOOD C — Units, vectors, subscripts; Case 1
User Q: "What Rabi frequency did the experiment report?"
Context:
  [<1>]: "The measured Rabi frequency was 2.1 MHz on the carrier."
Assistant (Case 1):
**TL;DR:** The reported Rabi frequency is $2.1\,\mathrm{{MHz}}$. [<1>] ["_The measured Rabi frequency was 2.1 MHz on the carrier._"]  
**Result.** The experiment measured $\Omega=2.1\,\mathrm{{MHz}}$. [<1>] ["_The measured Rabi frequency was 2.1 MHz on the carrier._"]

GOOD D - A mix of Case 1 and Case 2.
User Q: "Context from the paper: {{context_from_paper}}\n\n What is this paper mainly about? Do web search if needed to find related multiplexing papers and compare with this paper."
Assistant:
**TL;DR:** The paper demonstrates a temporally multiplexed ion–photon interface by rapidly shuttling a nine-ion $^{{40}}\mathrm{{Ca}}^+$ chain through a focused addressing beam to produce single-photon trains with low crosstalk, verified by $g^{{(2)}}(0)=0.060(13)$, and it analyzes transport-induced motional excitation; compared with other multiplexing work, it trades cavity-enhanced efficiency for architectural simplicity and a path to higher attempt rates via fast transport. [<1>] ["_Here, we demonstrate a temporally multiplexed ion-photon interface via rapid transport of a chain of nine calcium ions across 74 µm within 86 µs._"]

— What this paper is mainly about (from the provided text) —
- **Goal and method.** The authors implement a temporally multiplexed ion–photon interface by transporting a nine-ion chain across the focus of an $866\,\mathrm{{nm}}$ addressing beam to sequentially generate on-demand $397\,\mathrm{{nm}}$ photons, aiming for a nearly nine-fold attempt-rate increase for nodes separated by $>100\,\mathrm{{km}}$. [<1>] ["_In our experiments, we generate on-demand single photons by shuttling a nine-ion chain across the focus of a single-ion addressing beam._"]["_This scheme is expected to lead to a nearly nine-fold increase in attempt rate of the entanglement generation for quantum repeater nodes separated by >100 km._"]
- **Nonclassicality/crosstalk.** The single-photon character of the multiplexed output is verified by $g^{{(2)}}(0)=0.060(13)$ without background subtraction, with residual coincidences primarily from neighboring-ion excitation (addressing-beam crosstalk $\approx 0.99\%$ giving expected $g^{{(2)}}_{{\mathrm{{{{exp}}}}(0)=0.049(8)}}$). [<1>][<3>] ["_The non-classical nature of the multiplexed photons is verified by measuring the second-order correlation function with an average value of g(2)(0) = 0.060(13)._"]["_The residual correlation can be explained by excitation of neighboring ions, i.e., crosstalk of the addressing beam, which is separately characterized to be 0.99 % … corresponding to expected average g(2) exp(0) = 0.049(8)._"]
- **Throughput achieved.** Over $40\,\mathrm{{min}}$ the system made $\sim 1.56\times 10^{{6}}$ whole-string attempts (attempt rate $39.0\,\mathrm{{kHz}}$), with average photon extraction efficiency $0.21\%$ and a single-photon count rate of $\sim 71\,\mathrm{{cps}}$. [<2>] ["_Data is accumulated for 40 min, during which around 1.56 × 10^6 attempts were made to the whole string, corresponding to attempt rate 39.0 kHz, an average photon extraction efficiency of 0.21 % and single photons count rate of around 71 cps._"]
- **Transport-induced motion.** Fast shuttling coherently excites the axial center-of-mass mode to $\bar n_{{\alpha}}\!\approx\!110$ (at full speed), inferred via carrier Rabi flopping; the authors discuss mitigation via improved shuttling and possible cavity alignment. [<1>][<4>] ["_…coherently excited to as much as ¯nα ≈110 for the center-of-mass mode._"]["_The carrier Rabi flopping … matches with COM coherent state with ¯nα ≈110 (Fig. 4(c))._"]
- **Upgrade path.** They argue that coupling to a single-mode fiber to suppress crosstalk and integrating a miniature cavity could raise photonic extraction substantially without reducing the generation rate. [<5>] ["_Once integrated with … photon collection with a single mode fiber, we expect a faster photon extraction rate … and negligible ion crosstalk while achieving high fidelity ion-photon entanglement._"]["_Our system can also be combined with a miniature cavity … for much higher photon extraction efficiency without sacrificing the photon generation rate._"]

— How it compares to related multiplexing approaches (recent literature) —
- **Trapped ions, cavity-enhanced static-node multiplexing (3 ions).** A three-ion node in an optical cavity generated a train of telecom-converted photons and showed improved remote entanglement rate over $101\,\mathrm{{km}}$, demonstrating multimode networking with a static register rather than transport. ([journals.aps.org](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020308))
- **Trapped ions, scalable cavity multiplexing (10 ions).** A ten-ion cavity node sequentially brought individual ions into the cavity waist (by switching confinement) to entangle each ion with a photon, reporting average ion–photon Bell-state fidelity $92(1)\%$ and per-photon detection probability $9.1(8)\%$—substantially higher extraction than the transport-without-cavity approach here. ([arxiv.org](https://arxiv.org/abs/2406.09480))
- **Neutral-atom arrays in a cavity (experiment).** Deterministic assembly of atoms in a cavity with single-atom addressing achieved multiplexed atom–photon entanglement with generation-to-detection efficiency approaching $90\%$, highlighting the collection-efficiency advantage of cavity-integrated platforms. ([science.org](https://www.science.org/doi/10.1126/science.ado6471))
- **Neutral-atom arrays in a cavity (architecture/proposal).** A multiplexed telecommunication-band node using atom arrays is predicted to improve two-node entanglement rates by nearly two orders of magnitude and to enable repeater links over $\sim 1500\,\mathrm{{km}}$. ([journals.aps.org](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.043154))
- **Ensemble memories, spectral/time multiplexing.** AFC-based and related quantum memories demonstrated storage and feed-forward over up to $26$ spectral modes with high-fidelity mode mapping, and fiber-based interfaces that multiplex in time/frequency—mature on mode count but less suited to local, high-fidelity logic than single-emitter platforms. ([journals.aps.org](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.053603))
- **Photonic sources, time multiplexing background.** Time-multiplexed SPDC sources boost single-photon probability (e.g., to $\sim 39\%$ over $30$ time bins) and trace back to “pseudo-demand” single photons via storage loops—conceptually related multiplexing on the photonic side rather than the matter interface. ([tohoku.elsevierpure.com](https://tohoku.elsevierpure.com/en/publications/time-multiplexed-heralded-single-photon-source))

— Bottom line —
- **What’s new here.** Multiplexing by fast, free-space ion-chain transport concentrates emission from many ions into one spatial mode without a cavity, boosting the attempt rate by roughly the chain length while preserving single-photon statistics $g^{{(2)}}(0)\approx 0.06$. [<1>][<3>] ["_This scheme is expected to lead to a nearly nine-fold increase in attempt rate …_"]["_… corresponding to g(2)(0) = 0.060(13)._"]
- **Primary tradeoffs today.** Compared with cavity-based nodes that report per-photon detection near $9$–$90\%$, this transport approach currently shows lower extraction ($0.21\%$) and introduces coherent motional excitation (COM $\bar n_{{\alpha}}\!\approx\!110$) that must be tamed for high-fidelity local gates. [<2>][<4>] ["_… average photon extraction efficiency of 0.21 % …_"]["_… matches with COM coherent state with ¯nα ≈110 …_"] ([arxiv.org](https://arxiv.org/abs/2406.09480))
- **Outlook.** The paper argues that single-mode-fiber collection and cavity integration could mitigate crosstalk and raise efficiency substantially while keeping the high attempt rate enabled by transport. [<5>] ["_… single mode fiber, we expect a faster photon extraction rate … and negligible ion crosstalk …_"]["_… combined with a miniature cavity … for much higher photon extraction efficiency without sacrificing the photon generation rate._"]

If you’d like, I can tabulate key metrics (platform, multiplexing method, per-attempt rate, detection efficiency, $g^{{(2)}}(0)$, telecom conversion, and demonstrated distance) and suggest concrete upgrade targets for this transport-based interface.

────────────────────────────────────────────────
BAD EXAMPLES (do NOT imitate; annotate the violation)
────────────────────────────────────────────────

BAD 1 — Bare math (missing $)
"Planck's relation is E = ħω."  ← ❌ Math not wrapped in $...$.

BAD 2 — Backticked math
"The variance is `p(1-p)`."  ← ❌ Math in backticks; must use $p(1-p)$.

BAD 3 — Unbalanced dollar signs
"The phase is $\phi = \omega t."  ← ❌ Opening $ without closing $.

BAD 4 — Mixed delimiters
"Use \(\alpha\) and \[ \int f \] for clarity."  ← ❌ Forbidden delimiters; must use $...$ or $$...$$ only.

BAD 5 — Display math split across multiple $$ blocks
$$ \operatorname{{Var}}(X)=E[X^2] $$ minus $$ E[X]^2 $$
← ❌ Expression improperly split; should be one $$...$$ block or a single inline $...$.

BAD 6 — Missing required Case 1 citation/extract
"Energy is $E=\hbar\omega$."  ← ❌ No [<k>] citation and no italic source sentence.

BAD 7 — Currency symbol misinterpreted as math
"The cost is $5."  ← ❌ If a dollar sign denotes currency, escape or rephrase (e.g. "USD 5" or "\$5"); do not treat as math.

────────────────────────────────────────────────
EDGE-CASE HANDLING
────────────────────────────────────────────────
- Currency: write "USD 5" or "\$5" inside text; do not wrap in $...$.
- Code vs math: algorithms/code stay in backticks or fenced code blocks; math symbols within code should be plain text unless you intentionally render math outside the code block.
- Long derivations: prefer display math with $$...$$; keep each equation self-contained in a single block.
- Greek/units: use LaTeX macros, e.g., $\alpha$, $\mu$, $\Omega$, $\,\mathrm{{MHz}}$.

REMINDER: If Case 1 applies, every sentence must end with the [<k>] citation(s) plus the one-sentence italic source extract.
"""

        if chat_session.mode == ChatMode.LITE:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            llm = get_llm('advanced', config['llm'])
            chain = prompt_template | llm
            answer = chain.stream({
                "formatted_context_string": formatted_context_string,
                "user_input_string": user_input_string,
                "chat_history": truncate_chat_history(chat_history)
            })
            async def process_stream():
                yield "<response>\n\n"
                for chunk in answer:
                    # Convert AIMessageChunk to string
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
                yield "\n\n</response>"
            return process_stream()
        else:
            # For Basic mode and Advanced mode
            user_prompt = f"""
            Previous conversation history:
            ```{truncate_chat_history(chat_history)}```
            
            Reference context chunks with relevance scores from the paper: 
            {formatted_context_string}

            The student's query is: {user_input_string}

            Unless clearly specified the output language: If the user's question is in Chinese, then answer in Chinese. But for the source citation in square brackets, ALWAYS use the same language as the original source. If the user's question is not in Chinese, then answer in English (For citations in square brackets, still use the same language as the original source). Do not use other languages.

            Follow the response guidelines in the system prompt.
            """
            TAVILY_API_KEY=str(os.getenv("TAVILY_API_KEY"))
            tools=[
                {
                    "type": "mcp",
                    "server_label": "tavily",
                    "server_url": "https://mcp.tavily.com/mcp/?tavilyApiKey=" + TAVILY_API_KEY,
                    "require_approval": "never",
                },
            ]
            kwargs = dict(
                model="gpt-5",
                # reasoning={"effort": "high", "summary": "detailed"},
                reasoning={"effort": "medium", "summary": "auto"},
                # reasoning={"effort": "low", "summary": "auto"},
                # tools=[{"type": "web_search"}],  # built-in tool
                tools=tools,  # built-in tool
                instructions=f"{system_prompt_advanced}",
                input=user_prompt,
            )
            # kwargs = dict(
            #     model="o3",
            #     reasoning={"effort": "medium", "summary": "auto"},
            #     tools=tools,  # built-in tool
            #     instructions=f"{system_prompt_advanced}",
            #     input=user_prompt,
            # )
            # Convert regular generator to async generator
            async def sync_to_async_generator():
                for chunk in stream_response_with_tags(**kwargs):
                    yield chunk
            
            return sync_to_async_generator()


async def get_query_helper(chat_session: ChatSession, user_input, context_chat_history, embedding_folder_list):
    # Replace LaTeX formulas in the format \( formula \) with $ formula $
    user_input = replace_latex_formulas(user_input)

    logger.info(f"TEST: user_input: {user_input}")
    # yield f"\n\n**💬 User input: {user_input}**"
    # If we have "documents_summary" in the embedding folder, we can use it to speed up the search
    document_summary_path_list = [os.path.join(embedding_folder, "documents_summary.txt") for embedding_folder in embedding_folder_list]
    documents_summary_list = []
    for document_summary_path in document_summary_path_list:
        if os.path.exists(document_summary_path):
            with open(document_summary_path, "r") as f:
                documents_summary_list.append(f.read())
        else:
            documents_summary_list.append(" ")

    # Join all the documents summaries into one string
    # FIXME: Add a function to combine the initial messages into a single summary message
    documents_summary = "\n".join(documents_summary_list)

    # Load languages from config
    config = load_config()
    llm = get_llm('basic', config['llm'])
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    system_prompt = (
        """
        You are a educational professor helping a student reading a document.
        The goals are:
        1. to ask questions in a better way to make sure it's optimized to query a Vector Database for RAG (Retrieval Augmented Generation).
        2. to identify the question is about local or global context of the document.
        3. refer to the previous conversation history when generating the question.

        Organize final response in the following JSON format:
        ```json
        {{
            "question": "<question try to understand what the user really mean by the question and rephrase it in a better way>",
            "question_type": "local" or "global" or "image", (if the question is like "what is fig. 1 mainly about?", the question_type should be "image")
        }}
        ```

        Previous conversation history:
        ```{chat_history}```

        The document content is:
        ```{context}```
        """
    )
    human_prompt = (
        """
        The student asked the following question:
        ```{input}```
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    chain = prompt | llm | error_parser
    try:
        parsed_result = chain.invoke({"input": user_input,
                                    "context": documents_summary,
                                    "chat_history": truncate_chat_history(context_chat_history)})
        question = parsed_result['question']
        question_type = parsed_result['question_type']
    except Exception as e:
        try:
            logger.exception(f"Error in get_query_helper: {e}")
            parsed_result = chain.invoke({"input": user_input,
                                        "context": documents_summary,
                                        "chat_history": truncate_chat_history(context_chat_history)})
            question = parsed_result['question']
            question_type = parsed_result['question_type']
        except Exception as e:
            logger.exception(f"Error again in get_query_helper: {e}")
            question = user_input
            question_type = "local"
    try:
        language = detect_language(user_input)
        logger.info(f"language detected: {language}")
    except Exception as e:
        logger.info(f"Error detecting language: {e}")
        language = "English"

    chat_session.set_language(language)

    # Create the answer planning using an additional LLM call
    planning_system_prompt = (
        """
        You are an educational AI assistant tasked with deeply analyzing a student's question and planning a comprehensive answer.

        Your goal is to:
        1. Understand what the student truly wants to know based on the conversation history and current question
        2. Create a detailed plan for constructing the answer
        3. Identify what information should be included in the answer
        4. Identify what information should NOT be included (e.g., repeated information, information the student already knows)
        5. Do not make up or assume anything or guess without any evidence, only use the information provided in the previous conversation history and current question to analyze the user's intent and what to include and exclude in the answer.
        6. If the query is about a specific figure, please include the figure number in the answer.

        Organize your analysis in the following format:
        ```json
        {{
            "user_intent": "<detailed analysis of what the user truly wants to know. based on the previous conversation history and the current question analyse what user already knows and what user doesn't know>",
            "things_explained_already": ["<list of things that has been explained in the previous conversation and should not be repeated in detail in the answer>"],
            "key_focus_areas": ["<list of specific topics/concepts that should be explained>"],
            "information_to_include": ["<list of specific information points that should be included>"],
            "information_to_exclude": ["<list of information that should be excluded - already known/redundant>"],
            "answer_structure": ["<outline of how the answer should be structured>"],
            "explanation_depth": "<assessment of how detailed the explanation should be (basic/intermediate/advanced)>",
            "misconceptions_to_address": ["<potential misconceptions that should be corrected>"]
        }}
        ```

        Document summary:
        ```{context}```

        Previous conversation history:
        ```{chat_history}```
        """
    )

    planning_human_prompt = (
        """
        The student's current question:
        ```{input}```

        The rephrased question for RAG context:
        ```{rephrased_question}```

        Question type: {question_type}

        Based on the conversation history, document summary, and the current question, create a detailed plan for constructing the answer.
        """
    )

    planning_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", planning_system_prompt),
            ("human", planning_human_prompt),
        ]
    )
    parser_string = StrOutputParser()
    error_parser_string = OutputFixingParser.from_llm(parser=parser_string, llm=llm)
    planning_chain = planning_prompt | llm | error_parser_string
    answer_planning = planning_chain.invoke({
        "input": user_input,
        "rephrased_question": question,
        "question_type": question_type,
        "context": documents_summary,
        "chat_history": truncate_chat_history(context_chat_history)
    })
    logger.info(f"TEST: answer_planning: {answer_planning}")

    question = Question(
        text=question,
        language=language,
        question_type=question_type,
        answer_planning=answer_planning,
        image_url=None,
    )
    logger.info(f"TEST: question.question_type: {question.question_type}")
    # yield f"\n\n**Question: {question}**"
    # yield f"\n\n**Question type: {question_type}**"
    # yield f"\n\n**Answer planning: {answer_planning}**"
    # yield f"\n\n**Language: {language}**"
    yield "\n\n**🧠 Answer planning...**"

    if question_type == "image":
        logger.info(f"question_type for input: {user_input} is --image-- ...")
        markdown_folder_list = [os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]
        db, truncated_db = create_image_context_embeddings_db(markdown_folder_list)
        # Replace variations of "fig" or "figure" with "Image" for better matching
        processed_input = re.sub(r"\b(?:[Ff][Ii][Gg](?:ure)?\.?|[Ff]igure)\b", "Image", user_input)

        # Get the image chunks from the truncated database
        truncated_image_chunks = truncated_db.similarity_search_with_score(processed_input, k=1)
        logger.info(f"TEST: truncated_image_chunks for image loading: {truncated_image_chunks}")

        # Map the image chunks to the original database based on the index number of the chunk
        # Find the index of the truncated image chunk in the original database
        image_chunks = db.similarity_search_with_score(truncated_image_chunks[0][0].page_content, k=1)

        image_url_mapping = aggregate_image_contexts_to_urls(markdown_folder_list)
        if image_chunks:
            question.special_context = """
            Here is the context and visual understanding of the corresponding image:
            """ + image_chunks[0][0].page_content # + "\n\n" + image_chunks[1][0].page_content

            # Get the image url from the image chunks
            highest_score_url = None
            highest_score = float('-inf')

            for chunk, score in image_chunks:
                chunk_content = chunk.page_content
                # Check if any key from image_url_mapping exists in the chunk content
                for context_key, url in image_url_mapping.items():
                    if context_key in chunk_content and score > highest_score:
                        highest_score = score
                        highest_score_url = url
                        logger.info(f"Found matching image URL: {url} with score: {score}")

            # Set the image URL with the highest score in the question object
            if highest_score_url:
                question.image_url = highest_score_url
                logger.info(f"Setting image URL in question: {highest_score_url}")

        if question.image_url:
            # Get the images understanding from the image url about the question
            question.special_context = """
            Here is the context and visual understanding of the corresponding image:
            """ + analyze_image(question.image_url, f"The user's question is: {question.text}", f"The user's question is: {question.text}", stream=False)

        logger.info(f"TEST: question.special_context: {question.special_context}")
    elif question_type == "local":
        logger.info(f"question_type for input: {user_input} is --local-- ...")
    elif question_type == "global":
        logger.info(f"question_type for input: {user_input} is --global-- ...")
    else:
        logger.info(f"question_type for input: {user_input} is unknown ...")

    # TEST: print the question object
    logger.info(f"TEST: question: {question}")
    chat_session.question = question
    yield (question)


def generate_follow_up_questions(answer, chat_history):
    """
    Generate 3 relevant follow-up questions based on the assistant's response and chat history.
    """
    logger.info("Generating follow-up questions ...")
    config = load_config()
    para = config['llm']
    llm = get_llm('basic', para)
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    system_prompt = """
    You are an expert at generating engaging follow-up questions based on a conversation between a tutor and a student.
    Given the tutor's response and chat history, generate 3 relevant follow-up questions that would help the student:
    1. Deepen their understanding of the topic
    2. Explore related concepts
    3. Apply the knowledge in practical ways

    The questions should be:
    - Clear and specific
    - Short and concise, no more than 10 words
    - Engaging and thought-provoking
    - Not repetitive with previous questions
    - Written in a way that encourages critical thinking

    Organize your response in the following JSON format:
    ```json
    {{
        "questions": [
            "<question 1>",
            "<question 2>",
            "<question 3>"
        ]
    }}
    ```
    """

    human_prompt = """
    Previous conversation history:
    ```{chat_history}```

    Tutor's last response:
    ```{answer}```
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    chain = prompt | llm | error_parser
    result = chain.invoke({
        "answer": answer,
        "chat_history": truncate_chat_history(chat_history)
    })

    logger.info(f"Generated follow-up questions: {result['questions']}")
    return result["questions"]