"""
(deeptutor) bingran_you@wifi-10-45-8-48 Responses_API_test % python responsee_api_test.py
Here are the most recent, peer‑reviewed/published results (roughly 2024–2025) on integrating photonics with trapped‑ion platforms, grouped by capability.

Light delivery and control on chip
- Multi‑site addressing with all qubit wavelengths delivered by on‑chip waveguides and splitters: Sandia demonstrated simultaneous, site‑resolved operations on 171Yb+ using a surface trap that routes cooling, state‑prep, coherent control and detection light via integrated waveguides and MMI splitters (Nature Communications, 2 May 2024). They showed simultaneous Rabi flopping at distinct sites from a single optical input per wavelength. ([osti.gov](https://www.osti.gov/pages/biblio/2469920?utm_source=openai))
- First coherent multi‑zone control in an integrated‑photonics QCCD device: the ETH Zürich/Cornell team used integrated waveguides to run a Ramsey sequence across two zones 375 μm apart with 200 μs ion transport between pulses, plus simultaneous control at separate zones with low crosstalk (arXiv, 31 Jan 2024). ([arxiv.org](https://arxiv.org/abs/2401.18056?utm_source=openai))
- Chip‑internal structured fields for new control primitives: integrated gratings forming a passively phase‑stable standing‑wave “optical lattice” above the trap enabled state‑dependent motional‑mode shifts (single‑ion shift 2π×3.33(4) kHz, corresponding to a bare potential 2π×76.8(5) kHz) and two‑ion mode control (arXiv, 5 Nov 2024). ([arxiv.org](https://arxiv.org/abs/2411.03301?utm_source=openai))
- Inverse‑designed multimode couplers for reconfigurable, low‑crosstalk individual addressing (proposal/simulation): TE10/TE20 interference promises −20 to −30 dB crosstalk at 5–8 μm ion spacings, up to −60 dB when addressing two of three ions simultaneously (arXiv, 13 May 2025). ([arxiv.org](https://arxiv.org/abs/2505.08997?utm_source=openai))

On‑chip collection and readout
- Trap‑integrated waveguide collection of ion fluorescence: MIT/MIT‑LL built a dual‑layer focusing grating under the electrodes that couples 422‑nm Sr+ fluorescence directly into a single‑mode waveguide. Measured single‑mode collection efficiency was 0.043% (design limited by fabrication, with a simulated 0.7% ideal vs. 2.18% solid‑angle limit); they also used the path to detect the ion’s state (arXiv, 2 May 2025). ([ar5iv.org](https://ar5iv.org/pdf/2505.01412))
- Integrated photon detectors: 
  - Room‑temperature SPADs embedded in the trap substrate achieved 99.92(1)% state‑readout fidelity for 88Sr+ in 450 μs (PRL, 2022; still the benchmark for trap‑integrated diode readout). ([osti.gov](https://www.osti.gov/pages/biblio/1982829?utm_source=openai))
  - Trap‑integrated SNSPDs with rf shielding now tolerate typical trapping rf (up to 54 Vpk at 70 MHz) and operate to 6 K with up to 68% system detection efficiency (APL, 2023), and NIST reported an improved design and 10× better rf tolerance in a 2024 SPIE paper. ([pubs.aip.org](https://pubs.aip.org/aip/apl/article/122/17/174001/2885293?utm_source=openai))

Photonic interconnects and networking
- Fast photon‑mediated entanglement using integrated optics in the link: Ba+ ions were entangled via single‑photon interference through an integrated fiber beamsplitter, sustaining 250 entanglement events/s with fidelity >94%, while maintaining continuous sympathetic cooling (arXiv, 24 Apr 2024). ([arxiv.org](https://arxiv.org/abs/2404.16167?utm_source=openai))
- Ion‑to‑PIC compatibility at telecom via frequency conversion: single photons from a trapped ion were converted and routed through a foundry‑fabricated SiN PIC, including programmable splitting (Phys. Rev. Applied, 1 Mar 2023). ([journals.aps.org](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.19.034001?utm_source=openai))

Enabling materials and active UV/visible PIC components (key for ion wavelengths)
- CMOS‑fabricated alumina PICs with piezo‑optomechanical modulators demonstrated down to 320 nm (waveguide loss 1.6 dB/cm at 320 nm; 30 dB amplitude modulation; racetrack Q≈4.7×10^5), indicating native UV beam routing/switching on chip (arXiv, 29 Jun 2024). ([arxiv.org](https://arxiv.org/abs/2407.00469?utm_source=openai))
- High‑index HfO2–Al2O3 composite PIC platform with low loss across UV–visible: ring resonators with intrinsic Q≈2.6×10^6 at 729 nm; inferred waveguide loss ≈0.25 dB/cm (729 nm), 2.6 dB/cm (405 nm), 7.7 dB/cm (375 nm) (arXiv, 12 Dec 2024). ([arxiv.org](https://arxiv.org/abs/2412.09421?utm_source=openai))
- Integrated thin‑film LiNbO3 achieved milliwatt‑level UV generation at 390 nm via sidewall poling (normalized SHG efficiency 5050 %/W/cm^2; 4.2 mW on‑chip power), relevant for compact ion‑control light sources (arXiv, 21 Mar 2025). ([arxiv.org](https://arxiv.org/abs/2503.16785?utm_source=openai))
- Additional UV PIC progress: reported Al2O3 waveguide loss ≈1.3 dB/cm at 369 nm (preprint, 23 Apr 2024). ([preprints.opticaopen.org](https://preprints.opticaopen.org/articles/preprint/UV_integrated_photonics_in_sputter_deposited_aluminum_oxide/25663656?utm_source=openai))

System engineering and integration challenges
- Dielectrics and apertures for photonics can distort trapping fields and add micromotion; FEM studies suggest mitigation via symmetry and transparent conductive oxides, consistent with experimental mitigation techniques used in multi‑zone integrated‑photonics traps (arXiv, 26 Mar 2025; arXiv, 31 Jan 2024). ([arxiv.org](https://arxiv.org/abs/2503.20387?utm_source=openai))
- Programs report delivery of deep‑UV beams (369/399 nm) from on‑chip couplers at ion heights down to 20 μm with no measurable additional heating, and roadmaps toward on‑chip SPADs for clocks (Sandia TICTOC page). ([sandia.gov](https://www.sandia.gov/quantum/quantum-information-sciences/projects/tictoc/?utm_source=openai))
- Practical cavity coupling for modular photonic links: recent design work details how to integrate miniature cavities into linear traps while shielding dielectrics and exploiting trap symmetry (Phys. Rev. Applied, 18 Feb 2025). ([journals.aps.org](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.23.024038?utm_source=openai))

Industrial roadmaps
- IonQ announced a collaboration with imec to co‑develop photonic integrated circuits and chip‑scale traps aimed at shrinking bulk optics, increasing qubit count, and improving robustness (press release, 7 Nov 2024). ([investors.ionq.com](https://investors.ionq.com/news/news-details/2024/IonQ-to-Increase-Performance-and-Scale-of-Quantum-Computers-with-Photonic-Integrated-Circuits-in-Collaboration-with-imec/default.aspx?utm_source=openai))

What this means
- As of September 2025, integrated waveguides/gratings have moved from single‑site demos to multi‑site, multi‑wavelength control consistent with QCCD scaling; structured fields from on‑chip gratings enable new gate/cooling modalities; and first trap‑integrated, single‑mode photon collection paths have been demonstrated. Detector integration (SPADs/SNSPDs) is maturing toward parallel, scalable readout. Materials and components for native UV/visible PICs (Al2O3, HfO2/Al2O3, TFLN) are now showing losses and active functions compatible with ion control, pointing to tighter, more manufacturable photonics‑ion co‑integration in the next hardware iterations. ([osti.gov](https://www.osti.gov/pages/biblio/2469920?utm_source=openai))

If you want, I can tailor a deeper dive on any of these threads (e.g., gate fidelities with integrated delivery, detector readout budgets, or packaging approaches).%
"""

from openai import OpenAI
from dotenv import load_dotenv
import os, re
from typing import Iterable

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _format_thinking_delta(delta: str) -> str:
    """
    Make bold step titles render as their own paragraphs.
    Heuristic: whenever we see **...**, ensure there is a blank line
    before and after the bold run.
    """
    # If a bold run appears without a leading newline, add blank line before.
    # Then ensure a blank line after.
    s = delta
    s = re.sub(r'(?<!\n)\*\*([^*][^*]*?)\*\*', r'\n\n**\1**', s)  # ensure leading blank line
    s = re.sub(r'\*\*([^*][^*]*?)\*\*(?!\n)', r'**\1**\n\n', s)   # ensure trailing blank line
    return s

def stream_response_with_tags(**create_kwargs) -> Iterable[str]:
    """
    Yields a single XML-like stream:
      <thinking> ...reasoning summary + tool progress... </thinking><response> ...final answer... </response>
    """
    stream = client.responses.create(stream=True, **create_kwargs)

    # Show a thinking container immediately
    thinking_open = True
    response_open = False
    yield "<thinking>\n\n"

    try:
        for event in stream:
            t = event.type

            # --- Reasoning summary stream ---
            if t == "response.reasoning_summary_text.delta":
                yield _format_thinking_delta(event.delta)

            elif t == "response.reasoning_summary_text.done":
                # keep <thinking> open for tool progress; we'll close when answer starts or at the very end
                pass

            # --- Tool progress (e.g., web_search) inside <thinking> ---
            elif t == "response.tool_call.created":
                tool_name = getattr(event.tool, "name", "tool")
                yield f"[tool:start name={tool_name}]\n"
            elif t == "response.tool_call.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta
            elif t == "response.tool_call.completed":
                tool_name = getattr(event.tool, "name", "tool")
                yield f"\n[tool:done name={tool_name}]\n\n"

            # --- Main model answer text ---
            elif t == "response.output_text.delta":
                if thinking_open:
                    yield "\n</thinking>\n\n"
                    thinking_open = False
                if not response_open:
                    response_open = True
                    yield "<response>\n\n"
                yield event.delta

            # ✅ Close <response> as soon as the model finishes its text
            elif t == "response.output_text.done":
                if response_open:
                    yield "\n\n</response>\n"
                    response_open = False

            # --- Finalization / errors ---
            elif t == "response.completed":
                # We may already have closed </response>; just ensure well-formed
                if thinking_open:
                    yield "\n</thinking>\n"
                    thinking_open = False

            elif t == "response.error":
                if thinking_open:
                    yield "\n</thinking>\n"
                    thinking_open = False
                if response_open:
                    yield "\n</response>\n"
                    response_open = False
                # Optionally surface the error:
                # yield f"<!-- error: {event.error} -->"

    finally:
        try:
            stream.close()
        except Exception:
            pass


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    kwargs = dict(
        model="gpt-5",
        reasoning={"effort": "high", "summary": "detailed"},
        tools=[{"type": "web_search"}],  # built-in tool
        instructions="Search the web as needed (multiple searches OK) and cite sources.",
        input="find bingran you in hartmut haeffner's group and review his google scholar page",
    )

    for chunk in stream_response_with_tags(**kwargs):
        print(chunk, end="", flush=True)
    print()