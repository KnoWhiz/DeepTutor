"""
<thinking>

**Searching for profiles**

I'm considering that the Scholar search can be a bit noisy. It might be better to look for a Google Scholar profile page, though I know they don’t always appear with a name search unless I use quotes. I’ll search for "Bingran You site:berkeley.edu" on Scholar. But, since Scholar search has its limits, I might as well search the general web with "Bingran You Berkeley Google Scholar" to see what I can find.

**Organizing the final response**

I’m getting ready to craft the final answer. I want to include a title like "What I found" along with bullet points for clarity.

I'll verify membership and note that I couldn't find a Google Scholar profile as of the current date, explaining that. I'll mention a 2024 preprint publication, specifically focusing on "Temporally multiplexed ion-photon quantum interface via fast ion-chain transport." 

I’ll also highlight another co-authored preprint related to integrated photonics and an APS talk. 

My aim is to keep everything concise and search for proper citations without raw URLs. Let's finalize this!
</thinking>

<response>

Here’s what I found:

- Membership: Bingran You is listed as a graduate student in Hartmut Häffner’s Ion and Electron Trap Group (UC Berkeley). ([ions.berkeley.edu](https://ions.berkeley.edu/members/))

- Google Scholar page: I could not find a public Google Scholar author profile for “Bingran You” as of September 25, 2025. It’s possible he hasn’t created a public profile or it’s set to private (Google Scholar profiles are only visible if made public). ([scholar.google.com](https://scholar.google.com/intl/pt-PT/scholar/citations.html))

- Publications visible via scholarly indexes:
  - First‑author preprint: “Temporally multiplexed ion‑photon quantum interface via fast ion‑chain transport” (May 17, 2024), coauthored with members of the Häffner group and collaborators. This demonstrates a temporally multiplexed ion‑photon interface using rapid transport of a nine‑ion Ca+ chain, relevant to high‑rate remote entanglement for quantum networking. ([arxiv.org](https://arxiv.org/abs/2405.10501))
  - Related conference talk: “Temporal multiplexing for improved ion‑photon interface,” presented at APS DAMOP 2024. ([meetings.aps.org](https://meetings.aps.org/Meeting/DAMOP24/Session/R06.6))
  - Co‑authored preprint: “Scalable Trapped Ion Addressing with Adjoint‑optimized Multimode Photonic Circuits” (May 16, 2025), proposing integrated photonics for targeted, reconfigurable ion addressing. ([preprints.opticaopen.org](https://preprints.opticaopen.org/articles/preprint/Scalable_Trapped_Ion_Addressing_with_Adjoint-optimized_Multimode_Photonic_Circuits/29087921))

If by “his Google Scholar page” you meant Hartmut Häffner’s profile, that does exist (linked from Berkeley’s faculty page); I can review it on request. ([vcresearch.berkeley.edu](https://vcresearch.berkeley.edu/faculty/hartmut-haeffner))

Would you like me to:
- keep watching for a public Scholar profile for Bingran You, or
- proceed to summarize Hartmut Häffner’s Google Scholar profile instead?

</response>
"""


context_from_paper = """
{'[<1>]': {'content': 'Temporally multiplexed ion-photon quantum interface via fast ion-chain transport Bingran You,1, 2 Qiming Wu,1, 2, ∗David Miron,1, 2 Wenjun Ke,1 Inder Monga,2 Erhan Saglamyurek,1, 2 and Hartmut Haeffner1, 2, † 1Department of Physics, University of California, Berkeley, CA 94720, USA 2Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA (Dated: September 27, 2024) High-rate remote entanglement between photon and matter-based qubits is essential for distributed quantum information processing. A key technique to increase the modest entangling rates of existing long-distance quantum networking approaches is multiplexing. Here, we demonstrate a temporally multiplexed ion-photon interface via rapid transport of a chain of nine calcium ions across 74 µm within 86 µs. The non-classical nature of the multiplexed photons is verified by measuring the second-order correlation function with an average value of g(2)(0) = 0.060(13). This indicates low crosstalk between the multiplexed modes, and can be reduced to negligible level once fiber coupling of single photons is incorporated. In addition, we characterize the motional degree-of-freedom of the ion crystal after transport and find that it is coherently excited to as much as ¯nα ≈110 for the center-of-mass mode. Our proof-of-principle implementation paves the way for large-scale quantum networking with trapped ions, but highlights some challenges that must be overcome. Remote entanglement across distant quantum nodes [1, 2] may be used for long-distance quantum key distribution [3, 4], modular quantum computer architectures [5, 6], as well as quantum enhanced metrology and sensing [7–9]. Light-matter quantum interfaces are a fundamental building block for such applications, and allow for distributed entanglement between stationary matter qubits by using "flying" photons. For practical purposes, these quantum interfaces need to be capable of establishing remote entanglement at high rates across a large-scale network of quantum nodes. However, in widely adopted probabilistic schemes based on heralded photon detection [10–12], it is infeasible to distribute entanglement at useful rates over a few kilometers, as the attempt rate is limited by the round-trip travel-time of photons in a single mode. For example, for a communication distance of 100 km, photon-travel time will be around 1 ms, meaning that the attempt cannot be higher than 1 kHz. In this scenario, even with use of a state-of-the-art photon-matter interface that can yield close to unity photon extraction efficiency [13], the matter-photon entanglement rate beyond tens of kilometers would be limited to sub 10 s−1 with total loss more than 20 dB after considering the collection probability of generated photons, quantum frequency conversion efficiency, photon detector efficiency, and the optical fiber loss [14, 15]. A key solution to this challenge is multiplexing: to combine multiple signals into a single channel and therefore increase the attempt rate [16, 17]. Multiplexing has become a mature technology for ensemble-based quantum interfaces, both in atomic gases and rare-earth ion doped solid-state systems [18–21]. However, large-scale local quantum information processing is technically challenging in these platforms [22]. In contrast, single emitters, including trapped ions and neutral atoms, offer excellent local quantum information processing capability beside their natural interface with light at convenient wavelengths for quantum frequency conversion (QFC) [23, 24], and the possibility of longlived storage of entanglement [25, 26]. On the other hand, implementing a multiplexed light-matter interface with these systems is technically challenging. Towards overcoming this problem, a few multiplexing schemes have already been proposed for ion and atom-based quantum processors [27–30]. The only reported experimental work, we are aware of, is the demonstration of multiplexing using a static three-ion chain [15]. In view of the recent advances of the quantum CCD architecture [31–33], a complementary approach to multiplexing is the process of ion-transport through a specific spatial location with maximized photon coupling efficiency. In this work, we demonstrate a temporal multiplexing scheme based on the transport of an ion-chain for improving the rate of ion-photon entanglement over long distances. In our experiments, we generate on-demand single photons by shuttling a nine-ion chain across the focus of a single-ion addressing beam. This scheme is expected to lead to a nearly nine-fold increase in attempt rate of the entanglement generation for quantum repeater nodes separated by >100 km. We verify the single-photon nature of the photon trains by measuring a second-order time correlation of g(2)(0) = 0.060(13) without background subtraction. Furthermore, we address the problem of motional excitation during the transport, which is detrimental to local entangling operations [34] and in the case of using a cavity for stimulating the photons would lead to uncertainty in the coupling strength. [35]. Using a shuttling function designed to mitigate motional excitation, we find coherent excitation as high as ¯nα ∼110 on the center-of-mass (COM) mode during one round of ion chain transport. These results show that the proposed multiplexing scheme can be scaled up to higher rates provided that more optimal transport methods are', 'score': 0.39765414595603943, 'page_num': 1, 'source_index': 1}, '[<2>]': {'content': '2 74 um 397 nm single photons SPD & Objective 866 nm addressing BS (a) (b) 397 nm 397 nm 866 nm 866 nm Doppler cooling … Photon generation sequence × 2 200 μs 10.8 μs … Repeat attempts (c) mode 1 866 addressing Transport Delay Delay mode 9 866 addressing Transport Delay Delay 397 pumping 3 μs GND VRF VRF GND VDC1 + ΔVDC1(t) VDC2 - ΔVDC2(t) zi zf SPD z x y FIG. 1. Schematics of multiplexed ion-photon interface. (a) A nine-ion chain is confined in an RF Paul trap. Controlling DC endcap voltages allows for ion transport. A beam of 397 nm and 866 nm light illuminating all ions is used for Doppler cooling. An objective collects the 397 nm single photons and guides them to a 50/50 beamsplitter, followed by a photomultiplier tube on each exit port for photon detection. An 866 nm beam individual addressing beam counter-propagates with the single photons. (b), (c) Excitation scheme and pulse sequence for the 397 nm single-photon generation. First, a global 397 nm beam prepares the ions to the 32D3/2 state. Then, the 866 nm addressing beam (resonance with 32D3/2 ↔42P1/2) is stroboscopically switched on during the transport to extract photons from the target ions. applied. The schematics of the experimental procedures is illustrated in Fig. 1(a). Our experiment is conducted using an (a) (b) FIG. 2. Temporal profile of single-photon generation. (a) The black (red) line is the programmed (measured) voltage ramp on endcap 1 during the forward transport, showing negligible latency effect. The voltage on endcap 2 is an inverse function of endcap 1 with an offset of +2.51 V. Not shown is the ramp shuttling the ions back to the original position. (b) Detection time of the photons (bin with 250 ns) color coded with the associated temporal mode. Data is accumulated for 40 min, during which around 1.56 × 106 attempts were made to the whole string, corresponding to attempt rate 39.0 kHz, an average photon extraction efficiency of 0.21 % and single photons count rate of around 71 cps. RF Paul trap, composed of four RF blades for generating radial pseudopotential and two DC endcap electrodes for providing an axial harmonic confinement. We typically trap a string of nine 40Ca+ ions in a linear configuration with the COM mode frequency of ωx,z/(2π) = × {1.15, 0.179} MHz and Lamb-Dicke (LD) parameters of the axial modes, ranging between 0.09 and 0.23 on the 42S1/2 ↔32D5/2 transition. Two global laser beams at 397 nm and 866 nm (not shown in the figure) evenly illuminate the entire ion chain for Doppler cooling and optical pumping, and a tightly focused 866 nm addressing beam with beam waist ≈6.0 µm allows for resonant excitation to extract the 397 nm single photons from individual ions. The generated single photons are collected by an objective with a numerical aperture of NA = 0.3 (collection efficiency Pc ≈2.5%) and directed to a 50/50 beam splitter (BS). At the exit port of BS, photons are detected by two photomultiplier tube (PMT)-based single-photon detectors (SPD), and their arrival times are recorded with a time-tagger for subsequent analysis. We generate on-demand single-photons based on a background-free scheme, as illustrated in Fig. 1(b) [36]. In this process, we first Doppler cool (detuning ∆= −Γ/2) the ion chain for 200 µs, with the first 100 µs assisted by another beam −500 MHz detuned from the 42S1/2 ↔42P1/2 transition (not shown) to mitigate col-', 'score': 0.5361793041229248, 'page_num': 2, 'source_index': 1}, '[<3>]': {'content': '3 lision induced ion chain melting in the Paul trap [37]. Then we begin the photon generation sequences with optical pumping to the 32D3/2 state for 3 µs, followed by transport of the chain to position each ion in the tight focus of the 866 nm addressing beam resonant with 32D3/2 ↔42P1/2 transition to generate 397 nm single photons (see Fig. 1(c)). In the ion-chain transport process, the endcap voltages are controlled by an arbitrary waveform generator (AWG) amplified by a custom-made, low-noise amplifier circuit with a gain of ten through low-pass filters and a vacuum feedthrough. The low-pass filters have cutoff (a) (b) (c) Shuttling nine-ion chain Fifth ion in a static nine-ion chain Static single ion FIG. 3. Two-photon correlation measurements. (a) g(2) measurement when shuttling a nine-ion chain, g(2)(0) = 0.060(13). The horizontal axis represents the delay between individual photons corresponding to the time between each voltage step. The shuttling function in Fig. 2 is repeated twice before the next cooling cycle (see Fig. 1). (b) g(2) measurement when addressing the fifth ion in a static nine-ion chain shows g(2)(0) = 0.062(20). There are 25 attempts after each cooling cycle. The horizontal axis represents the delay time between attempts on the same ion. (c) g(2) measurement of a single ion with the addressing beam using the same pulse sequence as (b). The measured g(2)(0) = 0.010(6). The horizontal axis is the same as (b). frequencies of 1.9 MHz to allow fast transport of the ion chain close to the speed of the COM mode frequency. The programmed and the measured waveform show a negligible latency effect from the filters (Fig. 2(a)). The forward shuttling function has eight steps, during each of which a different ion is placed in the focus of the addressing beam for 1.7 µs with the beam turned on simultaneously. After completing this sequence, we move the entire ion chain back to the original position in 35 µs using the same function form in one step. The voltage ramping function on the two endcaps V1,2(t) is in the form of a sigmoid-like polynomial function such that the first and second order derivative at the beginning and the end of the transport vanish [38]. VDC1(t) = VDC1 + ∆V \x10 10 \x00 t T \x013 −15 \x00 t T \x014 + 6 \x00 t T \x015\x11 VDC2(t) = VDC2 −∆V \x10 10 \x00 t T \x013 −15 \x00 t T \x014 + 6 \x00 t T \x015\x11 , (1) where ∆V is the voltage difference between the beginning and the end of a step, t is the time after the end of the previous step, and T = 9.1 µs is the total time of each transport. The details of voltage optimization and numerical simulation of motional heating can be found in [39]. We reconstruct the temporal profile of 397 nm photons during transport using the recorded arrival times of photons on the PMTs. Fig 2(b) shows the emission from individual ions (modes). Next, we perform a two-photon correlation experiment to test the non-classical characteristics of our multiplexed photon source [40]. The probability of two-photon correlation when detecting a correlation event on two detectors at different times is given by ρc(τ) = ρ1(τ)ρ2(τ + δT), (2) where ρ1(τ) and ρ2(τ + δT) are the probability of detecting a photon at t = τ and τ + δT on detector 1 and 2. Fig. 3(a) shows the normalized correlation counts as a function of the delay mode window. We choose a coincidence window of 300 ns in each mode and measure 8 coincident counts at zero delay in 4.8 hours, corresponding to g(2)(0) = 0.060(13). The residual correlation can be explained by excitation of neighboring ions, i.e., crosstalk of the addressing beam, which is separately characterized to be 0.99 % using fluorescence of the nineion chain on the camera, corresponding to expected average g(2) exp(0) = 0.049(8) (see Supplemental [39]). To further verify this hypothesis, we repeat the measurement with a single ion (Fig. 3(c)) and compare it to addressing only the fifth ion in a static nine-ion chain (Fig. 3(b)). The two experiments yield g(2)(0) = 0.010(6) and g(2)(0) = 0.062(20) with 6.0 and 4.8 hours of data accumulation, respectively. While the single ion g2 is limited by the detector dark counts and ambient light, the measurement of the static 9-ion chain g2 is limited by the crosstalk of the', 'score': 0.537509560585022, 'page_num': 3, 'source_index': 1}, '[<4>]': {'content': '4 (c) (b) (a) FIG. 4. |↓⟩↔|↑⟩carrier excitation of nine-ion chain before and after shuttling. The horizontal axis is the global 729 nm beam probe time, and the vertical axis is the average ion excitation on the |↑⟩state. Error bars denote one standard deviation of the quantum projection noise. (a) Rabi oscillations of the sideband-cooled ions (red dots). The red line is a numerical simulation of thermal distribution with ¯nth = 4.0 ± 3.0. (b) Rabi oscillation after the transport at half speed of the transport function in Fig. 2(a). The blue line is a numerical simulation with with ¯nth = 4, ¯nα = 50 ± 5. (c) Rabi oscillation after the transport at full speed. The green line is a numerical simulation with with ¯nth = 4, ¯nα = 110 ± 5. addressing beam. The results indicate the major source of residual correlation is addressing crosstalk [39]. This can be mitigated by coupling the single photons into a single-mode fiber or improving the optical quality of the excitation beam. After characterizing the single-photon nature of the transport-multiplexing scheme, we characterize the motion of the ions introduced by shuttling. This is important as the quality of subsequent quantum operations on the ions or ion-photon entanglement will depend on the ions’ motional states. We further explore the system performance by measuring the motional heating from the ion transport. To do this, we first perform sideband cooling for all axial modes sequentially using the method similar to that in [41] and prepare the ion in the state |↓⟩= |42S1/2, mJ = −1/2⟩. We compare the |↓⟩↔|↑⟩= |32D5/2, mJ = −1/2⟩carrier transition before and after transport with a global 729 nm beam along the axial direction to determine how the transport affects the ion-motion (Fig. 4). The carrier Rabi flopping is motional state sensitive, and the Hamiltonian has the form of [42, 43] Hi,M = (ℏ/2)Ω(i)σ(i) x M Y m=1 exp \x02 iηi,m \x02 am + a† m \x01\x03 , (3) where Ω(i) is the Rabi frequency of the ith ion, am and a† m are the creation and annihilation operators on the mth mode, and ηi,m is the LD parameter of the ith ion and the mth mode. Considering the computational difficulty of including all motional modes in the simulation, we only consider the COM mode which we expect to be excited most because the electric fields, both from the transport and surface noise, are correlated over the whole ion string (see Supplemental [39]). Therefore, the average carrier Rabi flopping can be simplified to Pe(t) = 1 2N " 1 − ∞ X n=0 N X i=0 Pn cos \x10 Ω(i) n t \x11# , (4) where Pn is the occupation on the nth number state and encodes a convolution between a thermal and a coherent phonon distribution [44]. Ω(i) n is the Rabi frequency of the ith ion on the nth number state [45]. To verify the effectiveness of our approximation, we probe the sideband-cooled motional spectrum of the nine-ion chain before the transport and verify that only the COM mode is not cooled to near the ground state [39], for which we find a cooling limit of ¯nth = 4.0 ± 3.0. We also measure the electric-field noise induced heating and find a heating rate of 20 quanta / ms.(Fig. 4(a)), indicating that the remaining thermal population is likely limited by the COM mode heating which scales as ion number N [46]. Fig. 4(b) shows the carrier Rabi flopping after ion transport twice as slow as in Fig. 2(a). From numerical simulations (blue line), we find that the data can be explained well by a coherent state ¯nα = |α|2 ≈50 on the COM mode after the transport. Similarly, we perform the full-speed transport and the carrier Rabi flopping matches with COM coherent state with ¯nα ≈110 (Fig. 4(c)). As shown in the Rabi flopping plots, there is mismatch between the experimental data and numerical simulation at full speed, which could be due to thermal and coherent occupation of other modes and will require additional investigation. For example, one can use an individual 729 nm addressing beam to probe the blue sideband transition of different modes [47]. The optimal fast transport of a long ion chain remains an open question and is beyond the scope of this work. However,', 'score': 0.43412184715270996, 'page_num': 4, 'source_index': 1}, '[<5>]': {'content': '5 we note that further optimization can be done by energy self-neutral shuttling [31, 44], implementing closed-loop optimization of the shuttling function [48], etc. To summarize, we have presented a multiplexed ionphoton interface by transporting a nine-ion chain with synchronized excitation in sub-hundred µs. The speed is restricted by the motional frequency and can be increased by an order of magnitude, for instance, using a 3D-printed ion trap [49] with radial frequency beyond 10 MHz. The 397 nm photon can be converted to the telecommunication band via a two-step QFC [24]. Once integrated with state preparation on 32D3/2 Zeeman sublevel and photon collection with a single mode fiber, we expect a faster photon extraction rate [50] and negligible ion crosstalk while achieving high fidelity ion-photon entanglement [51, 52]. Our system can also be combined with a miniature cavity [35] for much higher photon extraction efficiency without sacrificing the photon generation rate, while the ion’s positional spread caused by coherent excitation can be mitigated by aligning the cavity along the radial direction or further optimization of the shuttling function. These results stimulate the research of fast shuttling of a chain of tens of ions as a unit cell of logical qubit with heralded entanglement [28] and highrates entanglement of quantum processors across large distances. During the resubmission of the manuscript, we became aware of related works on multiplexing for large scale quantum networks [53, 54]. B.Y and Q.W have contributed equally to this work. We thank Alp Sipahigil, Ben Lanyon, Tracy Northup, You-Wei Hsieh, and Wenji Wu for their helpful discussion. Q.W. and H.H. acknowledge funding by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences under Awards No. DE-SC0023277. This work is supported by the Office of Science (S.C.) in Advanced Scientific Computing Research (ASCR) through FOA - Quantum Internet to Accelerate Scientific Discovery (LAB 21-2495) and by NSF Grant No. 2016245. ∗qiming.wu@berkeley.edu † hhaeffner@berkeley.edu [1] H. J. Kimble, The quantum internet, Nature 453, 1023 (2008). [2] L.-M. Duan, M. D. Lukin, J. I. Cirac, and P. Zoller, Longdistance quantum communication with atomic ensembles and linear optics, Nature 414, 413 (2001). [3] T. van Leent, M. Bock, F. Fertig, R. Garthoff, S. Eppelt, Y. Zhou, P. Malik, M. Seubert, T. Bauer, W. Rosenfeld, et al., Entangling single atoms over 33 km telecom fibre, Nature 607, 69 (2022). [4] D. P. Nadlinger, P. Drmota, B. C. Nichol, G. Araneda, D. Main, R. Srinivas, D. M. Lucas, C. J. Ballance, K. Ivanov, E.-Z. Tan, et al., Experimental quantum key distribution certified by bell’s theorem, Nature 607, 682 (2022). [5] C. Monroe, R. Raussendorf, A. Ruthven, K. R. Brown, P. Maunz, L.-M. Duan, and J. Kim, Large-scale modular quantum-computer architecture with atomic memory and photonic interconnects, Physical Review A 89, 022317 (2014). [6] J. P. Covey, H. Weinfurter, and H. Bernien, Quantum networks with neutral atom processing nodes, npj Quantum Information 9, 90 (2023). [7] P. Komar, E. M. Kessler, M. Bishof, L. Jiang, A. S. Sørensen, J. Ye, and M. D. Lukin, A quantum network of clocks, Nature Physics 10, 582 (2014). [8] B. Nichol, R. Srinivas, D. Nadlinger, P. Drmota, D. Main, G. Araneda, C. Ballance, and D. Lucas, An elementary quantum network of entangled optical atomic clocks, Nature 609, 689 (2022). [9] X. Guo, C. R. Breum, J. Borregaard, S. Izumi, M. V. Larsen, T. Gehring, M. Christandl, J. S. NeergaardNielsen, and U. L. Andersen, Distributed quantum sensing in a continuous-variable entangled network, Nature Physics 16, 281 (2020). [10] D. L. Moehring, P. Maunz, S. Olmschenk, K. C. Younge, D. N. Matsukevich, L.-M. Duan, and C. Monroe, Entanglement of single-atom quantum bits at a distance, Nature 449, 68 (2007). [11] J. Hofmann, M. Krug, N. Ortegel, L. Gérard, M. Weber, W. Rosenfeld, and H. Weinfurter, Heralded entanglement between widely separated atoms, Science 337, 72 (2012). [12] H. Bernien, B. Hensen, W. Pfaff, G. Koolstra, M. S. Blok, L. Robledo, T. H. Taminiau, M. Markham, D. J. Twitchen, L. Childress, et al., Heralded entanglement between solid-state qubits separated by three metres, Nature 497, 86 (2013). [13] J. Schupp, V. Krcmarsky, V. Krutyanskiy, M. Meraner, T. Northup, and B. Lanyon, Interface between trappedion qubits and traveling photons with close-to-optimal efficiency, PRX quantum 2, 020331 (2021). [14] V. Krutyanskiy, M. Canteri, M. Meraner, J. Bate, V. Krcmarsky, J. Schupp, N. Sangouard, and B. P. Lanyon, Telecom-wavelength quantum repeater node based on a trapped-ion processor, Physical Review Letters 130, 213601 (2023). [15] V. Krutyanskiy, M. Canteri, M. Meraner, V. Krcmarsky, and B. Lanyon, Multimode ion-photon entanglement over 101 kilometers, PRX Quantum 5, 020308 (2024). [16] T. Pittman, B. Jacobs, and J. Franson, Single photons on pseudodemand from stored parametric down-conversion, Physical Review A 66, 042303 (2002). [17] F. Kaneda, B. G. Christensen, J. J. Wong, H. S. Park, K. T. McCusker, and P. G. Kwiat, Time-multiplexed heralded single-photon source, Optica 2, 1010 (2015). [18] Y. Pu, N. Jiang, W. Chang, H. Yang, C. Li, and L. Duan, Experimental realization of a multiplexed quantum memory with 225 individually accessible memory cells, Nature communications 8, 15359 (2017). [19] N. Sinclair, E. Saglamyurek, H. Mallahzadeh, J. A. Slater, M. George, R. Ricken, M. P. Hedges, D. Oblak, C. Simon, W. Sohler, et al., Spectral multiplexing for scalable quantum photonics using an atomic frequency comb quantum memory and feed-forward control, Physical review letters 113, 053603 (2014). [20] E. Saglamyurek, M. Grimau Puigibert, Q. Zhou, L. Giner, F. Marsili, V. B. Verma, S. Woo Nam, L. Oesterling, D. Nippa, D. Oblak, et al., A multiplexed', 'score': 0.28677064180374146, 'page_num': 5, 'source_index': 1}, '[<6>]': {'content': '6 light-matter interface for fibre-based quantum networks, Nature communications 7, 11202 (2016). [21] S. Zhang, J. Shi, Y. Liang, Y. Sun, Y. Wu, L. Duan, and Y. Pu, Fast delivery of heralded atom-photon quantum correlation over 12km fiber through multiplexing enhancement, arXiv preprint arXiv:2403.13623 (2024). [22] C. E. Bradley, J. Randall, M. H. Abobeih, R. Berrevoets, M. Degen, M. A. Bakker, M. Markham, D. Twitchen, and T. H. Taminiau, A ten-qubit solid-state spin register with quantum memory up to one minute, Physical Review X 9, 031045 (2019). [23] V. Krutyanskiy, M. Meraner, J. Schupp, and B. Lanyon, Polarisation-preserving photon frequency conversion from a trapped-ion-compatible wavelength to the telecom c-band, Applied Physics B 123, 228 (2017). [24] U. Saha, J. D. Siverns, J. Hannegan, Q. Quraishi, and E. Waks, Low-noise quantum frequency conversion of photons from a trapped barium ion to the telecom o-band, ACS Photonics 10, 2861 (2023), https://doi.org/10.1021/acsphotonics.3c00581. [25] P. Wang, C.-Y. Luan, M. Qiao, M. Um, J. Zhang, Y. Wang, X. Yuan, M. Gu, J. Zhang, and K. Kim, Single ion qubit with estimated coherence time exceeding one hour, Nature communications 12, 233 (2021). [26] P. Drmota, D. Main, D. Nadlinger, B. Nichol, M. Weber, E. Ainley, A. Agrawal, R. Srinivas, G. Araneda, C. Ballance, et al., Robust quantum memory in a trapped-ion quantum network node, Physical Review Letters 130, 090803 (2023). [27] W. Huie, S. G. Menon, H. Bernien, and J. P. Covey, Multiplexed telecommunication-band quantum networking with atom arrays in optical cavities, Physical Review Research 3, 043154 (2021). [28] Y. Li and J. Thompson, High-rate and high-fidelity modular interconnects between neutral atom quantum processors, arXiv preprint arXiv:2401.04075 (2024). [29] P. Dhara, N. M. Linke, E. Waks, S. Guha, and K. P. Seshadreesan, Multiplexed quantum repeaters based on dual-species trapped-ion systems, Physical Review A 105, 022623 (2022). [30] J. Ramette, J. Sinclair, Z. Vendeiro, A. Rudelis, M. Cetina, and V. Vuletić, Any-to-any connected cavitymediated architecture for quantum computing with trapped ions or rydberg arrays, PRX Quantum 3, 010344 (2022). [31] R. Bowler, J. Gaebler, Y. Lin, T. R. Tan, D. Hanneke, J. D. Jost, J. Home, D. Leibfried, and D. J. Wineland, Coherent diabatic ion transport and separation in a multizone trap array, Physical review letters 109, 080502 (2012). [32] J. M. Pino, J. M. Dreiling, C. Figgatt, J. P. Gaebler, S. A. Moses, M. Allman, C. Baldwin, M. Foss-Feig, D. Hayes, K. Mayer, et al., Demonstration of the trapped-ion quantum ccd computer architecture, Nature 592, 209 (2021). [33] S. Moses, C. Baldwin, M. Allman, R. Ancona, L. Ascarrunz, C. Barnes, J. Bartolotta, B. Bjork, P. Blanchard, M. Bohn, et al., A race track trapped-ion quantum processor, arXiv preprint arXiv:2305.03828 (2023). [34] A. E. Webb, S. C. Webster, S. Collingbourne, D. Bretaud, A. M. Lawrence, S. Weidt, F. Mintert, and W. K. Hensinger, Resilient entangling gates for trapped ions, Physical review letters 121, 180501 (2018). [35] H. Takahashi, E. Kassa, C. Christoforou, and M. Keller, Strong coupling of a single ion to an optical cavity, Physical review letters 124, 013602 (2020). [36] H. Takahashi, A. Wilson, A. Riley-Watson, F. Oručević, N. Seymour-Smith, M. Keller, and W. Lange, An integrated fiber trap for single-ion photonics, New Journal of Physics 15, 053011 (2013). [37] M. W. van Mourik, P. Hrmo, L. Gerster, B. Wilhelm, R. Blatt, P. Schindler, and T. Monz, rf-induced heating dynamics of noncrystallized trapped ions, Physical Review A 105, 033101 (2022). [38] A. Tobalina, E. Torrontegui, I. Lizuain, M. Palmero, and J. G. Muga, Invariant-based inverse engineering of time-dependent, coupled harmonic oscillators, Physical Review A 102, 063112 (2020). [39] See supplemental materials. [40] F. Diedrich and H. Walther, Nonclassical radiation of a single stored ion, Physical review letters 58, 203 (1987). [41] Y. Yu, N. R. Hutzler, J. T. Zhang, L. R. Liu, J. D. Hood, T. Rosenband, and K.-K. Ni, Motional-ground-state cooling outside the lamb-dicke regime, Physical Review A 97, 063423 (2018). [42] D. J. Wineland, C. Monroe, W. M. Itano, D. Leibfried, B. E. King, and D. M. Meekhof, Experimental issues in coherent quantum-state manipulation of trapped atomic ions, Journal of research of the National Institute of Standards and Technology 103, 259 (1998). [43] Q. Wu, Y. Shi, and J. Zhang, Continuous raman sideband cooling beyond the lamb-dicke regime in a trapped ion chain, Physical Review Research 5, 023022 (2023). [44] A. Walther, F. Ziesel, T. Ruster, S. T. Dawkins, K. Ott, M. Hettrich, K. Singer, F. Schmidt-Kaler, and U. Poschinger, Controlling fast transport of cold trapped ions, Physical review letters 109, 080501 (2012). [45] D. Leibfried, R. Blatt, C. Monroe, and D. Wineland, Quantum dynamics of single trapped ions, Reviews of Modern Physics 75, 281 (2003). [46] M. Joshi, A. Fabre, C. Maier, T. Brydges, D. Kiesenhofer, H. Hainzer, R. Blatt, and C. Roos, Polarization-gradient cooling of 1d and 2d ion coulomb crystals, New Journal of Physics 22, 103013 (2020). [47] G.-X. Wang, Y.-K. Wu, R. Yao, W.-Q. Lian, Z.-J. Cheng, Y.-L. Xu, C. Zhang, Y. Jiang, Y.-Z. Xu, B.-X. Qi, et al., Simulating the spin-boson model with a controllable reservoir in an ion trap, arXiv preprint arXiv:2402.07461 (2024). [48] J. D. Sterk, H. Coakley, J. Goldberg, V. Hietala, J. Lechtenberg, H. McGuinness, D. McMurtrey, L. P. Parazzoli, J. Van Der Wall, and D. Stick, Closed-loop optimization of fast trapped-ion shuttling with sub-quanta excitation, npj Quantum Information 8, 68 (2022). [49] S. Xu, X. Xia, Q. Yu, S. Khan, E. Megidish, B. You, B. Hemmerling, A. Jayich, J. Biener, and H. Häffner, 3dprinted micro ion trap technology for scalable quantum information processing, arXiv preprint arXiv:2310.00595 (2023). [50] C. Crocker, M. Lichtman, K. Sosnova, A. Carter, S. Scarano, and C. Monroe, High purity single photons entangled with an atomic qubit, Optics express 27, 28143 (2019). [51] L. Stephenson, D. Nadlinger, B. Nichol, S. An, P. Drmota, T. Ballance, K. Thirumalai, J. Goodwin, D. Lucas, and C. Ballance, High-rate, high-fidelity entanglement of qubits across an elementary quantum network, Physical review letters 124, 110501 (2020). [52] J. O’Reilly, G. Toh, I. Goetting, S. Saha, M. Sha-', 'score': 0.22642064094543457, 'page_num': 6, 'source_index': 1}, '[<7>]': {'content': '7 laev, A. Carter, A. Risinger, A. Kalakuntla, T. Li, A. Verma, et al., Fast photon-mediated entanglement of continuously-cooled trapped ions for quantum networking, arXiv preprint arXiv:2404.16167 (2024). [53] L. Hartung, M. Seubert, S. Welte, E. Distante, and G. Rempe, A quantum-network register assembled with optical tweezers in an optical cavity, Science 385, 179 (2024). [54] M. Canteri, Z. Koong, J. Bate, A. Winkler, V. Krutyanskiy, and B. Lanyon, A photon-interfaced ten qubit quantum network node, arXiv preprint arXiv:2406.09480 (2024).', 'score': 0.6625192761421204, 'page_num': 7, 'source_index': 1}}
"""


system_prompt = """You are a deep thinking tutor helping a student reading a paper.

MATH RENDERING — HARD RULES (must follow):
- Wrap ALL math in $...$ (inline) or $$...$$ (display). Never write bare math.
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


from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import os, re
from typing import Iterable

def _format_thinking_delta(delta: str) -> str:
    """
    Only transform '**XXX' -> '\n\n**XXX'.
    If the chunk is exactly '**', or starts with '**' followed by a newline
    (e.g., '**\\n', '**\\r\\n') or only whitespace, treat it as a closing marker
    and do nothing.
    """
    if not delta:
        return delta

    if delta == "**":
        return delta

    if delta.startswith("**"):
        after = delta[2:]
        # If the very next char is a newline, or there's only whitespace after '**',
        # it's likely a closing '**' chunk -> leave unchanged.
        if after[:1] in ("\n", "\r") or after.strip() == "":
            return delta
        # Otherwise it's an opening '**Title' chunk -> add two leading newlines
        if not delta.startswith("\n\n**"):
            return "\n\n" + delta

    return delta

def stream_response_with_tags(**create_kwargs) -> Iterable[str]:
    """
    Yields a single XML-like stream:
      <thinking> ...reasoning summary + tool progress... </thinking><response> ...final answer... </response>
    Without detailed tool calling updates inside <thinking>.
    """
    load_dotenv(".env")
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
        api_version="2025-03-01-preview",
    )
    stream = client.responses.create(stream=True, **create_kwargs)

    # Show a thinking container immediately
    thinking_open = True
    response_open = False
    yield "<thinking>"

    try:
        for event in stream:
            t = event.type

            # --- Reasoning summary stream ---
            if t == "response.reasoning_summary_text.delta":
                yield _format_thinking_delta(event.delta)

            elif t == "response.reasoning_summary_text.done":
                # keep <thinking> open for tool progress; we'll close when answer starts or at the very end
                pass

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
    load_dotenv(".env")
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
        instructions=f"{system_prompt}\n\n You should search the web as needed (multiple searches OK) and cite sources.",
        input=f"Context from the paper: {context_from_paper}\n\n What is this paper mainly about? Do web search if needed to find related multiplexing papers and compare with this paper.",
        # input="What is Second law of thermodynamics , search Wikipedia to get answers",
    )

    for chunk in stream_response_with_tags(**kwargs):
        print(chunk, end="", flush=True)
    print()