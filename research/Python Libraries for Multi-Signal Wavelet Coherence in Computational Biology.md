

# **Advanced Comparative Analysis: Python Libraries for Multi-Signal Wavelet Coherence in Computational Biology**

## **I. Foundational Principles and Mathematical Framework**

### **1.1 The Theoretical Imperative: CWT for Non-Stationary Biological Signals**

The analysis of biological signals presents unique challenges due to their inherently non-stationary nature. Time series derived from electrophysiology, genomics, or chronobiology—such as those contained within CircaDB—are characterized by spectral content that fluctuates over time, reflecting changes in biological state or regulatory mechanisms.1 Classical Fourier analysis, which operates under the assumption that the spectral properties of a signal remain constant across the entire duration of observation (stationarity), is therefore fundamentally inadequate for capturing critical transient dynamics present in complex biological systems.1  
The Continuous Wavelet Transform (CWT) overcomes these limitations by providing simultaneous resolution in both the time and frequency (scale) domains. This decomposition allows researchers to localize specific oscillatory events and track how the spectral composition evolves over the signal duration. For analysis in geophysical and computational biology, the Morlet wavelet stands as the definitive standard.2 This choice is justified by the Morlet wavelet's structure—a plane wave multiplied by a Gaussian envelope—which provides the optimal balance, dictated by the Heisenberg uncertainty principle, between temporal precision and frequency resolution. Critically, the primary Python library for this domain,  
pycwt, adheres precisely to the standard Morlet wavelet definition formalized in the canonical paper by Torrence and Compo (1998).3 This historical compatibility grounds the  
pycwt framework as the mathematical reference point in Python for publication-ready analyses, guaranteeing reproducibility against widely accepted scientific benchmarks.

### **1.2 Multi-Signal Wavelet Analysis: XWT and WCT**

For investigating the dynamic relationship between two time series—for example, measuring the coupling between two biological oscillators or regulatory pathways—multi-signal wavelet analysis is required. The analytical progression moves from individual CWTs to the Cross-Wavelet Transform (XWT) and finally to Wavelet Coherence Transform (WCT).  
The XWT is calculated as the product of the CWT of signal ![][image1] and the complex conjugate of the CWT of signal ![][image1] (![][image1]). This transformation identifies regions in time-scale space where both signals share high common power, simultaneously providing the relative phase relationship (lead/lag) between the two signals.4  
The WCT builds upon the XWT to quantify the linear correlation, or coherence, between the signals in time-scale space. WCT is defined as the squared magnitude of the smoothed cross-wavelet spectrum, normalized by the smoothed individual wavelet spectra. The resulting metric behaves similarly to a time-localized correlation coefficient, bounded between 0 (no relationship) and 1 (perfect coupling).3 The Python module  
pycwt is explicitly engineered for this purpose, providing high-level, dedicated routines specifically named xwt (cross-wavelet transform) and wct (wavelet coherence transform) as core functionalities.3

### **1.3 The Cone of Influence (COI) and Edge Effects**

A central constraint in wavelet analysis is the treatment of edge effects, which is formalized through the Cone of Influence (COI). The COI mathematically delineates the region within the time-scale domain where the wavelet power or coherence results are statistically unreliable due to the finite length of the input time series and the convolution process.3  
The interpretation of any result falling within the COI is generally unreliable and must be avoided. The pycwt.wct function ensures this boundary is managed by returning the COI as a critical output vector, indicating the maximum useful Fourier period at each point in time.3 For biological data, especially those concerning long-period phenomena like the  
![][image1]\-hour circadian rhythm observed in datasets like CircaDB, the COI represents a significant limitation. Resolving circadian periods requires extensive time series data; if the recording duration is insufficiently long, these periods will be confined to the COI, rendering the results statistically inconclusive. Therefore, a successful wavelet analysis pipeline must demonstrate proficiency in resolving shorter, yet biologically relevant, oscillations, such as ultradian rhythms (periods of ![][image1] or ![][image1] hours).6 These shorter scales are more likely to fall outside the COI, confirming the methodology’s successful time-frequency localization and validity for investigating transient biological coupling dynamics.3

## **II. Comparative Assessment of Python Library Implementations**

This critical assessment evaluates the functional completeness, architectural efficiency, and specialization of the primary Python libraries available for wavelet analysis.

### **2.1 PyCWT: Functional Completeness and Statistical Integration**

pycwt is identified as the mandatory primary tool for fulfilling the core requirements of multi-signal coherence analysis. It is explicitly designed for continuous wavelet spectral analysis and integrates dedicated routines for XWT and WCT.4 The library provides standard wavelet families including Morlet, Paul, and Derivative of a Gaussian (DOG).3  
The computational method employed by pycwt relies on the Fast Fourier Transform (FFT) algorithm for calculation of the wavelet transform.4 While this approach maintains rigorous mathematical consistency with the original geophysical methodology developed by Torrence and Compo, it presents a functional trade-off. For the analysis of extremely large or high-sampling-rate biological signals—such as massive EEG datasets—the FFT-based approach may introduce performance bottlenecks when compared to libraries leveraging lower-level C/Cython implementations or GPU acceleration. Despite this potential computational limitation,  
pycwt's indispensable role is affirmed by its native integration of high-level statistical analysis features, including routines for significance testing.3

### **2.2 PyWavelets: High-Performance CWT Engine and Feature Gaps**

PyWavelets (pywt) serves a distinct and necessary role as a high-performance CWT engine. Its architecture is built upon low-level C and Cython code, optimizing the computation of wavelet transforms for speed.7 This design often grants  
PyWavelets a significant speed advantage over purely Python or FFT-based alternatives for general transform calculation.9  
The library is comprehensive in its support for wavelet primitives, offering 1D Continuous Wavelet Transform (CWT) calculation 2 alongside an extensive portfolio of over 100 built-in wavelet filters, and its results are explicitly designed to be compatible with the proprietary MATLAB Wavelet Toolbox (TM).7 However,  
PyWavelets lacks native, high-level support for cross-wavelet or coherence analysis routines.7 Consequently, it cannot function as the primary tool for WCT/XWT. Strategically,  
PyWavelets is viewed as an essential architectural component. For future production environments requiring maximal computational efficiency, the core CWT complex coefficients for signals ![][image1] and ![][image1] could be rapidly generated using pywt.cwt 2, and these highly optimized coefficients could then be seamlessly integrated into  
pycwt’s proven and statistically robust WCT calculation and significance routines.

### **2.3 ssqueezepy: Advanced Time-Frequency Resolution**

The ssqueezepy library represents the vanguard of time-frequency analysis in Python. This module specializes in the Synchrosqueezing CWT (SSQ\_CWT), a highly refined post-processing technique that significantly sharpens the time-frequency localization of the coefficients compared to standard CWT. The implementation is optimized for performance, with benchmarks indicating that it can surpass MATLAB in computational speed, offering native support for multi-threaded CPU operation and optional acceleration via GPU libraries like CuPy and PyTorch.9  
In biological signal analysis, where subtle frequency changes or the differentiation of closely spaced oscillatory components (e.g., distinguishing adjacent neural bands or resolving overlapping ultradian rhythms 6) are crucial for mechanistic understanding, SSQ\_CWT offers superior resolution. Although  
ssqueezepy does not natively provide XWT or WCT routines, its capacity for generating high-fidelity CWT coefficients establishes it as a critical asset. Integrating SSQ-filtered coefficients into a customized cross-spectrum framework represents a high-level optimization goal for achieving maximal precision in the analysis of highly non-stationary or noisy biological data.  
A summary of the core capabilities and specializations of the leading Python packages for multi-signal wavelet analysis is provided below:  
Table: Core Feature Matrix for Multi-Signal Wavelet Libraries (XWT/WCT Focus)

| Library | Core Analysis Type | XWT/WCT Support | Primary Wavelets | Native Significance Testing | Licensing | Maintenance Velocity (2023-2025) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| pycwt | CWT, Spectral Analysis (FFT-based) 4 | Explicit and Native (wct, xwt) 4 | Morlet, Paul, DOG 3 | Red Noise, Monte Carlo 3 | BSD-style 4 | Stable (Functionally complete) |
| PyWavelets | DWT, 1D CWT (C/Cython-based) 7 | None Native 7 | \>100 Built-in Filters 7 | None | MIT 7 | High (Active 2024 development) 7 |
| ssqueezepy | CWT, Synchrosqueezing (Optimized) 9 | None Native | Generalized Morse | None | TBD | High (Performance-driven) |

## **III. Implementation of Rigorous Statistical Significance Testing**

Achieving publication-ready results requires statistical validation that rigorously assesses whether the observed wavelet power or coherence is genuinely non-random, rather than an artifact of the data's underlying noise properties. This mandates moving beyond simple theoretical noise models toward data-driven surrogate methods.

### **3.1 The Limitations of the Theoretical Null Hypothesis**

The pycwt library provides built-in routines for performing significance testing on CWT results against a theoretical null hypothesis, specifically the ![][image1] test against a Red Noise (AR(1)) background, following the standard procedure outlined in Torrence and Compo (1998).3 This approach relies on the  
pycwt.significance function, which can compute the theoretical red-noise spectrum, fft\_theor, as a function of period.3  
However, reliance solely on this theoretical null hypothesis, often based on white or red noise, is statistically unsound for complex biological and ecological time series.10 Biological systems frequently exhibit complex memory, non-linear dependencies, or autocorrelation structures that extend beyond the simplest autoregressive model (AR(1)). When the null hypothesis fails to capture the complexity of the true background noise, the significance thresholds are set artificially low, leading to an increased probability of Type I errors (false positives).10 For robust research, the statistical assessment must be based on data-driven resampling techniques.

### **3.2 Surrogate Data Generation Methods: Building an Empirical Null**

To establish an appropriate empirical null distribution, the WCT must be calculated thousands of times on synthetic signal pairs (surrogates) that preserve specific statistical properties of the original data while disrupting the phase relationship or coupling being tested.10  
The pycwt framework supports this advanced approach through pycwt.wct\_significance, which is designed for Monte Carlo simulation of significance.3 Crucially, the user is responsible for supplying the statistically relevant surrogate data sets to this routine.

#### **Fourier Phase Randomization**

One of the most common and robust data-driven methods is Fourier Phase Randomization. This technique creates surrogate time series that retain the exact power spectrum of the original data, thereby preserving the signal's frequency content and variance, but randomize the phases. The resulting surrogates represent a linear Gaussian process with the same power spectrum as the original signal.12 Testing the observed coherence against a set of surrogates generated this way rigorously evaluates whether the coherence is due to a non-linear or non-Gaussian mechanism.

#### **AR(p) Surrogates and Permutation Testing**

For biological data where complex temporal memory may be present, Autoregressive (AR) surrogates of order ![][image1] are necessary. These surrogates preserve the estimated autocorrelation structure of the original signal.12 This simulation requires interfacing  
pycwt with external Python libraries, such as statsmodels, to estimate high-order AR or ARIMA coefficients for the input signals, generating thousands of simulations that maintain the appropriate time-series 'memory' before submitting them to the pycwt.wct calculation.10  
Alternatively, non-parametric Permutation Testing provides a powerful means to assess the hypothesis of non-random coupling. Permutation tests involve randomly shuffling elements of the data to create an empirical null distribution for a given statistic.13 For WCT, this entails shuffling the temporal sequence or pairing between time series  
![][image1] and ![][image1] to evaluate the null hypothesis that there is no meaningful coupling in time-scale space.14 Python packages such as  
scipy.stats.permutation\_test 14 or  
PyPermut 15 offer robust frameworks for conducting these randomization tests, which are essential when complex, unknown noise characteristics preclude the use of parametric models.

### **3.3 Comparative Statistical Implementation Protocol**

The following table summarizes the necessity and implementation strategy for integrating rigorous statistical hypothesis testing into the multi-signal wavelet analysis pipeline:  
Table: Comparison of Wavelet Significance Testing Methodologies for Biological Data

| Null Model Type | Underlying Hypothesis Tested | Standard Use Case | PyCWT Implementation | Recommended Python Integration (External) |
| :---- | :---- | :---- | :---- | :---- |
| Theoretical Red Noise (AR(1)) | Significance against simple background noise | Preliminary CWT Analysis (quick screening) | pycwt.significance (sigma\_test=0/1) 3 | Direct, for comparison only. |
| Monte Carlo Simulation | Non-random coherence / Common power | Primary WCT/XWT Validation (requires surrogates) | pycwt.wct\_significance 3 | statsmodels (for AR(p) generation); run WCT ![][image1] times 10 |
| Fourier Phase Randomization | Periodicity is not due to non-linear structure | Signals prone to non-linear coupling (e.g., neural) | Custom generation algorithm (via NumPy/SciPy FFT) | Mandatory generation step followed by Monte Carlo 12 |
| Permutation Testing | Correlation/Coherence is non-random | Non-parametric testing of coupling 13 | Custom statistical comparison wrapper | scipy.stats.permutation\_test or PyPermut 14 |

## **IV. Validation Protocol: Biological and Synthetic Benchmarking**

A validated computational pipeline for wavelet coherence must be rigorously tested against datasets possessing known, ground-truth temporal structures to ensure accuracy and statistical precision.

### **4.1 Utilizing Canonical Biological Datasets (CircaDB and BioClock)**

Validation requires the use of publicly available biological data repositories with defined periodic components.

#### **Circadian Rhythm Validation (CircaDB)**

CircaDB provides comprehensive multi-omics time-series profiles from mouse and human subjects, specifically engineered to capture phenomena with known ![][image1]\-hour periodicity.16 These datasets are crucial for validating the pipeline's ability to detect coupling at the long-scale (circadian) level, such as the correlation between two oscillating gene transcripts.

#### **Ultradian Rhythm Validation (BioClock/Physiology)**

While circadian periods offer a long-scale benchmark, biological signals often feature higher-frequency components. For example, physiological recordings, such as blood pressure analysis, commonly reveal ultradian rhythms with periods around ![][image1] hours superimposed on the diurnal cycle.6 Furthermore, traditional Discrete Fourier Transform (DFT) methods are often incapable of revealing these ultradian periods, frequently showing only harmonics of the primary 24-hour cycle.1  
Therefore, the wavelet pipeline must be explicitly validated on its capacity to resolve these shorter, potentially non-sinusoidal ultradian rhythms. The successful detection and localization of these ultradian scales, crucially outside the constraints of the COI, confirm the CWT's superior temporal and frequency localization capabilities compared to global analysis techniques.6 This validation demonstrates the utility of the chosen methodology in identifying dynamic patterns that would otherwise be obscured.

### **4.2 Synthetic Data Generators for Controlled Validation**

Synthetic data generation is essential for creating controlled environments where the ground truth is precisely known, allowing for quantitative verification of the library's accuracy, especially regarding phase shifts and non-stationary frequency characteristics.17

#### **Multi-Component Signal Generators**

The simplest form of synthetic data involves generating signals by summing known sine wave components (e.g., ![][image1]\-unit and ![][image1]\-unit periods) with added normally distributed random noise.18 This controlled construction verifies the pipeline's basic functionality in resolving and isolating defined periodicities and scale resolutions.

#### **Fourier-based Statistical Preservation**

A more advanced class of generators utilizes methods based on the Discrete Fourier Transform to generate synthetic time series that accurately mimic the statistical moments, autocorrelation function, and power spectrum of a real input signal.17 These generators are invaluable because they allow the creation of surrogate datasets that are statistically relevant to the observed biological signal. This quantitative capability allows the validation protocol to be directly integrated with the statistical rigor outlined in Section III, ensuring that Monte Carlo simulations used for significance testing are based on realistic null processes.17

#### **Generative Models for Event Data**

For time series related to discrete biological events (e.g., neural spike trains or event onsets), specialized generative models are necessary. The Gaussian Mixture Periodicity Detection Algorithm (GMPDA), for instance, introduces the Clock Model and Random Walk Model for generating binary time series with complex periodic onsets and varying noise levels.19 These sophisticated models provide a reliable ground truth for testing the pipeline's sensitivity and accuracy when analyzing non-continuous, noisy biological event data.

## **V. Interactive Visualization Pipeline: Plotly and Bokeh**

Wavelet Coherence output is inherently high-dimensional, represented by a complex matrix of time, scale, magnitude, and phase. Effective scientific communication and interpretation necessitate an interactive visualization layer that clearly delineates critical information, such as the COI and statistical significance boundaries.

### **5.1 Visualization Components and Plotly Primitives**

Plotly is an effective tool for high-fidelity, interactive visualization within Python notebooks or detailed reporting environments.

#### **Scalogram and Coherence Heatmap**

The core output—the magnitude of the XWT or WCT coefficients—must be rendered as a two-dimensional heatmap (Time x Scale matrix) to intuitively display regions of intense power or correlation.20 Plotly's  
px.imshow() function is well-suited for transforming this 2D numerical array into colored rectangular tiles, which represent the scalogram or coherence map.20

#### **Significance Contours Overlay**

The statistical boundaries derived from the Monte Carlo simulation (Section III) must be visually overlaid onto the heatmap to distinguish truly significant features from background fluctuations. Plotly's go.Contour primitive is the necessary tool for this task.21 Contour plots are designed to show interpolated lines of isovalues of a 2D array, allowing the  
![][image1] confidence region calculated by pycwt.wct\_significance to be clearly delineated as a contour line or filled area on the coherence map.3

#### **COI Delineation**

The Cone of Influence boundary requires clear demarcation to serve as a mandatory visual warning against misinterpreting edge artifacts. This can be achieved in Plotly using custom shape layers, such as a filled polygon or hatching 22, to mask the affected region. Furthermore, the phase relationships (lead/lag) determined by the WCT output must be visualized, typically through quiver plots or annotations that show vector direction across significant regions.

### **5.2 Interactive Features and Web Deployment**

Interactive features—including pan, zoom, hover tools to inspect specific time-scale values, and parameter adjustment widgets—are crucial for navigating the complexity of wavelet output.  
Bokeh is a robust visualization library explicitly designed for creating highly interactive graphs and visualizations optimized for web deployment, utilizing HTML and JavaScript.23 This makes Bokeh a superior choice over traditional static plotting libraries for creating dynamic dashboards intended for collaborative research or for presenting results to stakeholders.25 A production architecture will therefore often employ Plotly for detailed analysis and debugging during the development phase, and Bokeh for user-facing, high-performance web applications.  
The functional mapping between core wavelet outputs and the required interactive visualization components is detailed below:  
Table: Mapping Wavelet Components to Interactive Visualization Primitives

| Wavelet Output | Dimension/Type | Primary Role | Plotly Implementation Primitive | Key Interactive Feature |
| :---- | :---- | :---- | :---- | :---- |
| Scalogram/Coherence Magnitude | 2D Array (Time x Scale) | Core Visualization | Heatmap (px.imshow/go.Heatmap) 20 | Zoom, Hover (Time/Scale/Value) |
| Significance Contours | 2D Binary Mask/Threshold | Statistical Boundary | Contour Plot (go.Contour) 21 | Widget for dynamic confidence level adjustment |
| Cone of Influence (COI) | Time-Varying Period Vector | Interpretation Boundary | Custom Shape/Mask Layer (e.g., filled polygon or hatching) 22 | Static boundary, mandatory visual warning |
| Phase Arrows (WCT) | 2D Vector Field (Angle/Direction) | Causal Relationship (Lead/Lag) | Custom Annotation/Quiver Layer | Hover for precise phase angle value |

## **VI. Production Readiness and Strategic Recommendations**

### **6.1 Licensing and Maintenance Stability (2023-2025)**

The licensing terms for the primary and secondary tools are highly permissive, mitigating legal constraints for proprietary research development. pycwt is released under a BSD-style open-source license 4, and  
PyWavelets is released under the MIT license.7  
Regarding maintenance stability, PyWavelets demonstrates high velocity and strong community health, with recent commits recorded as "last week" and a long history of professional development and maintenance.7 While detailed 2023-2025 activity metrics for  
pycwt are less explicit in the public domain, its functional completeness suggests stability for its core mission. However, the specialized nature of pycwt introduces dependency risk. This risk is effectively managed by the robustness and high maintenance velocity of complementary, CWT-capable libraries like PyWavelets and ssqueezepy. Should pycwt development slow, the core CWT coefficient generation, which is the most computationally intensive step, can be safely migrated to these actively supported, high-performance engines, ensuring the long-term architectural stability of the entire pipeline.

### **6.2 Final Strategic Recommendations for Tool Selection**

The investigation yields a clear, multi-tiered strategy for deploying a production-ready multi-signal wavelet analysis pipeline:

1. **Primary Analytical Framework:** The **pycwt** library must serve as the primary engine for high-level XWT and WCT calculation. This selection is non-negotiable due to its inherent mathematical grounding in the Torrence & Compo methodology and its native integration of the specific cross-wavelet and coherence routines required.3  
2. **Statistical Augmentation:** A mandatory, customized workflow must be implemented to generate data-driven surrogate datasets (specifically using AR(p) models or Fourier phase randomization) utilizing external statistical packages. These surrogates must then be processed via pycwt.wct\_significance to establish an empirical null distribution.10 This step is critical for ensuring the resulting significance boundaries are biologically relevant and meet peer-review standards.  
3. **Visualization Strategy:** Adopt **Plotly** for creating high-fidelity, interactive scalograms and coherence maps in development and reporting environments. The visualization must focus on compositing the WCT magnitude (Heatmap) with the statistical boundaries (Contour plots) and clearly defining the COI region.20 For user deployment, the pipeline should integrate  
   **Bokeh** to deliver web-ready, interactive dashboards.

### **6.3 Future Directions: Integrating High-Resolution Methods**

For biological signals characterized by extreme noise or highly dynamic frequency changes, the next evolution of the wavelet analysis pipeline should incorporate advanced time-frequency techniques. This involves moving beyond standard Morlet CWT coefficients by utilizing **ssqueezepy** to generate high-resolution Synchrosqueezing CWT output.9 These enhanced coefficients can then be input into a customized WCT calculation. This approach leverages the superior localization of Synchrosqueezing to potentially resolve fine-grained, subtle shifts in coupling dynamics that may be blurred or obscured by traditional CWT methods, maximizing the informational yield from complex biological time series.

#### **Works cited**

1. Wavelet analysis of circadian and ultradian behavioral rhythms \- PMC \- PubMed Central, accessed October 2, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3717080/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3717080/)  
2. Continuous Wavelet Transform (CWT) — PyWavelets Documentation \- Read the Docs, accessed October 2, 2025, [https://pywavelets.readthedocs.io/en/latest/ref/cwt.html](https://pywavelets.readthedocs.io/en/latest/ref/cwt.html)  
3. Reference \- PyCWT: spectral analysis using wavelets in Python, accessed October 2, 2025, [https://pycwt.readthedocs.io/en/latest/reference/](https://pycwt.readthedocs.io/en/latest/reference/)  
4. regeirk/pycwt: A Python module for continuous wavelet ... \- GitHub, accessed October 2, 2025, [https://github.com/regeirk/pycwt](https://github.com/regeirk/pycwt)  
5. Wavelet Analysis with Pyleoclim \- Linked Earth, accessed October 2, 2025, [http://linked.earth/PyleoTutorials/notebooks/L2\_wavelet\_analysis.html](http://linked.earth/PyleoTutorials/notebooks/L2_wavelet_analysis.html)  
6. WAVECLOCK: wavelet analysis of circadian oscillation \- PMC \- PubMed Central, accessed October 2, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2639275/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2639275/)  
7. PyWavelets \- Wavelet Transforms in Python — PyWavelets ..., accessed October 2, 2025, [https://pywavelets.readthedocs.io/](https://pywavelets.readthedocs.io/)  
8. Wavelet Transforms in Python — PyWavelets Documentation, accessed October 2, 2025, [https://pywavelets.readthedocs.io/en/v0.5.1/](https://pywavelets.readthedocs.io/en/v0.5.1/)  
9. OverLordGoldDragon/ssqueezepy: Synchrosqueezing, wavelet transforms, and time-frequency analysis in Python \- GitHub, accessed October 2, 2025, [https://github.com/OverLordGoldDragon/ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy)  
10. Wavelet analysis in ecology and epidemiology: impact of statistical tests \- PubMed Central, accessed October 2, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3869155/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3869155/)  
11. WaveletComp 1.1: A guided tour through the R package, accessed October 2, 2025, [http://www.hs-stat.com/projects/WaveletComp/WaveletComp\_guided\_tour.pdf](http://www.hs-stat.com/projects/WaveletComp/WaveletComp_guided_tour.pdf)  
12. Help for package WaveletComp, accessed October 2, 2025, [https://cran.r-project.org/web/packages/WaveletComp/refman/WaveletComp.html](https://cran.r-project.org/web/packages/WaveletComp/refman/WaveletComp.html)  
13. Permutation testing | Python, accessed October 2, 2025, [https://campus.datacamp.com/courses/statistical-simulation-in-python/resampling-methods?ex=11](https://campus.datacamp.com/courses/statistical-simulation-in-python/resampling-methods?ex=11)  
14. permutation\_test — SciPy v1.16.2 Manual, accessed October 2, 2025, [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation\_test.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)  
15. qbarthelemy/PyPermut: Python package for permutation tests, for statistics and machine learning. \- GitHub, accessed October 2, 2025, [https://github.com/qbarthelemy/PyPermut](https://github.com/qbarthelemy/PyPermut)  
16. RhythmicDB: A Database of Predicted Multi-Frequency Rhythmic Transcripts \- PMC, accessed October 2, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9237250/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9237250/)  
17. Synthetic Random Environmental Time Series Generation with Similarity Control, Preserving Original Signal's Statistical Characteristics \- arXiv, accessed October 2, 2025, [https://arxiv.org/html/2502.02392v1](https://arxiv.org/html/2502.02392v1)  
18. Using Wavelets to Analyze Time Series Data \- UVA Library \- The University of Virginia, accessed October 2, 2025, [https://library.virginia.edu/data/articles/using-wavelets-analyze-time-series-data](https://library.virginia.edu/data/articles/using-wavelets-analyze-time-series-data)  
19. Generative Models for Periodicity Detection in Noisy Signals \- MDPI, accessed October 2, 2025, [https://www.mdpi.com/2624-5175/6/3/25](https://www.mdpi.com/2624-5175/6/3/25)  
20. Heatmaps in Python \- Plotly, accessed October 2, 2025, [https://plotly.com/python/heatmaps/](https://plotly.com/python/heatmaps/)  
21. Contour plots in Python \- Plotly, accessed October 2, 2025, [https://plotly.com/python/contour-plots/](https://plotly.com/python/contour-plots/)  
22. ct6502/wavelets: Torrence & Compo Wavelet Analysis Software \- GitHub, accessed October 2, 2025, [https://github.com/ct6502/wavelets](https://github.com/ct6502/wavelets)  
23. Data Visualization in Python using Bokeh \[Easy Guide\] \- Simplilearn.com, accessed October 2, 2025, [https://www.simplilearn.com/tutorials/python-tutorial/python-bokeh](https://www.simplilearn.com/tutorials/python-tutorial/python-bokeh)  
24. Python Bokeh tutorial \- Interactive Data Visualization with Bokeh \- GeeksforGeeks, accessed October 2, 2025, [https://www.geeksforgeeks.org/data-visualization/python-bokeh-tutorial-interactive-data-visualization-with-bokeh/](https://www.geeksforgeeks.org/data-visualization/python-bokeh-tutorial-interactive-data-visualization-with-bokeh/)  
25. Data Visualization with Bokeh in Python, Part I: Getting Started | by Will Koehrsen \- Medium, accessed October 2, 2025, [https://medium.com/data-science/data-visualization-with-bokeh-in-python-part-one-getting-started-a11655a467d4](https://medium.com/data-science/data-visualization-with-bokeh-in-python-part-one-getting-started-a11655a467d4)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAAsUlEQVR4Xu3BAQEAAACCIP+vbkhAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8GXHmAAEdo+NeAAAAAElFTkSuQmCC>