# BioXen Wavelet Coherence Research

This directory contains research documents and analysis for optimizing wavelet coherence analysis for biological signals.

## Contents

### Performance Research Prompts

1. **FFT-based-performance-bottleneck-massive-datasets-prompt.md** (Main)
   - Comprehensive deep research document (911 lines)
   - 5 critical research questions
   - Performance characterization across dataset scales
   - Comparative analysis with PyWavelets and ssqueezepy
   - Hybrid architecture design
   - Production scalability thresholds
   - 5-7 week research timeline

2. **FFT-based-performance-bottleneck-massive-datasets-prompt.mvp.md** (MVP)
   - Focused MVP version (633 lines)
   - Streamlined to 5 essential research questions
   - Go/no-go decision framework
   - Defers GPU and advanced optimization to Phase 2

3. **FFT-based-performance-bottleneck-massive-datasets-prompt.FPGA.md** (FPGA)
   - FPGA hardware acceleration exploration (701 lines)
   - Real-time processing with deterministic latency
   - Energy efficiency for portable/wearable devices
   - Use cases: BCI, adaptive stimulation, seizure detection

### Analysis Documents

4. **pwt-report.md**
   - Comprehensive code analysis of pycwt library
   - FFT-based implementation details
   - Performance bottleneck identification
   - Monte Carlo significance testing analysis (2,400 FFT operations)
   - Hybrid architecture feasibility assessment
   - API compatibility analysis
   - Numerical validation requirements
   - MVP research recommendations

5. **Python Libraries for Multi-Signal Wavelet Coherence in Computational Biology.md**
   - Comparative analysis of Python wavelet libraries
   - pycwt, PyWavelets, ssqueezepy comparison
   - Statistical significance testing requirements
   - Validation protocol for biological datasets
   - Visualization pipeline (Plotly/Bokeh)
   - Production deployment recommendations

6. **development-branch-analysis.md**
   - Analysis of upstream pycwt development branch
   - Summary of refactoring changes
   - Assessment of missing performance features
   - Merge strategy recommendations

## Key Findings

### pycwt Performance Characteristics
- **FFT-based implementation**: O(N log N) complexity
- **8 FFT operations per WCT**: 2 CWT + 3 smoothing operations
- **Monte Carlo bottleneck**: 300 iterations = 2,400 FFT operations
- **Sequential execution**: No parallelization

### Optimization Opportunities
1. **Immediate**: Parallelize Monte Carlo (4-8× speedup)
2. **Medium-term**: Hybrid PyWavelets architecture (2-5× speedup)
3. **Long-term**: GPU acceleration (10-50× speedup)

### Target Biological Applications
- Circadian rhythms (~100-200 timepoints) ✅
- Single-cell imaging (~1k-10k timepoints) ⚠️
- Physiological signals (~100k-1M timepoints) 🔴
- Neural recordings (~10M+ timepoints) 🔴

## Research Status

- ✅ Code analysis complete
- ✅ Bottlenecks identified
- ✅ Hybrid architecture validated as feasible
- ⏳ Empirical benchmarking needed (MVP Phase 1)
- ⏳ PyWavelets integration (Phase 2)
- ⏳ GPU acceleration evaluation (Phase 3)

## References

All documents reference:
- Torrence & Compo (1998) - Wavelet analysis methodology
- Grinsted et al. (2004) - Cross-wavelet and coherence
- pycwt codebase (v0.3.0a22)
- PyWavelets documentation
- ssqueezepy documentation

## Next Steps

1. Run MVP benchmarking (Week 1-2)
2. Validate PyWavelets numerical equivalence
3. Implement proof-of-concept hybrid architecture
4. Decision: Optimize or accept baseline performance

---

**Last Updated**: October 2, 2025
**Project**: BioXen Four-Lens Signal Analysis Library
