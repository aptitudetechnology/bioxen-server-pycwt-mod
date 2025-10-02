# FPGA Acceleration for Biological Wavelet Coherence Analysis

**Research Question:** Can FPGA-based hardware acceleration provide superior performance, energy efficiency, and deterministic latency for real-time wavelet coherence analysis in biological signal processing compared to GPU and CPU implementations?

---

## 🎯 Strategic Context

While GPU acceleration (Phase 2) addresses extreme-scale batch processing, FPGAs offer distinct advantages for biological signal analysis:

- **Real-time processing:** Deterministic latency for closed-loop experiments (brain-computer interfaces, adaptive stimulation)
- **Energy efficiency:** 10-100× lower power consumption than GPUs for deployment in portable medical devices
- **Custom pipeline optimization:** Hardware tailored specifically for wavelet coherence computation
- **Continuous streaming:** Process infinite-length signals without batching constraints
- **Edge deployment:** Embedded systems for wearable biosensors, in-vivo monitoring

**Critical Question:** Do these advantages justify FPGA development complexity for BioXen's biological analysis targets?

---

## 📊 Core Research Questions

### **Question 1: FPGA Performance Characterization vs. GPU/CPU**

**Investigation Focus:**
Establish empirical performance baselines comparing FPGA, GPU, and CPU implementations across biologically relevant scenarios.

**Metrics to Compare:**

| Metric | CPU Baseline | GPU | FPGA Target |
|--------|--------------|-----|-------------|
| Throughput (samples/sec) | 1× | 10-50× | 5-100×? |
| Latency (milliseconds) | Variable | 5-20 ms | <1 ms? |
| Power consumption (watts) | 65-150W | 150-400W | 10-50W? |
| Energy per computation (joules) | 1× | 0.2-0.5× | 0.01-0.1×? |
| Determinism (latency jitter) | High | Medium | Minimal? |

**Biological Use Cases to Test:**

1. **Real-time neural recording analysis:**
   - 128-channel EEG at 1 kHz (128,000 samples/sec)
   - Compute wavelet coherence between all channel pairs in real-time
   - Requirement: <10 ms latency for closed-loop brain stimulation

2. **Continuous physiological monitoring:**
   - ECG, blood pressure, respiration (5 signals at 250 Hz)
   - Sliding-window wavelet coherence for autonomic coupling
   - Requirement: 24/7 operation on battery power

3. **High-throughput calcium imaging:**
   - 1,000 neurons at 30 fps (30,000 samples/sec per neuron)
   - Pairwise wavelet coherence for functional connectivity
   - Requirement: Process during experiment for adaptive stimulus selection

**What to Determine:**
- Does FPGA achieve lower latency than GPU for small batch sizes?
- What is the crossover point where GPU batch processing beats FPGA throughput?
- Can FPGA maintain deterministic performance under varying signal characteristics?
- Is FPGA energy efficiency advantage sufficient for portable device deployment?

---

### **Question 2: FPGA Architecture Design Trade-offs**

**Investigation Focus:**
Design optimal FPGA pipeline architecture for continuous wavelet transform and wavelet coherence computation.

**Key Architectural Decisions:**

**A. Precision vs. Performance:**
- **Fixed-point vs. Floating-point:**
  - Fixed-point: Higher throughput, lower resource usage, potential precision loss
  - Floating-point: Matches CPU/GPU results, higher resource usage
  - **Question:** Is 16-bit or 32-bit fixed-point sufficient for biological signal analysis?
  
- **Numerical Validation:**
  - Compare FPGA fixed-point output with CPU double-precision reference
  - Measure SNR degradation, correlation coefficients, biological interpretation accuracy
  - Determine minimum precision for statistically valid results

**B. Pipeline Architecture:**

```
Option 1: Fully Pipelined (Maximum Throughput)
[Input] → [CWT Core] → [Complex Multiply] → [Smoothing] → [WCT] → [Output]
         |            |                  |            |
         [Parallel]   [Parallel]         [Sliding]    [Normalize]
         [Scales]     [Time Points]      [Window]     
```

```
Option 2: Time-Multiplexed (Resource Efficient)
[Input Buffer] → [Shared CWT Engine] → [Coefficient Store] → [WCT Engine] → [Output]
                 (Process scales sequentially)
```

```
Option 3: Hybrid (Balanced)
[Input] → [Pipelined CWT × 4 Parallel Units] → [Shared WCT Engine] → [Output]
```

**Trade-off Analysis:**
- **Fully Pipelined:** Max throughput, high LUT/DSP usage, limited by FPGA size
- **Time-Multiplexed:** Fits smaller FPGAs, lower throughput, complex control logic
- **Hybrid:** Balanced performance/resources, most flexible

**Resource Constraints:**
- Modern FPGAs: 100k-500k LUTs, 1k-5k DSP blocks
- CWT requires ~50-100 scales × complex multiply-accumulate
- How many parallel wavelet cores can we fit on target FPGA?

**C. Memory Architecture:**

**Challenge:** Wavelet coherence requires buffering signals across multiple scales with different time-smoothing windows.

**Buffering Strategies:**
- **On-chip BRAM:** Fast but limited (10-50 MB typical)
- **External DRAM:** Large capacity (GB) but higher latency
- **Streaming with minimal buffering:** Only store necessary samples for smoothing

**Question:** Can we achieve real-time processing with pure streaming, or do we need external memory?

---

### **Question 3: Real-Time Streaming vs. Batch Processing**

**Investigation Focus:**
Evaluate FPGA's unique advantage for continuous signal processing without batching.

**Streaming Architecture Benefits:**

**Continuous Wavelet Transform Pipeline:**
```
Time t → t+1 → t+2 → t+3 → ...
   ↓      ↓      ↓      ↓
[Morlet wavelet convolution at scale 1]
[Morlet wavelet convolution at scale 2]
...
[Morlet wavelet convolution at scale N]
   ↓      ↓      ↓      ↓
[WCT computation] → Output coherence at time t
```

**Advantages over Batch Processing:**
- **Zero startup latency:** No need to accumulate batch before processing
- **Infinite signals:** Process hours-long recordings without segmentation
- **Memory efficiency:** Only store sliding window, not entire signal history
- **Closed-loop compatibility:** Sub-millisecond response for adaptive experiments

**Biological Applications Requiring Streaming:**

1. **Adaptive Deep Brain Stimulation (DBS):**
   - Monitor LFP coherence between brain regions
   - Detect pathological synchronization patterns
   - Trigger therapeutic stimulation within 1-10 ms
   - **FPGA Advantage:** Deterministic low-latency response

2. **Seizure Prediction Systems:**
   - Continuous EEG wavelet coherence monitoring
   - Detect pre-ictal coherence changes
   - Alert patient or trigger intervention
   - **FPGA Advantage:** 24/7 low-power operation

3. **Brain-Computer Interface (BCI):**
   - Real-time motor imagery classification using wavelet features
   - Decode intended movement from neural coherence patterns
   - Update prosthetic control signals at 100+ Hz
   - **FPGA Advantage:** Consistent latency for smooth control

**Validation Questions:**
- Can FPGA maintain <1 ms latency for 128-channel EEG coherence?
- How many simultaneous signal pairs can be processed in real-time?
- Does streaming FPGA architecture outperform GPU batch processing for latency-critical tasks?

---

### **Question 4: FPGA Development Complexity vs. Benefit**

**Investigation Focus:**
Assess whether FPGA performance advantages justify significant development and deployment challenges.

**Development Complexity Factors:**

**A. Hardware Design Skills:**
- FPGA requires Verilog/VHDL or HLS (High-Level Synthesis)
- Signal processing expertise in fixed-point arithmetic
- Timing closure and resource optimization
- **Barrier:** Steep learning curve for software-focused teams

**B. Validation and Testing:**
- Must validate against CPU/GPU reference implementations
- Fixed-point numerical accuracy testing
- Hardware-in-the-loop testing for real-time performance
- **Effort:** 2-4× longer than software testing

**C. Deployment Infrastructure:**
- Requires FPGA board (Xilinx, Intel) - $500-$5,000+ per device
- Host interface (PCIe, Ethernet) driver development
- FPGA configuration/programming infrastructure
- **Barrier:** Users need FPGA hardware expertise

**D. Maintenance and Updates:**
- FPGA synthesis takes hours vs. seconds for software compilation
- Harder to debug than CPU/GPU code (limited visibility into hardware state)
- FPGA-specific bugs (timing violations, metastability)
- **Cost:** Slower iteration cycles

**Cost-Benefit Analysis:**

| Development Phase | CPU/GPU | FPGA | Multiplier |
|------------------|---------|------|------------|
| Initial implementation | 2 weeks | 8-12 weeks | 4-6× |
| Validation/testing | 1 week | 4-6 weeks | 4-6× |
| Optimization | 1 week | 3-4 weeks | 3-4× |
| Documentation | 1 week | 2-3 weeks | 2-3× |
| **Total** | **5 weeks** | **17-25 weeks** | **3.4-5×** |

**When FPGA is Worth It:**
- Real-time latency is critical (BCI, adaptive stimulation)
- Deployment in power-constrained environments (wearables, implants)
- High-volume production (amortize NRE cost across many devices)
- Regulatory requirements for deterministic medical devices

**When GPU is Better:**
- Batch processing of recorded data (research analysis)
- Flexibility for algorithm updates
- Availability of GPU infrastructure
- Prototype/research stage (not production deployment)

---

### **Question 5: High-Level Synthesis (HLS) vs. RTL Implementation**

**Investigation Focus:**
Can High-Level Synthesis tools (Vitis HLS, Intel HLS) reduce FPGA development complexity while maintaining performance?

**HLS Approach:**
```cpp
// C++ code for wavelet transform (simplified)
void cwt_fpga(hls::stream<float> &input, 
              hls::stream<float> &output,
              float wavelet_params[N_SCALES]) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=wavelet_params complete
    
    for (int scale = 0; scale < N_SCALES; scale++) {
        #pragma HLS UNROLL
        float result = convolve(input, wavelet_params[scale]);
        output.write(result);
    }
}
```

**HLS Advantages:**
- Write in C/C++ instead of Verilog/VHDL
- Faster development (weeks vs. months)
- Easier to maintain and update
- Access to software libraries and tools

**HLS Disadvantages:**
- 10-30% performance penalty vs. hand-optimized RTL
- Less control over hardware architecture
- HLS tools can be unpredictable (quality of results varies)
- May not achieve theoretical maximum throughput

**Validation Questions:**
- Can HLS achieve 80%+ of hand-coded RTL performance?
- Is HLS-generated FPGA still faster than GPU?
- Does HLS reduce development time sufficiently to justify FPGA path?
- Can HLS code be shared between FPGA and CPU/GPU implementations?

**Recommended Approach:**
1. **Phase 1:** Prototype using HLS for rapid feasibility testing
2. **Phase 2:** Profile HLS implementation to identify bottlenecks
3. **Phase 3:** Selectively optimize critical paths in RTL if needed
4. **Validation:** Compare HLS vs. RTL performance on target FPGA

---

## 🔍 Investigation Methodology

### **FPGA Prototyping Approach**

**Step 1: Algorithm Simplification**
- Start with Morlet wavelet (most common in biology)
- Fixed set of scales (e.g., 50 scales covering 0.1-100 Hz)
- Single signal pair coherence (extend to multi-pair later)

**Step 2: Fixed-Point Precision Analysis**
- Simulate fixed-point arithmetic in MATLAB/Python
- Test 8-bit, 16-bit, 32-bit representations
- Measure numerical error vs. double-precision reference
- Determine minimum precision for <1% error

**Step 3: HLS Implementation**
- Write C++ reference implementation
- Add HLS pragmas for pipeline and parallelization
- Synthesize and measure resource usage (LUTs, DSPs, BRAM)
- Validate output against CPU reference

**Step 4: FPGA Hardware Testing**
- Deploy to development board (e.g., Xilinx Zynq, Intel Stratix)
- Measure actual throughput, latency, power consumption
- Test with synthetic and real biological signals
- Compare with CPU/GPU baseline benchmarks

---

### **Real-Time Performance Validation**

**Latency Measurement Setup:**
```
[Signal Generator] → [FPGA] → [Output Capture]
        ↓                           ↑
    [Trigger]───────────────────────┘
          (Measure time from input to output)
```

**Test Scenarios:**
1. **Single-sample latency:** Time from new sample arrival to coherence output
2. **Throughput:** Maximum sustained sample rate before buffer overflow
3. **Jitter:** Variance in processing latency over 10,000 samples
4. **Multi-channel:** Latency vs. number of simultaneous signal pairs

**Biological Signal Testing:**
- Use actual EEG/ECG recordings from PhysioNet
- Validate coherence patterns match CPU reference
- Ensure no artifacts from fixed-point quantization
- Test edge cases (signal saturation, noise bursts)

---

### **Power Consumption Measurement**

**Methodology:**
- Measure FPGA power draw during idle, processing, peak load
- Compare with GPU and CPU power for same workload
- Calculate energy-per-coherence-computation
- Project battery life for portable device deployment

**Target Platforms:**
- **Low-power:** Xilinx Zynq UltraScale+ (2-10W) for wearables
- **High-performance:** Xilinx Alveo U280 (75W) for server deployment
- **Comparison:** NVIDIA RTX 4090 (450W), Intel i9 CPU (125W)

---

## 🎯 Expected Outcomes

### **Deliverable 1: FPGA Feasibility Report**

**Content:**
1. **Performance Comparison Table:**
   - Latency: FPGA vs. GPU vs. CPU
   - Throughput: samples/sec for each platform
   - Power consumption and energy efficiency
   - Resource requirements (FPGA size, GPU VRAM, CPU cores)

2. **Numerical Validation:**
   - Fixed-point precision analysis
   - Correlation with CPU double-precision reference
   - Biological signal test results (EEG, ECG)

3. **Cost-Benefit Analysis:**
   - Development time: HLS vs. RTL vs. GPU CUDA
   - Hardware costs: FPGA boards vs. GPU vs. CPU
   - Deployment complexity and user requirements

4. **Use Case Recommendations:**
   - When FPGA is optimal (real-time, low-power, deterministic)
   - When GPU is better (batch processing, research, flexibility)
   - When CPU is sufficient (small datasets, prototyping)

---

### **Deliverable 2: FPGA Architecture Specification**

**If FPGA path is pursued, provide:**

1. **Hardware Architecture Diagram:**
   - Pipeline stages for CWT and WCT
   - Memory hierarchy and buffering strategy
   - Parallel processing units and resource allocation

2. **Resource Budget:**
   - LUTs, DSPs, BRAM for target FPGA
   - Scaling analysis: can we fit 4×, 8×, 16× parallel units?
   - Minimum FPGA model required (e.g., Zynq-7000 vs. UltraScale+)

3. **Interface Specification:**
   - Host-to-FPGA data transfer (PCIe, Ethernet, AXI)
   - Input/output data formats
   - Control registers for runtime configuration

4. **Performance Specifications:**
   - Guaranteed maximum latency (99th percentile)
   - Sustained throughput with zero data loss
   - Power consumption under typical workloads

---

### **Deliverable 3: HLS Reference Implementation**

**Open-source HLS code for community validation:**

```cpp
// Xilinx Vitis HLS example
#include "hls_stream.h"
#include "hls_math.h"

// Configuration
#define N_SCALES 50
#define WINDOW_SIZE 256
#define PRECISION 16  // bits

typedef ap_fixed<PRECISION, 8> fixed_t;

void bioxen_cwt(
    hls::stream<fixed_t> &signal_x,
    hls::stream<fixed_t> &signal_y,
    hls::stream<fixed_t> &coherence_out,
    fixed_t scale_params[N_SCALES]
) {
    #pragma HLS INTERFACE mode=axis port=signal_x
    #pragma HLS INTERFACE mode=axis port=signal_y
    #pragma HLS INTERFACE mode=axis port=coherence_out
    #pragma HLS PIPELINE II=1
    
    // CWT computation for both signals
    // Wavelet coherence calculation
    // Output coherence coefficients
    
    // (Full implementation ~200-500 lines)
}
```

**Include:**
- Full HLS source code with comments
- Testbench for validation
- Python script to generate test vectors
- Synthesis reports (resource usage, timing)

---

## 🔗 Integration with Existing Research

### **Relationship to MVP and Phase 2:**

**MVP Focus:** Validate pycwt performance on CPU
- **FPGA Relevance:** Establish CPU baseline for comparison

**Phase 2 Focus:** GPU acceleration for massive batch processing
- **FPGA Relevance:** FPGA targets different use cases (real-time, low-latency)

**FPGA Focus:** Real-time, low-power, deterministic processing
- **Complementary:** FPGA doesn't replace GPU; offers different deployment scenarios

---

### **Decision Tree for Backend Selection:**

```
User Requirements
    ├── Real-time latency critical (<10 ms)?
    │   ├── Yes → Evaluate FPGA
    │   │   ├── Budget for FPGA development? → FPGA
    │   │   └── No budget → GPU with optimized batch size
    │   └── No → Continue
    ├── Portable/wearable deployment?
    │   ├── Yes + Low power critical → FPGA (UltraScale+)
    │   └── No → Continue
    ├── Massive batch processing (>1M samples)?
    │   ├── Yes → GPU (datacenter)
    │   └── No → Continue
    └── Typical biological datasets (<100k samples)
        └── CPU with pycwt (MVP baseline)
```

---

## ⚠️ Research Constraints and Limitations

### **Known FPGA Challenges:**

**1. Fixed Algorithm:**
- FPGA pipeline optimized for specific wavelet type (Morlet)
- Changing to different wavelet requires re-synthesis
- Less flexible than GPU shader code

**2. Limited On-Chip Memory:**
- Can't store full signal history for very long recordings
- May need external DRAM for large-scale analysis
- Memory bandwidth can become bottleneck

**3. Scalability:**
- FPGA size limits number of parallel processing units
- Can't dynamically allocate resources like GPU threads
- Scaling requires larger/more expensive FPGA

**4. Ecosystem Maturity:**
- Fewer open-source wavelet libraries for FPGA
- Limited community compared to GPU/CUDA ecosystem
- Steeper learning curve for new developers

---

### **Out of Scope for Initial FPGA Research:**

**Deferred to Future Work:**
- Multi-FPGA clustering for extreme scalability
- Adaptive wavelet selection at runtime
- Integration with machine learning models
- Neuromorphic/spiking neural network integration
- Quantum-inspired algorithms on FPGA

**Focus on Proven Use Cases:**
- Standard Morlet wavelet continuous transform
- Fixed scale set optimized for biological frequencies
- Single-FPGA deployment scenarios
- Comparison with CPU/GPU baseline

---

## 🎓 Success Criteria

### **FPGA Research is Complete When:**

1. ✅ **Performance characterized:** Latency, throughput, power for FPGA vs. GPU vs. CPU
2. ✅ **Numerical validation:** FPGA fixed-point output matches CPU reference (<1% error)
3. ✅ **Real-time capability:** FPGA achieves <10 ms latency for 128-channel EEG coherence
4. ✅ **Use case clarity:** Clear guidance on when FPGA provides value vs. GPU/CPU
5. ✅ **Cost-benefit analysis:** Development effort justified by performance gains for specific applications

---

### **Decision Gate Output:**

**Primary Question:** Should BioXen invest in FPGA acceleration, and for which deployment scenarios?

**Possible Outcomes:**

**Outcome A: FPGA High-Priority**
- FPGA achieves 10× lower latency and 10× better energy efficiency than GPU
- Real-time biological applications (BCI, adaptive stimulation) are primary targets
- Recommendation: Develop FPGA backend alongside GPU path

**Outcome B: FPGA Niche-Only**
- FPGA offers advantages only for specialized real-time applications
- Most users prefer GPU batch processing flexibility
- Recommendation: Document FPGA path for advanced users; focus BioXen core on GPU

**Outcome C: FPGA Not Justified**
- GPU provides sufficient performance for real-time needs
- FPGA development complexity outweighs benefits
- Recommendation: Defer FPGA to future version if user demand emerges

---

## 📚 Background Research

### **Literature Review:**

**FPGA Wavelet Transform Implementations:**
- Survey of FPGA architectures for continuous wavelet transform
- Fixed-point arithmetic optimization for signal processing
- Real-time ECG analysis using FPGA-based wavelet decomposition
- Comparison of HLS vs. RTL for DSP applications

**Biological Real-Time Systems:**
- Closed-loop brain stimulation requirements (latency, jitter)
- Wearable biosensor design constraints (power, size, cost)
- Brain-computer interface signal processing pipelines
- Seizure detection algorithm computational requirements

**FPGA vs. GPU Studies:**
- Latency comparison for streaming vs. batch processing
- Energy efficiency analysis for edge computing
- Total cost of ownership for different acceleration platforms

---

### **Existing FPGA Biosignal Projects:**

**Prior Art to Study:**
- Open-source FPGA EEG processing (OpenBCI, OpenEEG)
- Medical device FPGA implementations (FDA-approved systems)
- Academic research FPGA wavelet processors
- Commercial FPGA biosignal analyzers (NI hardware, etc.)

**Questions to Answer:**
- What architectures have proven successful?
- What pitfalls should we avoid?
- Can we reuse existing IP cores?
- Are there licensing considerations?

---

## 🚀 Recommended Research Phases

### **Phase 1: Feasibility Study (2-3 weeks)**

**Minimal Investment:**
- Literature review of FPGA wavelet implementations
- Fixed-point precision simulation in MATLAB
- HLS prototype for single-scale CWT
- Resource estimation for target FPGA

**Go/No-Go Decision:**
- If HLS prototype achieves <50% resource usage on low-cost FPGA → Proceed
- If fixed-point precision analysis shows <1% error → Proceed
- Otherwise → Defer FPGA to future research

---

### **Phase 2: HLS Implementation (4-6 weeks)**

**If Phase 1 is Promising:**
- Complete HLS implementation of multi-scale CWT + WCT
- Synthesize for target FPGA platform
- Validate against CPU reference with biological signals
- Measure resource usage and timing closure

**Deliverable:**
- Working HLS code
- Synthesis reports
- Performance comparison with CPU/GPU

---

### **Phase 3: Hardware Validation (2-3 weeks)**

**If Phase 2 Meets Performance Targets:**
- Deploy to FPGA development board
- Real-time testing with signal generator
- Power consumption measurement
- Latency and jitter characterization

**Deliverable:**
- Hardware test results
- Cost-benefit analysis
- Deployment guide

---

### **Phase 4: Production Integration (6-8 weeks)**

**If Phase 3 Justifies Production:**
- Develop BioXen Python API for FPGA backend
- PCIe driver and host interface
- User documentation and examples
- Integration testing with full BioXen stack

**Deliverable:**
- `BioXen.wavelet_coherence(..., backend='fpga')`
- Complete deployment pipeline
- User-facing documentation

---

## 🎯 Ultimate Research Goal

### **Answer the Strategic Question:**

**"Does FPGA acceleration unlock fundamentally new biological applications that GPU and CPU cannot address?"**

If the answer is **YES** (real-time closed-loop systems, portable medical devices), then FPGA investment is justified.

If the answer is **NO** (GPU handles all practical use cases), then GPU optimization should be the focus.

---

**END OF FPGA RESEARCH PROMPT**

---

## 📋 Quick Reference: FPGA vs. GPU vs. CPU

| Criterion | CPU | GPU | FPGA |
|-----------|-----|-----|------|
| **Development Time** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Moderate | ⭐⭐ Slow |
| **Batch Throughput** | ⭐ Low | ⭐⭐⭐⭐⭐ Highest | ⭐⭐⭐ Moderate |
| **Real-Time Latency** | ⭐⭐ Variable | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Best |
| **Energy Efficiency** | ⭐⭐⭐ Moderate | ⭐⭐ High Power | ⭐⭐⭐⭐⭐ Best |
| **Flexibility** | ⭐⭐⭐⭐⭐ Highest | ⭐⭐⭐⭐ High | ⭐⭐ Limited |
| **Cost (Hardware)** | ⭐⭐⭐⭐⭐ Lowest | ⭐⭐⭐ Moderate | ⭐⭐ Expensive |
| **Determinism** | ⭐⭐ Poor | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Excellent |
| **Portability** | ⭐⭐⭐⭐ Good | ⭐⭐ Power-hungry | ⭐⭐⭐⭐⭐ Best |

---

## 🎯 Recommended Reading Order

1. **Start Here:** Read MVP prompt to understand CPU baseline
2. **Then:** Read Phase 2 GPU prompt for batch processing context
3. **Finally:** Read this FPGA prompt to understand real-time / edge deployment

**Decision Priority:**
1. **MVP first** - Validate if pycwt CPU is sufficient (lowest risk)
2. **GPU next** - If batch processing bottleneck identified (medium risk)
3. **FPGA last** - Only if real-time/embedded requirements emerge (highest risk/reward)
