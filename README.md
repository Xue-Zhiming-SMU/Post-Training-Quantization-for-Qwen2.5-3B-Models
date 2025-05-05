
# Model Quantization Evaluation

This repository contains a notebook that evaluates the performance of a Qwen2.5 using various quantization techniques. The evaluation is performed on the MMLU, ARC-easy, and ARC-Challenge benchmarks.

Metrics collected include accuracy, perplexity, memory footprint (model size), and inference latency.

## Evaluation Results

The following table summarizes the key performance metrics across the different quantization methods and benchmarks:

| Quantization        | Compute Type | MMLU Acc (%) | MMLU Mem (GB) | MMLU Lat (ms/tok) | ARC-e Acc (%) | ARC-e Mem (GB) | ARC-e Lat (ms/tok) | ARC-c Acc (%) | ARC-c Mem (GB) | ARC-c Lat (ms/tok) |
| :------------------ | :----------- | :----------- | :------------ | :---------------- | :------------ | :------------- | :----------------- | :------------ | :------------- | :----------------- |
| **FP32**            | FP32         | 61.0         | 11.50         | 239               | 90.5          | 11.50          | 175                | 90.5          | 11.50          | 176                |
| **FP16**            | FP16         | 61.0         | 5.75          | 109               | 90.5          | 5.75           | 106                | 90.5          | 5.75           | 105                |
| **NF4**             | FP32         | 58.5         | 1.87          | 412               | 90.0          | 1.87           | 344                | 90.0          | 1.87           | 343                |
| **NF4**             | FP16         | 58.5         | 1.87          | 197               | 90.0          | 1.87           | 194                | 90.0          | 1.87           | 194                |
| **FP4**             | FP32         | 50.5         | 1.87          | 405               | 84.0          | 1.87           | 334                | 84.0          | 1.87           | 334                |
| **FP4**             | FP16         | 51.0         | 1.87          | 197               | 84.0          | 1.87           | 194                | 84.0          | 1.87           | 194                |
| **INT8**            | FP32         | 60.0         | 3.16          | 409               | 89.5          | 3.16           | 398                | 89.5          | 3.16           | 399                |
| **INT8**            | FP16         | 60.0         | 3.16          | 409               | 89.5          | 3.16           | 398                | 89.5          | 3.16           | 402                |

*(Note: ARC-e refers to ARC-easy, ARC-c refers to ARC-Challenge. Latency is measured in milliseconds per token. Mem is the model's memory footprint.)*

## Observations & Insights

*   **FP16 Efficiency:** FP16 quantization emerges as a highly effective baseline, providing the same accuracy as FP32 across all benchmarks but with a ~50% reduction in memory footprint and significantly lower inference latency. This makes it a strong candidate when maximum accuracy is desired with improved efficiency.
*   **4-bit Quantization (NF4/FP4):** These methods achieve the most significant memory reduction (~84% vs FP32), making them suitable for highly memory-constrained environments.
    *   **NF4** shows only a minor accuracy drop compared to FP16/FP32, making it preferable to FP4.
    *   **FP4** suffers a more pronounced accuracy degradation, especially on ARC benchmarks.
*   **Compute Type Impact:** For 4-bit formats (NF4, FP4), using **FP16 compute** significantly reduces latency compared to using FP32 compute. This highlights the performance benefits of using compute kernels optimized for lower-precision data types when available.
*   **Dequantization Overhead:** Quantization methods like those provided by `bitsandbytes` (NF4, FP4, INT8) often involve storing weights in the quantized format but performing computations in a higher precision (like FP16 or FP32). This requires **dequantization** during inference, which adds computational overhead. This overhead can explain why the latency reduction isn't always proportional to the memory reduction, especially when comparing FP32 compute vs FP16 compute for these formats. The high latency observed with FP32 compute for NF4/FP4 is likely influenced by this dequantization cost.
*   **INT8 Performance:** INT8 quantization maintains accuracy remarkably well (very close to FP16/FP32 levels) while offering a substantial memory saving (~73% vs FP32). However, the inference latency observed in these tests was consistently **high**, often exceeding even FP32 latency. This is counter-intuitive, as INT8 inference is typically expected to be faster. Potential reasons include:
    *   The specific INT8 implementation used might not be fully optimized for the hardware.
    *   Significant dequantization overhead (to FP32 or FP16 for computation) cancelling out the potential speed benefits.
    *   The specific model architecture might interact poorly with this particular INT8 scheme. Further investigation would be needed to pinpoint the cause of this high latency.
*   **Trade-offs:** The results clearly illustrate the fundamental trade-off in model quantization: reducing memory footprint and (potentially) latency often comes at the cost of some reduction in model accuracy and potentially increased perplexity. The choice of method depends on the specific constraints and performance requirements of the application.
*   **Benchmark Consistency:** The relative performance ranking of the quantization methods remained consistent across the different benchmarks (MMLU, ARC-e, ARC-c), suggesting the observed trade-offs are generally applicable for this model.

## How to Reproduce

To reproduce these results, run the `PTQ Models' Evaluation via BnB.ipynb` notebook. Ensure you have the necessary dependencies installed.

## Dependencies (Example)

Key Python libraries used likely include:
*   `torch`
*   `transformers`
*   `accelerate`
*   `bitsandbytes`
*   `datasets`
*   `evaluate`

*(Please adjust the model name and dependency list as needed based on your actual notebook.)*
