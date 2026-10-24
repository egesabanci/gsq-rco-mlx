"""gsq_rco: Gumbel-Softmax quantization (GSQ) and resource-constrained bit allocation (RCO) for
Qwen2.5-0.5B on MLX. Command line: `gsq-rco --help`.

  gsq_rco.common  WikiText-2 data, the FP model helpers, chunked NLL/KL and perplexity
  gsq_rco.gsq     GSQ quantizers (RTN, GPTQ, App-A inits), Lion, the staged block pipeline, export
  gsq_rco.rco     RCO manifold, Alg.1 loop, the quantized database and the KL-objective allocation
"""
