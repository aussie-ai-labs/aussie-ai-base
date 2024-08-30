# Aussie AI Base C++ Library

You're still here? You haven't been replaced by code-regurgitating
robots yet? Well, then...Welcome to the Aussie AI codespace!

This repository is the Aussie AI Base C++ library, which implements a number of low-level Transformer and LLM
components and related algorithms from ML and linear algebra. It's also a sneaky way to
release all of the source code examples
for the book "Generative AI in C++" published in March 2024.

## About Aussie AI

Aussie AI is based in Australia (surprise!) and focused on Generative AI (surprise again).
We are focused on consumer AI applications (the top layer) and kernel optimization research
for Transformers (the bottom layer). Some of our work:

- [AI Research Literature Survey](https://www.aussieai.com/research/overview)
- [Aussie AI Research Blog](https://www.aussieai.com/blog/index)
- [Inference Optimization Techniques List](https://www.aussieai.com/research/list)
- [A-Z on Inference Optimization Research](https://www.aussieai.com/research/index-techniques)

## Transformer Components in C++

This C++ library presents implementations of various Transformer components in C++.
The purpose of this library is mainly educational
and complements the descriptions in the book "Generative AI in C++".
Sorry, these are not tight CUDA kernels for your GPUs.

Some of the implemented Transformer components include:

- Activation Functions (RELU, GELU)
- Normalization (BatchNorm)
- AVX Vectorization (for x86 CPUs such as Microsoft Windows)
- Softmax normalization
- Top-k decoding

Some general linear algebra methods include:

- Vector operations
- Vector norms (L1/L2/L3)
- MatMul/GEMM (basic/tiled)

## Advanced General C++ Coding

Some portions of this library are not AI-specific, but 
are advanced general-purpose programming methods, including:

- Bitwise tricks
- Floating-point tricks
- Dynamic arrays/tensors (1D/2D/3D)
- Arithmetic Operators (basic)

## Optimizing and Profiling C++ Libraries

Some of the performance improvement general techniques include:

- Benchmarking/timing of code execution
- Precomputation optimizations
- Loop optimizations (loop unrolling, loop fusion, loop fission)

## Debugging C++ Libraries

Some of the debugging-related areas include:

- Assertion macros
- Debug wrapper functions
- Debug trace macros
- Portability checking

## Generative AI in C++ Book

This repository includes the C++ source code examples from 
the book "Generative AI in C++: Coding Transformers and LLMs" by David Spuler, March 2024.

Here are some more details about the book itself.
The full text of all chapters of the book is available online in HTML format (no login required):

- [Generative AI Book](https://www.aussieai.com/book/overview)
- [Full Text Chapters Online](https://www.aussieai.com/book/free)
- [Detailed Table of Contents](https://www.aussieai.com/book/toc)
- [Bonus Materials](https://www.aussieai.com/book/bonus)

If you have an expense account or a professional development allowance, here's where you go:

- [Amazon E-Book and Print](https://www.amazon.com/dp/B0CXJKCWX9)

## Table of Contents for Generative AI in C++

Full text is available online for all chapters, which includes descriptions of many of these coded functions.

**Front Matter**

- [Foreword](https://www.aussieai.com/book/foreword)
- [Preface](https://www.aussieai.com/book/preface)
- [Source Code Availability](https://www.aussieai.com/book/source-code)
- [About the Author](https://www.aussieai.com/book/about)
- [About the Contributors](https://www.aussieai.com/book/about)

**Part I: AI Projects in C++**

- [Chapter 1. Introduction to AI in C++](https://www.aussieai.com/book/chapter-1-intro)
- [Chapter 2. Transformers & LLMs](https://www.aussieai.com/book/chapter-2-transformers-llms.htm)
- [Chapter 3. AI Phones](https://www.aussieai.com/book/ch3-phone-ai.htm)
- [Chapter 4. AI on Your Desktop](https://www.aussieai.com/book/ch4-desktop-ai.htm)
- [Chapter 5. Design Choices & Architectures](https://www.aussieai.com/book/ch5-design-architectures.htm)
- [Chapter 6. Training, Fine-Tuning & RAG](https://www.aussieai.com/book/ch6-training-ft-rag.htm)
- [Chapter 7. Deployment Architecture](https://www.aussieai.com/book/ch7-deployment-architecture.htm)

**Part II: Basic C++ Optimizations**

- [Chapter 8. Bitwise Operations](https://www.aussieai.com/book/ch8-bitwise.htm)
- [Chapter 9. Floating Point Arithmetic](https://www.aussieai.com/book/ch9-floating-point.htm)
- [Chapter 10. Arithmetic Optimizations](https://www.aussieai.com/book/ch10-arithmetic-optimizations.htm)
- [Chapter 11. Compile-Time Optimizations](https://www.aussieai.com/book/ch11-compile-time.htm)
- [Chapter 12. Pointer Arithmetic](https://www.aussieai.com/book/ch12-pointer-arithmetic.htm)
- [Chapter 13. Algorithm Speedups](https://www.aussieai.com/book/ch13-algorithms.htm)
- [Chapter 14. Memory Optimizations](https://www.aussieai.com/book/ch14-memory-optimize.htm)

**Part III: Parallel C++ Optimizations**

- [Chapter 15. Loop Vectorization](https://www.aussieai.com/book/ch15-loop-vectorization.htm)
- [Chapter 16. Hardware Acceleration](https://www.aussieai.com/book/ch16-hardware-assembler.htm)
- [Chapter 17. AVX Intrinsics](https://www.aussieai.com/book/ch17-avx-intrinsics.htm)
- [Chapter 18. Parallel Data Structures](https://www.aussieai.com/book/ch18-parallel-data-structures.htm)

**Part IV: Transformer Components in C++**

- [Chapter 19. Encoders & Decoders](https://www.aussieai.com/book/ch19-intro-encoder-decoder.htm)
- [Chapter 20. Attention](https://www.aussieai.com/book/ch20-attention.htm)
- [Chapter 21. Activation Functions](https://www.aussieai.com/book/ch21-activation-functions.htm)
- [Chapter 22. Vector Algorithms](https://www.aussieai.com/book/ch22-vector-algorithms.htm)
- [Chapter 23. Tensors](https://www.aussieai.com/book/ch23-tensors.htm)
- [Chapter 24. Normalization](https://www.aussieai.com/book/ch24-normalization.htm)
- [Chapter 25. Softmax](https://www.aussieai.com/book/ch25-softmax.htm)
- [Chapter 26. Decoding Algorithms](https://www.aussieai.com/book/ch26-decoding.htm)
- [Chapter 27. Tokenizer and Vocabulary](https://www.aussieai.com/book/ch27-tokenizer-embedding.htm)

**Part V: Optimizing Transformers in C++**

- [Chapter 28. Deslugging AI Engines](https://www.aussieai.com/book/ch28-deslugging-ai.htm)
- [Chapter 29. Caching Optimizations](https://www.aussieai.com/book/ch29-caching.htm)
- [Chapter 30. Vectorization](https://www.aussieai.com/book/ch30-vectorization.htm)
- [Chapter 31. Kernel Fusion](https://www.aussieai.com/book/ch31-kernel-fusion.htm)
- [Chapter 32. Quantization](https://www.aussieai.com/book/ch32-quantization.htm)
- [Chapter 33. Pruning](https://www.aussieai.com/book/ch33-pruning.htm)
- [Chapter 34. MatMul/GEMM](https://www.aussieai.com/book/ch34-matmul-gemm.htm)
- [Chapter 35. Lookup Tables & Precomputation](https://www.aussieai.com/book/ch35-lut-precomputation.htm)
- [Chapter 36. AI Memory Optimizations](https://www.aussieai.com/book/ch36-memory-ai.htm)

**Part VI: Enterprise AI in C++**

- [Chapter 37. Tuning, Profiling & Benchmarking](https://www.aussieai.com/book/ch37-benchmark-profiling.htm)
- [Chapter 38. Platform Portability](https://www.aussieai.com/book/ch38-platform-portability.htm)
- [Chapter 39. Quality](https://www.aussieai.com/book/ch39-quality.htm)
- [Chapter 40. Reliability](https://www.aussieai.com/book/ch40-reliability.htm)
- [Chapter 41. Self-Testing Code](https://www.aussieai.com/book/ch41-self-testing-code.htm)
- [Chapter 42. Debugging](https://www.aussieai.com/book/ch42-debugging.htm)

**Part VII: Research on AI Optimization**

- [Chapter 43. Overview of AI Research](https://www.aussieai.com/book/ch43-research-overview.htm)
- [Chapter 44. Advanced Quantization](https://www.aussieai.com/book/ch44-advanced-quantization.htm)
- [Chapter 45. Knowledge Distillation](https://www.aussieai.com/book/ch45-knowledge-distillation-research.htm)
- [Chapter 46. Structured Pruning](https://www.aussieai.com/book/ch46-structured-pruning.htm)
- [Chapter 47. Early Exit and Layer Pruning](https://www.aussieai.com/book/ch47-layer-prune-early-exit.htm)
- [Chapter 48. Width Pruning](https://www.aussieai.com/book/ch48-width-pruning.htm)
- [Chapter 49. Length Pruning](https://www.aussieai.com/book/ch49-length-pruning.htm)
- [Chapter 50. Adaptive Inference](https://www.aussieai.com/book/ch50-adaptive-inference.htm)
- [Chapter 51. Zero-Multiplication Models](https://www.aussieai.com/book/ch51-zero-multiplication.htm)
- [Chapter 52. Logarithmic Models](https://www.aussieai.com/book/ch52-logarithmic-model.htm)
- [Chapter 53. Arithmetic Optimization Research](https://www.aussieai.com/book/ch53-arithmetic-optimization.htm)
- [Chapter 54. Ensemble Multi-Model Architectures](https://www.aussieai.com/book/ch54-ensemble-research.htm)
- [Chapter 55. Advanced Number Systems](https://www.aussieai.com/book/ch55-advanced-number-systems.htm)
- [Chapter 56. Neural Architecture Search](https://www.aussieai.com/book/ch56-nas-research.htm)

**Appendices**

- [Appendix 1: C++ Slug Catalog](https://www.aussieai.com/book/appendix-1-slug-catalog)
- [Bonus Appendix: C++ Bug Catalog](https://www.aussieai.com/book/bug-catalog)
- [Bonus Appendix: C++ Bug Symptom Diagnosis](https://www.aussieai.com/book/appendix-symptom-diagnosis-bugs)
- [Bonus Appendix: C++ Portability Bug Catalog](https://www.aussieai.com/book/appendix-portability-bug-catalog)

## Programming Language Support

The Aussie AI Base C++ Library supports these languages:

- C++ - All of the main code is written in C++ and that's about all that matters.
- C - This library does not compile as standard C, but some portions of it would quite easily.
- Java - To create a Java version of this library, create 100 subdirectories in hierarchies at least 6 layers deep,
put 10 ".java" files in each subdirectory, then copy-paste one line of C++ source into each Java file. Presto! You now have a Java version.
- Python - Feel free to create your own Python wrappers for this library, if you want it to run slower.

## Portability

This C++ code should compile on these platforms:

- Linux - build using the Makefile ("make" and then "make test")
- Microsoft Windows - build the project inside Microsoft Visual Studio (MSVS) IDE.
- MacOS - sorry, you're on your own for now.

Most of the code, except the AVX stuff and anything else I've forgotten about,
is standard portable C++ and should compile
pretty much anywhere after an hour or two of fighting with compiler warnings.

## Building on Linux

Make is the build method.
The command to build the unit test executable:

    make

If you get an error about g++ then you may need this command:

    scl enable devtoolset-8 -- bash

The command to run the unit tests live:

    make test

To run with Valgrind runtime error checking:

    make valgrind

To run with gprof profiling:

    make prof

## License

The Aussie AI Base C++ Library is licensed under an MIT License; refer to the "LICENSE" file for details.

Please note that this license only applies to the C++ code examples uploaded herein, and not otherwise to the written text of the book "Generative AI in C++", which remains copyrighted and restricted,
even though it is available freely online.

## Citations

* David Spuler, March 2024, Generative AI in C++: Coding Transformers and LLMs, https://www.amazon.com/dp/B0CXJKCWX9.
* Aussie AI C++ Base Library, Aussie AI Labs, 2024, https://github.com/aussie-ai-labs/.

## Final Thoughts

Always try to be positive, except for blood tests.

