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

- Foreword
- Preface
- Source Code Availability
- About the Author
- About the Contributors

**Part I: AI Projects in C++**

- Chapter 1. Introduction to AI in C++
- Chapter 2. Transformers & LLMs
- Chapter 3. AI Phones
- Chapter 4. AI on Your Desktop
- Chapter 5. Design Choices & Architectures
- Chapter 6. Training, Fine-Tuning & RAG
- Chapter 7. Deployment Architecture

**Part II: Basic C++ Optimizations**

- Chapter 8. Bitwise Operations
- Chapter 9. Floating Point Arithmetic
- Chapter 10. Arithmetic Optimizations
- Chapter 11. Compile-Time Optimizations
- Chapter 12. Pointer Arithmetic
- Chapter 13. Algorithm Speedups
- Chapter 14. Memory Optimizations

**Part III: Parallel C++ Optimizations**

- Chapter 15. Loop Vectorization
- Chapter 16. Hardware Acceleration
- Chapter 17. AVX Intrinsics
- Chapter 18. Parallel Data Structures

**Part IV: Transformer Components in C++**
- Chapter 19. Encoders & Decoders
- Chapter 20. Attention
- Chapter 21. Activation Functions
- Chapter 22. Vector Algorithms
- Chapter 23. Tensors
- Chapter 24. Normalization
- Chapter 25. Softmax
- Chapter 26. Decoding Algorithms
- Chapter 27. Tokenizer and Vocabulary

**Part V: Optimizing Transformers in C++**

- Chapter 28. Deslugging AI Engines
- Chapter 29. Caching Optimizations
- Chapter 30. Vectorization
- Chapter 31. Kernel Fusion
- Chapter 32. Quantization
- Chapter 33. Pruning
- Chapter 34. MatMul/GEMM
- Chapter 35. Lookup Tables & Precomputation
- Chapter 36. AI Memory Optimizations

**Part VI: Enterprise AI in C++**

- Chapter 37. Tuning, Profiling & Benchmarking
- Chapter 38. Platform Portability
- Chapter 39. Quality
- Chapter 40. Reliability
- Chapter 41. Self-Testing Code
- Chapter 42. Debugging

**Part VII: Research on AI Optimization**

- Chapter 43. Overview of AI Research
- Chapter 44. Advanced Quantization
- Chapter 45. Knowledge Distillation
- Chapter 46. Structured Pruning
- Chapter 47. Early Exit and Layer Pruning
- Chapter 48. Width Pruning
- Chapter 49. Length Pruning
- Chapter 50. Adaptive Inference
- Chapter 51. Zero-Multiplication Models
- Chapter 52. Logarithmic Models
- Chapter 53. Arithmetic Optimization Research
- Chapter 54. Ensemble Multi-Model Architectures
- Chapter 55. Advanced Number Systems
- Chapter 56. Neural Architecture Search

**Appendices**

- Appendix 1: C++ Slug Catalog
- Bonus Appendix: C++ Bug Catalog
- Bonus Appendix: C++ Bug Symptom Diagnosis
- Bonus Appendix: C++ Portability Bug Catalog

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

