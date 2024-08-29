// ybenchmark.h -- Benchmarking (performance timing) -- Aussie AI Base Library  
// Created Nov 17th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YBENCHMARK_INCLUDE_HEADER_H
#define AUSSIE_YBENCHMARK_INCLUDE_HEADER_H

//---------------------------------------------------


void aussie_benchmark_all();
void yap_test_operator_accuracy();    // Test approximate operations accuracy...
void yap_benchmark_operations();
void aussie_benchmark_normalization();   // Benchmark BatchNorm (LayerNorm?)
void aussie_benchmark_softmax();
void aussie_benchmark_vecdot();  // vector dot product benchmarks...
void aussie_benchmark_vector_scalar_operations();  // Vector-scalar benchmarks...
void aussie_benchmark_vector_exponentiation_operations();   // Vector expf
void aussie_benchmark_matrix_vector_multiply();
void aussie_benchmark_matrix_matrix_multiplication();

void run_vector_float_N(char* name, long int niter, long int nvecsize, 
	void (*voidvectorfnptr)(const float v[], int n),
	float (*floatvectorfnptr)(const float v[], const float v2[], int n));

void run_vector_int_N(char* name, long int niter, long int nvecsize,
	int (*intvectorfnptr)(int v[], int v2[], int n));

void run_vector_scalar_N(char* name, long int niter, long int nvecsize, void (*vectorscalarfnptr)(float v[], int n, float scalar));

void run_arith_float_1000(char* name, long int n, float (*fnptr)(float a, float b), int (*ifnptr)(int a, int b));

void test_accuracy_1000(char* name, long int n, float (*fnptr)(float a, float b), int (*ifnptr)(int a, int b));

//---------------------------------------------------


#endif //AUSSIE_YBENCHMARK_INCLUDE_HEADER_H

