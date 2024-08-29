// ybenchmark.h -- Benchmarking (performance timing) -- Aussie AI Base Library  
// Created Nov 17th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>

//---------------------------------------------------
//---------------------------------------------------

#include "aport.h"
#include "aussieai.h"
#include "aassert.h"
#include "aops.h"
#include "avector.h"
#include "anormalize.h"
#include "asoftmax.h"
#include "aavx.h"
#include "amatmul.h"

#include "abenchmark.h"  // self-include

//---------------------------------------------------


void test_accuracy_1000(char* name, long int n, float (*fnptr)(float a, float b), int (*ifnptr)(int a, int b))
{
	float fstart = (float)-n;
	float fend = (float)n;
	long int inc = 1;

	float finc = (float)inc;
	float maxferrorabs = 0.0;
	float maxferrpercent = 0.0;

	for (float fi = fstart; fi <= fend; fi += finc) {
		for (float fj = fstart; fj <= fend; fj += finc) {
			float factual = fi * fj;   // Basic float multiplication

			float fcalc = 0.0;
			if (fnptr) {
				fcalc = fnptr(fi, fj);
			}
			else if (ifnptr) {
				fcalc = (float)ifnptr((int)fi, (int)fj);
			}

			float fpercenterr = 0.0;
			if (factual != 0.0) fpercenterr = (((float)fcalc - (float)factual) / (float)factual) * 100.0f;
			float ferr = fcalc - factual;
			float ferrabs = ferr >= 0.0 ? ferr : -ferr;
			if (ferrabs > maxferrorabs) maxferrorabs = ferrabs;
			if (fpercenterr < 0.0) fpercenterr = -fpercenterr;
			if (fpercenterr > maxferrpercent) maxferrpercent = fpercenterr;
			// Compare accuracy...
			fprintf(stdout, "%s: %3.2f * %3.2f = %3.2f vs %3.2f, error: %3.2f (%3.2f%%)\n", name,
				(float)fi,
				(float)fj,
				(float)fcalc,
				(float)factual,
				ferr,
				fpercenterr);
		}
	}
	fprintf(stdout, "%s: Max error: %3.2f %3.2f%%)\n", name, maxferrorabs, maxferrpercent);

}

void run_arith_float_1000(char* name, long int n, float (*fnptr)(float a, float b), int (*ifnptr)(int a, int b))
{
	unsigned long start_us = clock();
	if (fnptr) {
		float c = 0.0;
		float a = 555.5;
		float b = 555.5;
		for (int i = 0; i <= n; i++) {
			c = fnptr(a, b);
		}
	}
	if (ifnptr) {
		int c = 0;
		int a = 555;
		int b = 555;
		for (int i = 0; i <= n; i++) {
			c = ifnptr(a, b);
		}

	}

	unsigned long after_us = clock();
	int diff = after_us - start_us;

	fprintf(stdout, "%s: %d ticks (%3.2f seconds)\n", name, diff, diff / (double)CLOCKS_PER_SEC);

}


void run_vector_int_N(char* name, long int niter, long int nvecsize,
	int (*intvectorfnptr)(int v[], int v2[], int n))
{
	yassert(intvectorfnptr);
#define BENCHMARK_VECTOR_MAXSIZE 100000
	static int v1[BENCHMARK_VECTOR_MAXSIZE];
	static int v2[BENCHMARK_VECTOR_MAXSIZE];
	static int vtemp[BENCHMARK_VECTOR_MAXSIZE];
	yassert(nvecsize <= BENCHMARK_VECTOR_MAXSIZE);
	if (nvecsize >= BENCHMARK_VECTOR_MAXSIZE) return;  // fail
	aussie_ivector_set_1_N(v1, nvecsize);  // Dummy data 1..N
	aussie_ivector_set_1_N(v2, nvecsize);  // Dummy data 1..N
	aussie_ivector_set_1_N(vtemp, nvecsize);  // Dummy data 1..N

	int memcopybytes = nvecsize * sizeof(v1[0]);
	unsigned long start_us = clock();  // Before time
	if (intvectorfnptr) {  // Functions that return void on 1 vector
		int ret = 0;
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			//memcpy(v1, vtemp, memcopybytes);
			//memcpy(v2, vtemp, memcopybytes);
			ret = intvectorfnptr(v1, v2, nvecsize);
		}
	}

	unsigned long after_us = clock();  // After time
	int diff = after_us - start_us;

	fprintf(stdout, "%s: %d ticks (%3.2f seconds)\n", name, diff, diff / (double)CLOCKS_PER_SEC);

}


void run_vector_scalar_N(char* name, long int niter, long int nvecsize, void (*vectorscalarfnptr)(float v[], int n, float scalar))
{
	// Run a vector-scalar operation (e.g. multiply by scalar)
	const float fscalar = 1.5f;
	yassert(vectorscalarfnptr);

#undef BENCHMARK_VECTOR_MAXSIZE
#define BENCHMARK_VECTOR_MAXSIZE 10000
	yassert(nvecsize <= BENCHMARK_VECTOR_MAXSIZE);
	if (nvecsize >= BENCHMARK_VECTOR_MAXSIZE) return;  // fail

	alignas(32) float v1[BENCHMARK_VECTOR_MAXSIZE];
	//alignas(32) float v2[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) float vcopy[BENCHMARK_VECTOR_MAXSIZE];

	aussie_vector_set_1_N(v1, nvecsize);  // Dummy data 1..N
	aussie_vector_copy_basic(vcopy, v1, nvecsize);


	int memcopybytes = nvecsize * sizeof(v1[0]);
	unsigned long start_us = clock();  // Before time
	if (vectorscalarfnptr) {  // Functions that return void on 1 vector
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			memcpy(v1, vcopy, memcopybytes);
			vectorscalarfnptr(v1, nvecsize, fscalar);
		}
	}
	unsigned long after_us = clock();  // After time
	int diff = after_us - start_us;

	fprintf(stdout, "%s: %d ticks (%3.2f seconds)\n", name, diff, diff / (double)CLOCKS_PER_SEC);


}

void run_matrix_matrix_matmul(char* name, long int niter, long int nvecsize,
	void (*matrixmatrixfnptr)(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
)
{
	yassert(matrixmatrixfnptr);
#undef BENCHMARK_VECTOR_MAXSIZE
#define BENCHMARK_VECTOR_MAXSIZE 2048
	alignas(32) static ymatrix m1;
	alignas(32) static ymatrix m2;
	alignas(32) static ymatrix m3;
	yassert(nvecsize <= BENCHMARK_VECTOR_MAXSIZE);
	if (nvecsize > BENCHMARK_VECTOR_MAXSIZE) return;  // fail

	aussie_matrix_set_identity(m1);
	aussie_matrix_set_identity(m2);
	aussie_matrix_set_identity(m3);

	unsigned long start_us = clock();  // Before time
	if (matrixmatrixfnptr) {  // Functions that return void on 1 vector
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			matrixmatrixfnptr(m1, m2, nvecsize, m3);
		}
	}

	unsigned long after_us = clock();  // After time
	int diff = after_us - start_us;

	fprintf(stdout, "%s: %d ticks (%3.2f seconds)\n", name, diff, diff / (double)CLOCKS_PER_SEC);

}

void run_matrix_float_N(char* name, long int niter, long int nvecsize,
	void (*voidmatrixvectorfnptr)(const ymatrix m, const float v[], int n, float vout[])
	)
{
	yassert(voidmatrixvectorfnptr);
#undef BENCHMARK_VECTOR_MAXSIZE
#define BENCHMARK_VECTOR_MAXSIZE 2048
	alignas(32) float v1[BENCHMARK_VECTOR_MAXSIZE];
	//alignas(32) float v2[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) float vtemp[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) float vout[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) static ymatrix m;
	yassert(nvecsize <= BENCHMARK_VECTOR_MAXSIZE);
	if (nvecsize > BENCHMARK_VECTOR_MAXSIZE) return;  // fail

	aussie_matrix_set_identity(m);
	aussie_vector_set_1_N(v1, nvecsize);  // Dummy data 1..N
	//aussie_vector_set_1_N(v2, nvecsize);  // Dummy data 1..N
	aussie_vector_set_1_N(vtemp, nvecsize);  // Dummy data 1..N

	int memcopybytes = nvecsize * sizeof(v1[0]);
	unsigned long start_us = clock();  // Before time
	if (voidmatrixvectorfnptr) {  // Functions that return void on 1 vector
		float c = 0.0;
		float a = 555.5;
		float b = 555.5;
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			memcpy(v1, vtemp, memcopybytes);
			voidmatrixvectorfnptr(m, v1, nvecsize, vout);
		}
	}

	unsigned long after_us = clock();  // After time
	int diff = after_us - start_us;

	fprintf(stdout, "%s: %d ticks (%3.2f seconds)\n", name, diff, diff / (double)CLOCKS_PER_SEC);

}


void run_vector_float_N(char* name, long int niter, long int nvecsize, 
	void (*voidvectorfnptr)(const float v[], int n),
	float (*floatvectorfnptr)(const float v[], const float v2[], int n)
)
{
	yassert(voidvectorfnptr || floatvectorfnptr);
#undef BENCHMARK_VECTOR_MAXSIZE
#define BENCHMARK_VECTOR_MAXSIZE 10000
	alignas(32) float v1[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) float v2[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) float vtemp[BENCHMARK_VECTOR_MAXSIZE];
	yassert(nvecsize <= BENCHMARK_VECTOR_MAXSIZE);
	if (nvecsize >= BENCHMARK_VECTOR_MAXSIZE) return;  // fail
	aussie_vector_set_1_N(v1, nvecsize);  // Dummy data 1..N
	aussie_vector_set_1_N(v2, nvecsize);  // Dummy data 1..N
	aussie_vector_set_1_N(vtemp, nvecsize);  // Dummy data 1..N

	int memcopybytes = nvecsize * sizeof(v1[0]);
	unsigned long start_us = clock();  // Before time
	if (voidvectorfnptr) {  // Functions that return void on 1 vector
		float c = 0.0;
		float a = 555.5;
		float b = 555.5;
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			memcpy(v1, vtemp, memcopybytes);
			voidvectorfnptr(v1, nvecsize);
		}
	}

	if (floatvectorfnptr) {  // Functions that return FLOAT on 2 vectors
		float c = 0.0;
		float a = 555.5;
		float b = 555.5;
		float f = 0.0;
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			//memcpy(v1, vtemp, memcopybytes); // No need to copy, it's a read-only function like vecdot...
			//memcpy(v2, vtemp, memcopybytes);
			f = floatvectorfnptr(v1, v2, nvecsize);
		}
	}

	unsigned long after_us = clock();  // After time
	int diff = after_us - start_us;

	fprintf(stdout, "%s: %d ticks (%3.2f seconds)\n", name, diff, diff / (double)CLOCKS_PER_SEC);

}


void run_vector_float_N_non_const(char* name, long int niter, long int nvecsize,
	void (*voidvectorfnptr)(float v[], int n),
	float (*floatvectorfnptr)(float v[], float v2[], int n)
)
{
	yassert(voidvectorfnptr || floatvectorfnptr);
#undef BENCHMARK_VECTOR_MAXSIZE
#define BENCHMARK_VECTOR_MAXSIZE 10000
	alignas(32) float v1[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) float v2[BENCHMARK_VECTOR_MAXSIZE];
	alignas(32) float vtemp[BENCHMARK_VECTOR_MAXSIZE];
	yassert(nvecsize <= BENCHMARK_VECTOR_MAXSIZE);
	if (nvecsize >= BENCHMARK_VECTOR_MAXSIZE) return;  // fail
	aussie_vector_set_1_N(v1, nvecsize);  // Dummy data 1..N
	aussie_vector_set_1_N(v2, nvecsize);  // Dummy data 1..N
	aussie_vector_set_1_N(vtemp, nvecsize);  // Dummy data 1..N

	int memcopybytes = nvecsize * sizeof(v1[0]);
	unsigned long start_us = clock();  // Before time
	if (voidvectorfnptr) {  // Functions that return void on 1 vector
		float c = 0.0;
		float a = 555.5;
		float b = 555.5;
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			memcpy(v1, vtemp, memcopybytes);
			voidvectorfnptr(v1, nvecsize);
		}
	}

	if (floatvectorfnptr) {  // Functions that return FLOAT on 2 vectors
		float c = 0.0;
		float a = 555.5;
		float b = 555.5;
		float f = 0.0;
		for (int i = 0; i <= niter; i++) {
			// Re-copy test vector fast with memcpy
			//memcpy(v1, vtemp, memcopybytes); // No need to copy, it's a read-only function like vecdot...
			//memcpy(v2, vtemp, memcopybytes);
			f = floatvectorfnptr(v1, v2, nvecsize);
		}
	}

	unsigned long after_us = clock();  // After time
	int diff = after_us - start_us;

	fprintf(stdout, "%s: %d ticks (%3.2f seconds)\n", name, diff, diff / (double)CLOCKS_PER_SEC);

}

void aussie_benchmark_matrix_matrix_multiplication()
{
	long int thousand = 1000;
	long int million = 1000000;
	long int billion = 1000 * million;
	long int niter = 1 * 1;
	int nvecsize = AUSSIE_MATRIX_ROWS; //  512 * 2;  // How big a matrix/vector to test... N elements

	// Matrix-matrix multiply
	printf("Matrix-Matrix multiplication (MatMul) benchmarks (N=%d, ITER=%d):\n", nvecsize, niter);
	run_matrix_matrix_matmul("Matrix-matrix fake transpose AVX1 unrolled 4", niter, nvecsize, aussie_matmul_matrix_fake_transpose_vecdot_AVX1_inlined_unrolled4);

	

	run_matrix_matrix_matmul("Matrix-matrix fake transpose AVX1 inlined", niter, nvecsize, aussie_matmul_matrix_fake_transpose_vecdot_AVX1_inlined);
	run_matrix_matrix_matmul("Matrix-matrix fake transpose AVX2 inlined", niter, nvecsize, aussie_matmul_matrix_fake_transpose_vecdot_AVX2_inlined);
	run_matrix_matrix_matmul("Matrix-matrix fake transpose AVX1", niter, nvecsize, aussie_matmul_matrix_fake_transpose_vecdot_AVX1);
	run_matrix_matrix_matmul("Matrix-matrix fake transpose AVX2", niter, nvecsize, aussie_matmul_matrix_fake_transpose_vecdot_AVX2);

	run_matrix_matrix_matmul("Matrix-matrix fake transpose unrolled 4", niter, nvecsize, aussie_matmul_matrix_fake_transpose_unrolled4);
	run_matrix_matrix_matmul("Matrix-matrix fake transpose unrolled 8", niter, nvecsize, aussie_matmul_matrix_fake_transpose_unrolled8);

	
	run_matrix_matrix_matmul("Matrix-matrix fake transpose", niter, nvecsize, aussie_matmul_matrix_fake_transpose);

	run_matrix_matrix_matmul("Matrix-matrix basic unrolled 4", niter, nvecsize, aussie_matmul_matrix_unrolled4);

	run_matrix_matrix_matmul("Matrix-matrix mult basic", niter, nvecsize, aussie_matmul_matrix_basic);
	run_matrix_matrix_matmul("Matrix-matrix mult hoisted", niter, nvecsize, aussie_matmul_matrix_hoisted);
	
}

void aussie_benchmark_matrix_vector_multiply()
{
	long int thousand = 1000;
	long int million = 1000000;
	long int billion = 1000 * million;
	long int niter = 3 * 100;
	int nvecsize = AUSSIE_MATRIX_ROWS; //  512 * 2;  // How big a matrix/vector to test... N elements

	// Matrix-vector multiply
	printf("Matrix-Vector multiplication (MatMulVec) benchmarks (N=%d, ITER=%d):\n", nvecsize, niter);
	run_matrix_float_N("Matrix-vector nested tiled 2x2", niter, nvecsize, aussie_matmul_vector_tiled_2x2);
	run_matrix_float_N("Matrix-vector nested tiled 2x2 better", niter, nvecsize, aussie_matmul_vector_tiled_2x2_better);
	run_matrix_float_N("Matrix-vector nested tiled 2x2 hoisted", niter, nvecsize, aussie_matmul_vector_tiled_2x2_better_hoisted);
	run_matrix_float_N("Matrix-vector nested tiled 4x4", niter, nvecsize, aussie_matmul_vector_tiled_4x4);
	run_matrix_float_N("Matrix-vector nested tiled 4x4 CSE", niter, nvecsize, aussie_matmul_vector_tiled_4x4_CSE);
	run_matrix_float_N("Matrix-vector nested tiled 4x4 CSE+hoisted", niter, nvecsize, aussie_matmul_vector_tiled_4x4_CSE2);

	

	run_matrix_float_N("Matrix-vector nested interchange (basic)", niter, nvecsize, aussie_matmul_vector_basic_interchange);
	run_matrix_float_N("Matrix-vector nested interchange (hoisted)", niter, nvecsize, aussie_matmul_vector_hoisted_interchange);

	

	run_matrix_float_N("Matrix-vector row-wise vecdot", niter, nvecsize, aussie_matmul_vector_basic_out1);
	run_matrix_float_N("Matrix-vector nested loops", niter, nvecsize, aussie_matmul_vector_basic_out2);
	run_matrix_float_N("Matrix-vector nested loops simpler", niter, nvecsize, aussie_matmul_vector_basic_out3);
	run_matrix_float_N("Matrix-vector nested row-wise hoisted", niter, nvecsize, aussie_matmul_vector_basic_out2_rowwise);
	run_matrix_float_N("Matrix-vector nested ptr-arith", niter, nvecsize, aussie_matmul_vector_basic_out2_pointer_arith);
	run_matrix_float_N("Matrix-vector unrolled inner (4)", niter, nvecsize, aussie_matmul_vector_unrolled4);
	// Buggy! run_matrix_float_N("Matrix-vector unrolled inner (4B)", niter, nvecsize, aussie_matmul_vector_unrolled4b);
	run_matrix_float_N("Matrix-vector unrolled inner (8)", niter, nvecsize, aussie_matmul_vector_unrolled8);
	//run_matrix_float_N("Matrix-vector unrolled inner (8B)", niter, nvecsize, aussie_matmul_vector_unrolled8b);

	

	run_matrix_float_N("Matrix-vector vecdot AVX1 DP", niter, nvecsize, aussie_matmul_vector_vecdot_AVX1);
	run_matrix_float_N("Matrix-vector vecdot AVX2 FMA", niter, nvecsize, aussie_matmul_vector_vecdot_AVX2);


}

void aussie_benchmark_vector_exponentiation_operations()  // Vector expf benchmarks...
{
	long int million = 1000000;
	long int billion = 1000 * million;
	long int niter = 100 * 1000;
	int nvecsize = 512 * 2;  // How big a vector to test... N elements

	// Exponentiation of all elements of a vector (SIMD expf)
	printf("Vector-exponentiation operation benchmarks (N=%d, ITER=%d):\n", nvecsize, niter);
	run_vector_float_N_non_const("Vector expf basic", niter, nvecsize, aussie_vector_expf);
	run_vector_float_N_non_const("Vector expf pointer-arith", niter, nvecsize, aussie_vector_expf_pointer_arith);
#if !LINUX
	run_vector_float_N_non_const("Vector expf AVX1", niter, nvecsize, aussie_vector_expf_AVX1);
	run_vector_float_N_non_const("Vector expf AVX2", niter, nvecsize, aussie_vector_expf_AVX2);
#endif //LINUX

	

}

void aussie_benchmark_vector_scalar_operations()  // Vector-scalar benchmarks...
{
	long int million = 1000000;
	long int billion = 1000 * million;
	long int niter = 1 * million;
	int nvecsize = 512 * 2;  // How big a vector to test... N elements


	// Multiply-by-scalar
	printf("Vector-scalar operation benchmarks (N=%d, ITER=%d):\n", nvecsize, niter);
	run_vector_scalar_N("Vector mult-scalar C++", niter, nvecsize, aussie_vector_multiply_scalar);
	run_vector_scalar_N("Vector mult-scalar pointer-arith", niter, nvecsize, aussie_vector_multiply_scalar_pointer_arith);
#if !LINUX
	run_vector_scalar_N("Vector mult-scalar AVX1", niter, nvecsize, aussie_vector_multiply_scalar_AVX1);
	run_vector_scalar_N("Vector mult-scalar AVX2", niter, nvecsize, aussie_vector_multiply_scalar_AVX2);
	run_vector_scalar_N("Vector mult-scalar AVX2 + pointer arith", niter, nvecsize, aussie_vector_multiply_scalar_AVX2_pointer_arith);
#endif //LINUX
	
}

void aussie_benchmark_vecdot()  // vector dot product benchmarks...
{
	long int million = 1000000;
	long int billion = 1000 * million;
	long int niter = 1 * million;
	int nvecsize = 512 * 2;  // How big a vector to test... N elements

	printf("INT Vector dot product benchmarks: (N=%d, Iter=%d)\n", nvecsize, niter);
	run_vector_int_N("Vecdot int", niter, nvecsize, aussie_vecdot_int_basic);

	printf("FLOAT Vector dot product benchmarks: (N=%d, Iter=%d)\n", nvecsize, niter);

	run_vector_int_N("Vecdot integer (fixed-point)", niter, nvecsize, aussie_vecdot_integer_fixed_point);
	run_vector_int_N("Vecdot integer (bitshift)", niter, nvecsize, aussie_vecdot_integer_bitshift);
	run_vector_float_N("Vecdot basic", niter, nvecsize, NULL, aussie_vecdot_basic);
#if !LINUX
	run_vector_float_N("Vecdot AVX1 unroll (4 floats, 128-bits)", niter, nvecsize, NULL, aussie_vecdot_unroll_AVX1);
	run_vector_float_N("Vecdot AVX1 FMA (4 floats, 128-bits)", niter, nvecsize, NULL, aussie_vecdot_FMA_unroll_AVX1);
	run_vector_float_N("Vecdot AVX2 FMA (8 floats, 256-bits)", niter, nvecsize, NULL, aussie_vecdot_FMA_unroll_AVX2);
#endif //LINUX



#if !LINUX
	run_vector_float_N("Vecdot AVX2 unroll (8 floats, 256-bits)", niter, nvecsize, NULL, aussie_vecdot_unroll_AVX2);
#endif //LINUX



	run_vector_float_N_non_const("Vecdot pointer arith", niter, nvecsize, NULL, aussie_vecdot_pointer_arithmetic);
	run_vector_float_N_non_const("Vecdot reverse basic", niter, nvecsize, NULL, aussie_vecdot_reverse_basic);
	run_vector_float_N_non_const("Vecdot reverse basic2", niter, nvecsize, NULL, aussie_vecdot_reverse_basic2);
	run_vector_float_N_non_const("Vecdot reverse zero-test", niter, nvecsize, NULL, aussie_vecdot_reverse_zerotest);
	run_vector_float_N_non_const("Vecdot unroll4 basic", niter, nvecsize, NULL, aussie_vecdot_unroll4_basic);
	run_vector_float_N_non_const("Vecdot unroll4 better", niter, nvecsize, NULL, aussie_vecdot_unroll4_better);
	run_vector_float_N_non_const("Vecdot Duff's Device unroll", niter, nvecsize, NULL, aussie_vecdot_unroll4_duffs_device);



	//run_vector_float_N("Vecdot section512", niter, nvecsize, NULL, aussie_vecdot_section512);
	run_vector_float_N_non_const("Vecdot parallel basic", niter, nvecsize, NULL, aussie_vecdot_parallel_basic);
	run_vector_float_N_non_const("Vecdot parallel odd sizes", niter, nvecsize, NULL, aussie_vecdot_parallel_odd_sizes);
	run_vector_float_N_non_const("Vecdot parallel padding", niter, nvecsize, NULL, aussie_vecdot_parallel_padding);
	run_vector_float_N("Vecdot basic", niter, nvecsize, NULL, aussie_vecdot_basic);
}


void aussie_benchmark_softmax()
{
	long int million = 1000000;
	long int billion = 1000 * million;
	long int niter = 100 * 1000;

	int nvecsize = 1024;  // How big a vector to test... N elements

	printf("Softmax benchmarks (N=%d, ITER=%d)\n", nvecsize, niter);
	run_vector_float_N_non_const("Softmax basic", niter, nvecsize, aussie_vector_softmax_basic, NULL);
	run_vector_float_N_non_const("Softmax reciprocal", niter, nvecsize, aussie_vector_softmax_multiply_reciprocal, NULL);
	run_vector_float_N_non_const("Softmax expf-first", niter, nvecsize, aussie_vector_softmax_exponentiate_first, NULL);
	run_vector_float_N_non_const("Softmax expf-sum-fused", niter, nvecsize, aussie_vector_softmax_exponentiate_and_sum, NULL);
	run_vector_float_N_non_const("Softmax expf with AVX1", niter, nvecsize, aussie_vector_softmax_exponentiate_with_AVX1, NULL);
	run_vector_float_N_non_const("Softmax expf/sum AVX1", niter, nvecsize, aussie_vector_softmax_exponentiate_and_sum_AVX1, NULL);
	run_vector_float_N_non_const("Softmax fused expf/sum AVX1", niter, nvecsize, aussie_vector_softmax_fused_exponentiate_sum_AVX1, NULL);
	run_vector_float_N_non_const("Softmax fused expf/sum/mult AVX1", niter, nvecsize, aussie_vector_softmax_fused_exp_sum_mult_AVX1, NULL);
	run_vector_float_N_non_const("Softmax expf with AVX2", niter, nvecsize, aussie_vector_softmax_exponentiate_with_AVX2, NULL);
	run_vector_float_N_non_const("Softmax expf/sum AVX2", niter, nvecsize, aussie_vector_softmax_exponentiate_and_sum_AVX2, NULL);
	run_vector_float_N_non_const("Softmax fused expf/sum AVX2", niter, nvecsize, aussie_vector_softmax_fused_exponentiate_sum_AVX2, NULL);
	run_vector_float_N_non_const("Softmax fused expf/sum/mult AVX2", niter, nvecsize, aussie_vector_softmax_fused_exp_sum_mult_AVX2, NULL);
	
}

void aussie_benchmark_zscore_normalization()   // Benchmark z-score normalization
{
	long int million = 1000000;
	long int thousand = 1000;
	long int billion = 1000 * million;
	long int niter = 1 * million;
	int nvecsize = 128;  // How big a vector to test... N elements
	yassert(nvecsize % 4 == 0);  // AVX1
	yassert(nvecsize % 8 == 0);  // AVX2

	niter = 100 * thousand;
	nvecsize = 2048;  // How big a vector to test... N elements
	yassert(nvecsize % 4 == 0);  // AVX1
	yassert(nvecsize % 8 == 0);  // AVX2
	printf("Z-score normalization benchmarks (N=%d, ITER=%d)\n", nvecsize, niter);
	run_vector_float_N_non_const("Z-score norm basic", niter, nvecsize, aussie_vector_normalize_zscore);
	run_vector_float_N_non_const("Z-score norm fix mean", niter, nvecsize, aussie_vector_normalize_zscore_fix_mean);
	run_vector_float_N_non_const("Z-score norm reciprocal", niter, nvecsize, aussie_vector_normalize_zscore_reciprocal);
	run_vector_float_N_non_const("Z-score norm fused", niter, nvecsize, aussie_vector_normalize_zscore_fused);
	run_vector_float_N_non_const("Z-score norm AVX1 sum", niter, nvecsize, aussie_vector_normalize_zscore_sum_AVX1);
	run_vector_float_N_non_const("Z-score norm AVX1 sum+multiply", niter, nvecsize, aussie_vector_normalize_zscore_sum_mult_AVX1);
	run_vector_float_N_non_const("Z-score norm AVX2 sum", niter, nvecsize, aussie_vector_normalize_zscore_sum_AVX2);
	run_vector_float_N_non_const("Z-score norm AVX2 sum+multiply", niter, nvecsize, aussie_vector_normalize_zscore_sum_mult_AVX2);
	run_vector_float_N_non_const("Z-score norm AVX1 all", niter, nvecsize, aussie_vector_normalize_zscore_all_AVX1);
	run_vector_float_N_non_const("Z-score norm AVX2 all", niter, nvecsize, aussie_vector_normalize_zscore_all_AVX2);


}

void aussie_benchmark_minmax_normalization()   // MinMax normalization
{
	long int million = 1000000;
	long int thousand = 1000;
	long int billion = 1000 * million;

	// BatchNorm
	int nvecsize = 2048;  // How big a vector to test... N elements
	int niter = 100 * thousand;
	yassert(nvecsize % 4 == 0);  // AVX1
	yassert(nvecsize % 8 == 0);  // AVX2
	printf("MinMax benchmarks (N=%d, ITER=%d)\n", nvecsize, niter);
	run_vector_float_N_non_const("MinMax basic", niter, nvecsize, aussie_vector_normalize_min_max_basic);
	run_vector_float_N_non_const("MinMax reciprocal", niter, nvecsize, aussie_vector_normalize_min_max_reciprocal);
	run_vector_float_N_non_const("MinMax ptr arith", niter, nvecsize, aussie_vector_normalize_min_max_pointer_arith);
	run_vector_float_N_non_const("MinMax loop fusion", niter, nvecsize, aussie_vector_normalize_min_max_fusion);

	
	
}


void aussie_benchmark_RMS_normalization()   // RMSNorm normalization
{
	long int million = 1000000;
	long int thousand = 1000;
	long int billion = 1000 * million;

	// BatchNorm
	int nvecsize = 2048;  // How big a vector to test... N elements
	int niter = 100 * thousand;
	yassert(nvecsize % 4 == 0);  // AVX1
	yassert(nvecsize % 8 == 0);  // AVX2
	printf("MinMax benchmarks (N=%d, ITER=%d)\n", nvecsize, niter);
	run_vector_float_N_non_const("RMSNorm basic", niter, nvecsize, aussie_vector_rms_normalize_basic);
	run_vector_float_N_non_const("RMSNorm reciprocal", niter, nvecsize, aussie_vector_rms_normalize_reciprocal);
#if !LINUX
	run_vector_float_N_non_const("RMSNorm AVX1", niter, nvecsize, aussie_vector_rms_normalize_AVX1);
	run_vector_float_N_non_const("RMSNorm AVX2", niter, nvecsize, aussie_vector_rms_normalize_AVX2);
#endif //LINUX



}



void aussie_benchmark_batchnorm_normalization()   // BatchNorm normalization
{
	long int million = 1000000;
	long int thousand = 1000;
	long int billion = 1000 * million;

	// BatchNorm
	int nvecsize = 2048;  // How big a vector to test... N elements
	int niter = 100 * thousand;
	yassert(nvecsize % 4 == 0);  // AVX1
	yassert(nvecsize % 8 == 0);  // AVX2
	printf("BatchNorm benchmarks (N=%d, ITER=%d)\n", nvecsize, niter);
	run_vector_float_N_non_const("BatchNorm basic", niter, nvecsize, aussie_vector_batch_normalize_basic_wrapper);
	run_vector_float_N_non_const("BatchNorm reciprocal", niter, nvecsize, aussie_vector_batch_normalize_with_loop_fission_wrapper);
	run_vector_float_N_non_const("BatchNorm fission", niter, nvecsize, aussie_vector_batch_normalize_with_loop_fission2_wrapper);
	run_vector_float_N_non_const("BatchNorm fusion/fission", niter, nvecsize, aussie_vector_batch_normalize_with_loop_fusion_fission_wrapper);
	run_vector_float_N_non_const("BatchNorm no params", niter, nvecsize, aussie_vector_batch_normalize_NO_PARAMS_wrapper);

	run_vector_float_N_non_const("BatchNorm fission AVX1", niter, nvecsize, aussie_vector_batch_normalize_with_loop_fission2_wrapper_AVX1);
	run_vector_float_N_non_const("BatchNorm fission AVX2", niter, nvecsize, aussie_vector_batch_normalize_with_loop_fission2_wrapper_AVX2);
	run_vector_float_N_non_const("BatchNorm fusion/fission AVX1", niter, nvecsize, aussie_vector_batch_normalize_with_loop_fusion_fission_AVX1_wrapper);
	run_vector_float_N_non_const("BatchNorm fusion/fission AVX2", niter, nvecsize, aussie_vector_batch_normalize_with_loop_fusion_fission_AVX2_wrapper);

	printf("Variance benchmarks (N=%d, ITER=%d)\n", nvecsize, niter);
	run_vector_float_N_non_const("BatchNorm mean/variance", niter, nvecsize, aussie_vector_batchnorm_variance_basic_wrapper); // Low-level variance computations

}

void aussie_benchmark_normalization()   // Benchmark normalizations: z-score, BatchNorm (LayerNorm?)
{
	aussie_benchmark_RMS_normalization();   // RMSNorm normalization
	aussie_benchmark_minmax_normalization();     // MinMax normalization

	aussie_benchmark_batchnorm_normalization();  // BatchNorm normalization...

	aussie_benchmark_zscore_normalization();   // Benchmark z-score normalization


}

void yap_benchmark_operations()
{

	// Test benchmark performance of basic arithmetic operations
	int million = 1000000;
	long int billion = 1000 * million;
	long int n = 100 * million;
	printf("Operator benchmarks (ITER=%d)\n", n);
	run_arith_float_1000("Basic float divide", n, basic_float_divide, NULL);
	run_arith_float_1000("Basic float add", n, basic_float_add, NULL);
	run_arith_float_1000("Basic float equals", n, basic_float_equals, NULL);
	run_arith_float_1000("Basic float leq", n, basic_float_leq, NULL);
	run_arith_float_1000("Basic float geq", n, basic_float_geq, NULL);

	run_arith_float_1000("Float approx Mogami(2020)", n, float_approx_mogami, NULL);

	run_arith_float_1000("Float fake-add multiply", n, float_fake_add, NULL);
	run_arith_float_1000("Float truncated int's multiply", n, float_convert_to_int_multiply, NULL);


	run_arith_float_1000("Basic int multiply", n, NULL, basic_int_multiply);
	run_arith_float_1000("Basic int add", n, NULL, basic_int_add);
	run_arith_float_1000("Basic int divide", n, NULL, basic_int_divide);
	run_arith_float_1000("Basic int mod", n, NULL, basic_int_mod);
	run_arith_float_1000("Basic int bitor", n, NULL, basic_int_bitor);
	run_arith_float_1000("Basic int bitxor", n, NULL, basic_int_bitxor);
	run_arith_float_1000("Basic int bitand", n, NULL, basic_int_bitand);
	run_arith_float_1000("Basic int bitshift-left", n, NULL, basic_int_bitshift_left);

}

void yap_test_operator_accuracy()    // Test approximate operations accuracy...
{
	int n = 5;
	printf("Operator accuracy tests (N=%d)\n", n);
	test_accuracy_1000("Basic multiply", n, basic_float_multiply, NULL);
	//test_accuracy_1000("Float fake-add multiply", n, float_fake_add, NULL);
	test_accuracy_1000("Float approx Mogami", n, float_approx_mogami, NULL);


}

void aussie_benchmark_all()
{
	aussie_benchmark_matrix_matrix_multiplication();
	aussie_benchmark_matrix_vector_multiply();
	aussie_benchmark_softmax();
	aussie_benchmark_vector_exponentiation_operations();
	aussie_benchmark_vector_scalar_operations();  // Vector-scalar benchmarks...
	aussie_benchmark_vecdot();  // vector dot product benchmarks...
	aussie_benchmark_normalization();
	yap_benchmark_operations();
}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

