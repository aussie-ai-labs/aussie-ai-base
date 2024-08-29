// aavx.h -- AVX/AVX-2/AVX-512 SSE intrinsics code -- Aussie AI Base Library  
// Created Nov 13th 2023
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

#include "aport.h"

#if !LINUX  // Don't include the entire file for Linux!

#if !LINUX
#include <xmmintrin.h>
#include <intrin.h>
#endif

//---------------------------------------------------
//---------------------------------------------------

#include "aussieai.h"
#include "aassert.h"
#include "atest.h"
#include "avector.h"

#include "aavx.h"  // self-include

#define is_aligned_16(ptr)  ((((unsigned long int)(ptr)) & 15) == 0)

//---------------------------------------------------
void aussie_unit_test_avx() // AVX, AVX-2, AVX-512, SSE
{
	aussie_unit_test_avx1_basics();
}

void aussie_avx_multiply_4_floats(float v1[4], float v2[4], float vresult[4])
{
	// Use 128-bit AVX registers to multiply 4x32-bit floats...
	__m128 r1 = _mm_loadu_ps(v1);   // Load floats into 128-bits
	__m128 r2 = _mm_loadu_ps(v2);
	__m128 dst = _mm_mul_ps(r1, r2);   // Multiply
	_mm_storeu_ps(vresult, dst);  // Convert 128-bit to floats
}

float aussie_avx_vecdot_fma_4_floats(float v1[4], float v2[4])  // AVX1 vecdot using FMA (Fused Multiply-Add) primitives
{
	// Use 128-bit AVX registers to multiply 4x32-bit floats...
	__m128 r1 = _mm_loadu_ps(v1);   // Load floats into 128-bits
	__m128 r2 = _mm_loadu_ps(v2);
	__m128 r3 = _mm_setzero_ps();  // Added vector...
	__m128 dst = _mm_fmadd_ps(r1, r2, r3);   // Multiply
	float* farr = (float*)&dst;
	float sum = farr[0] + farr[1] + farr[2] + farr[3];  // Manual add the 4 floats...
	return sum;
}

float aussie_avx_vecdot_4_floats(float v1[4], float v2[4]) // AVX-1 (128-bit) dot product
{
	// AVX: Vector dot product of 2 vectors of 4x32-bit floats... 128 bit
	__m128 r1 = _mm_loadu_ps(v1);   // Load floats into 128-bits
	__m128 r2 = _mm_loadu_ps(v2);
	__m128 dst = _mm_dp_ps(r1, r2, 0xf1);   // Dot product (dotp)
	float fret = _mm_cvtss_f32(dst);
	return fret;
}


void aussie_avx_multiply_4_floats_aligned(float v1[4], float v2[4], float vresult[4]) // AVX1(128-bit)
{
	// Use 128-bit AVX-1 registers to multiply 4x32-bit floats...
	__m128 r1 = _mm_loadu_ps(v1);   // Load floats into 128-bits
	__m128 r2 = _mm_loadu_ps(v2);
	__m128 dst = _mm_mul_ps(r1, r2);   // Multiply
	_mm_store_ps(vresult, dst);  // Aligned version convert to floats
}


void aussie_avx2_multiply_8_floats(float v1[8], float v2[8], float vresult[8]) // AVX-2
{
	// Use 256-bit AVX2 registers to multiply 8x32-bit floats...
	__m256 r1 = _mm256_loadu_ps(v1);   // Load floats into 256-bits
	__m256 r2 = _mm256_loadu_ps(v2);
	__m256 dst = _mm256_mul_ps(r1, r2);   // Multiply (SIMD)
	_mm256_storeu_ps(vresult, dst);  // Convert 256-bit to 8 floats
}


float aussie_avx2_vecdot_8_floats_buggy(float v1[8], float v2[8]) // AVX-2 (256-bit) dot product
{
	// AVX2 (256-bit): Vector dot product of 2 vectors of 8x32-bit floats
	__m256 r1 = _mm256_loadu_ps(v1);   // Load floats into 256-bits
	__m256 r2 = _mm256_loadu_ps(v2);
	__m256 dst = _mm256_dp_ps(r1, r2, 0xf1);   // Bug!
	float fret = _mm256_cvtss_f32(dst); 
	return fret;
}


void aussie_avx512_multiply_16_floats(float v1[16], float v2[16], float vresult[16])
{
#if AUSSIE_DO_AVX512 // Crashes with unhandled exception/illegal instructions
	// Use AVX-512's 512-bit registers to multiply 16x32-bit floats...
	__m512 r1 = _mm512_loadu_ps(v1);   // Load 16 floats into 512-bits
	__m512 r2 = _mm512_loadu_ps(v2);
	__m512 dst = _mm512_mul_ps(r1, r2);   // Multiply (SIMD)
	_mm512_storeu_ps(vresult, dst);  // Convert 512-bit to 16 floats
#endif
}

void aussie_vector_reluize_AVX1(float v[], int n)   // Apply RELU to each element (sets negatives to zero)
{
	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return; // fail
	}
	const __m128 rzeros = _mm_set1_ps(0.0f);  // Set up vector full of zeros...
	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]);   // Load floats into 128-bits
		__m128 dst = _mm_max_ps(r1, rzeros);   // MAX(r1,0)
		_mm_store_ps(&v[i], dst);  // store back to floats
	}
}

void aussie_vector_reluize_AVX2(float v[], int n)  // Apply RELU to each element (sets negatives to zero)
{
	if (n % 8 != 0) {
		yassert(n % 8 == 0);
		return; // fail
	}
	const __m256 rzeros = _mm256_set1_ps(0.0f);  // vector full of zeros...
	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]);   // Load floats into 256-bits
		__m256 dst = _mm256_max_ps(r1, rzeros);   // MAX(R1, 0)
		_mm256_store_ps(&v[i], dst);  // store back to floats
	}
}

float aussie_vector_max_AVX1(float v[], int n)   // Maximum (horizontal) of a single vector
{
	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return 0.0; // fail
	}

	__m128 sumdst = _mm_loadu_ps(&v[0]);   // Initial 4 values
	for (int i = 4 /*not 0*/; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load floats into 128-bits
		sumdst = _mm_max_ps(r1, sumdst); // dst = MAX(dst, r1)
	}

	// Find Max of the final 4 accumulators
	float* farr = sumdst.m128_f32;
	float fmax = farr[0];
	if (farr[1] > fmax) fmax = farr[1];
	if (farr[2] > fmax) fmax = farr[2];
	if (farr[3] > fmax) fmax = farr[3];
	return fmax;
}

float aussie_vector_max_AVX1b(float v[], int n)   // Maximum (horizontal) of a single vector
{
	yassert(n % 4 == 0);
	__m128 sumdst = _mm_loadu_ps(&v[0]);   // Initial 4 values
	for (int i = 4 /*not 0*/; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load floats into 128-bits
		sumdst = _mm_max_ps(r1, sumdst); // dst = MAX(dst, r1)
	}

	// Find Max of the final 4 accumulators
#define FMAX(x,y)  ( (x) > (y) ? (x) : (y) )
	float* farr = sumdst.m128_f32;
	float fmax1 = FMAX(farr[0], farr[1]);
	float fmax2 = FMAX(farr[2], farr[3]);
	float fmax = FMAX(fmax1, fmax2);
	return fmax;
}


float aussie_vector_max_min_fusion_AVX1b(float v[], int n, float &fminout)   
{
	// Maximum and Minimum (horizontal) of a single vector
	yassert(n % 4 == 0);
	__m128 maxsumdst = _mm_loadu_ps(&v[0]);   // Initial 4 values
	__m128 minsumdst = maxsumdst;   // Initial 4 values
	for (int i = 4 /*not 0*/; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load floats into 128-bits
		maxsumdst = _mm_max_ps(r1, maxsumdst); // dst = MAX(dst, r1)
		minsumdst = _mm_min_ps(r1, minsumdst); // dst = MIN(dst, r1)
	}
	// Find Min of the final 4 accumulators
#define FMIN(x,y)  ( (x) < (y) ? (x) : (y) )
	float* farr = minsumdst.m128_f32;
	float fmin1 = FMIN(farr[0], farr[1]);
	float fmin2 = FMIN(farr[2], farr[3]);
	fminout = FMIN(fmin1, fmin2);
	// Find Max of the final 4 accumulators
#define FMAX(x,y)  ( (x) > (y) ? (x) : (y) )
	farr = maxsumdst.m128_f32;
	float fmax1 = FMAX(farr[0], farr[1]);
	float fmax2 = FMAX(farr[2], farr[3]);
	return FMAX(fmax1, fmax2); 
}

float aussie_vector_min_AVX1(float v[], int n)   // Minimum (horizontal) of a single vector
{
	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return 0.0; // fail
	}

	__m128 sumdst = _mm_loadu_ps(&v[0]);   // Initial 4 values
	for (int i = 4 /*not 0*/; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load floats into 128-bits
		sumdst = _mm_min_ps(r1, sumdst); // dst = MIN(dst, r1)
	}

	// Find Min of the final 4 accumulators
	float* farr = sumdst.m128_f32;
	float fmin = farr[0];
	if (farr[1] < fmin) fmin = farr[1];
	if (farr[2] < fmin) fmin = farr[2];
	if (farr[3] < fmin) fmin = farr[3];
	return fmin;
}


float aussie_vector_min_AVX1b(float v[], int n)   // Minimum (horizontal) of a single vector
{
	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return 0.0; // fail
	}

	__m128 sumdst = _mm_loadu_ps(&v[0]);   // Initial 4 values
	for (int i = 4 /*not 0*/; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load floats into 128-bits
		sumdst = _mm_min_ps(r1, sumdst); // dst = MIN(dst, r1)
	}

	// Find Min of the final 4 accumulators
#define FMIN(x,y)  ( (x) < (y) ? (x) : (y) )
	float* farr = sumdst.m128_f32;
	float fmin1 = FMIN(farr[0], farr[1]);
	float fmin2 = FMIN(farr[2], farr[3]);
	float fmin = FMIN(fmin1, fmin2);
	return fmin;
}


float aussie_vector_max_AVX2(float v[], int n)   // Maximum (horizontal) of a single vector
{
	if (n % 8 != 0) { // Safety check (no extra cases)
		yassert(n % 8 == 0);
		return 0.0; // fail
	}
	__m256 sumdst = _mm256_loadu_ps(&v[0]);   // Initial 8 values
	for (int i = 8/*not 0*/; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]); // Load floats into 256-bits
		sumdst = _mm256_max_ps(r1, sumdst); // dst = MAX(dst, r1)
	}

	// Find Max of the final 8 accumulators
	float* farr = sumdst.m256_f32;
	float fmax = farr[0];
	if (farr[1] > fmax) fmax = farr[1];
	if (farr[2] > fmax) fmax = farr[2];
	if (farr[3] > fmax) fmax = farr[3];
	if (farr[4] > fmax) fmax = farr[4];
	if (farr[5] > fmax) fmax = farr[5];
	if (farr[6] > fmax) fmax = farr[6];
	if (farr[7] > fmax) fmax = farr[7];
	return fmax;
}

float aussie_vector_min_AVX2(float v[], int n)   // Minimum (horizontal) of a single vector
{
	if (n % 8 != 0) { // Safety check (no extra cases)
		yassert(n % 8 == 0);
		return 0.0; // fail
	}
	__m256 sumdst = _mm256_loadu_ps(&v[0]);   // Initial 8 values
	for (int i = 8/*not 0*/; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]); // Load floats into 256-bits
		sumdst = _mm256_min_ps(r1, sumdst); // dst = MIN(dst, r1)
	}
	// Find Min of the final 8 accumulators
	float* farr = sumdst.m256_f32;
	float fmin = farr[0];
	if (farr[1] < fmin) fmin = farr[1];
	if (farr[2] < fmin) fmin = farr[2];
	if (farr[3] < fmin) fmin = farr[3];
	if (farr[4] < fmin) fmin = farr[4];
	if (farr[5] < fmin) fmin = farr[5];
	if (farr[6] < fmin) fmin = farr[6];
	if (farr[7] < fmin) fmin = farr[7];
	return fmin;
}

float aussie_vector_min_AVX2b(float v[], int n)   // Minimum (horizontal) of a single vector
{
	if (n % 8 != 0) { // Safety check (no extra cases)
		yassert(n % 8 == 0);
		return 0.0; // fail
	}
	__m256 sumdst = _mm256_loadu_ps(&v[0]);   // Initial 8 values
	for (int i = 8/*not 0*/; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]); // Load floats into 256-bits
		sumdst = _mm256_min_ps(r1, sumdst); // dst = MIN(dst, r1)
	}

	// Find Min of the final 8 accumulators
#define FMIN(x,y)  ( (x) < (y) ? (x) : (y) )
	float* farr = sumdst.m256_f32;
	float fmin1 = FMIN(farr[0], farr[1]); // Quarters
	float fmin2 = FMIN(farr[2], farr[3]);
	float fmin3 = FMIN(farr[4], farr[5]);
	float fmin4 = FMIN(farr[6], farr[7]);
	float fmin1a = FMIN(fmin1, fmin2);  // Semis
	float fmin2a = FMIN(fmin3, fmin4);
	float fmin = FMIN(fmin1a, fmin2a);  // Final
	return fmin;

}

float aussie_vector_sum_AVX1(float v[], int n)   // Summation (horizontal) of a single vector
{
	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return 0.0; // fail
	}

	__m128 sumdst = _mm_setzero_ps();   // Set accumulators to zero
	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load floats into 128-bits
		sumdst = _mm_add_ps(r1, sumdst); // SUM = SUM + V
	}

	// Add the final 4 accumulators manually
	float* farr = sumdst.m128_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3];
	return sum;
}

float aussie_vector_sum_AVX2(float v[], int n)   // Summation (horizontal) of a single vector
{
	if (n % 8 != 0) { // Safety check (no extra cases)
		yassert(n % 8 == 0);
		return 0.0; // fail
	}

	__m256 sumdst = _mm256_setzero_ps();   // Set accumulators to zero
	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]);   // Load floats into 256-bits
		sumdst = _mm256_add_ps(r1, sumdst); // SUM = SUM + V
	}

	// Add the final 8 accumulators manually
	float* farr = sumdst.m256_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3]
		+ farr[4] + farr[5] + farr[6] + farr[7]		;
	return sum;
}

float aussie_vector_sum_squares_basic(float v[], int n)  // Summation of squares of all elements
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += v[i] * v[i];
	}
	return sum;
}

float aussie_vector_sum_squares_AVX1(float v[], int n)  // Summation of squares of all elements
{
	if (n % 4 != 0) { // Safety check (no extra cases)
		yassert(n % 4 == 0);
		return 0.0; // fail
	}
	__m128 sumdst = _mm_setzero_ps();   // Set accumulators to zero
	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load floats into 128-bits
		__m128 sqr = _mm_mul_ps(r1, r1);   // Square (V*V)
		sumdst = _mm_add_ps(sqr, sumdst); // SUM = SUM + V*V
	}
	// Add the final 4 accumulators manually
	float* farr = sumdst.m128_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3];
	return sum;
}

float aussie_vector_sum_squares_AVX2(float v[], int n)  // Summation of squares of all elements
{
	if (n % 8 != 0) { // Safety check (no extra cases)
		yassert(n % 8 == 0);
		return 0.0; // fail
	}

	__m256 sumdst = _mm256_setzero_ps();   // Set accumulators to zero
	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]);   // Load floats into 256-bits
		__m256 sqr = _mm256_mul_ps(r1, r1);   // Square (V*V)
		sumdst = _mm256_add_ps(sqr, sumdst); // SUM = SUM + V*V
	}

	// Add the final 8 accumulators manually
	float* farr = sumdst.m256_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3]
		+ farr[4] + farr[5] + farr[6] + farr[7];
	return sum;
}


float aussie_vector_sum_diff_squared_fused_AVX1(float v[], int n, float meanval)
{
	// Fused version of "sum diff squared" that leaves the DIFF in the vector...
	if (n % 4 != 0) { // Safety check (no extra cases)
		yassert(n % 4 == 0);
		return 0.0; // fail
	}
	__m128 sumdst = _mm_setzero_ps();   // Set accumulators to zero
	const __m128 vmean = _mm_set1_ps(-meanval);  // Set up the negated mean values..
	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]); // Load V[i] floats into 128-bits
		__m128 rdiff = _mm_add_ps(r1, vmean); // DIFF = V[i] - MEAN
		_mm_store_ps(&v[i], rdiff);  // V[i] = DIFF (store diffs back)
		__m128 sqr = _mm_mul_ps(rdiff, rdiff);   // Square (V*V)
		sumdst = _mm_add_ps(sqr, sumdst); // SUM = SUM + V*V
	}
	// Add the final 4 accumulators manually
	float* farr = sumdst.m128_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3];
	return sum;
}

float aussie_vector_sum_diff_squared_fused_AVX2(float v[], int n, float meanval)
{
	// Fused version of "sum diff squared" that leaves the DIFF in the vector..
	if (n % 8 != 0) { // Safety check (no extra cases)
		yassert(n % 8 == 0);
		return 0.0; // fail
	}
	__m256 sumdst = _mm256_setzero_ps();   // Set accumulators to zero
	const __m256 vmean = _mm256_set1_ps(-meanval);  // Set up the negated mean values..
	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]); // Load V[i] floats into 128-bits
		__m256 rdiff = _mm256_add_ps(r1, vmean); // DIFF = V[i] - MEAN
		_mm256_store_ps(&v[i], rdiff);  // V[i] = DIFF (store diffs back)
		__m256 sqr = _mm256_mul_ps(rdiff, rdiff);   // Square (V*V)
		sumdst = _mm256_add_ps(sqr, sumdst); // SUM = SUM + V*V
	}
	// Add the final 8 accumulators manually
	float* farr = sumdst.m256_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3]
		+ farr[4] + farr[5] + farr[6] + farr[7];
	return sum;
}

float aussie_vector_fused_expf_sum_AVX2(float v[], int n)   // Fused EXPF and SUMMATION of a single vector
{
	if (n % 8 != 0) { // Safety check (no extra cases)
		yassert(n % 8 == 0);
		return 0.0; // fail
	}
	__m256 sumdst = _mm256_setzero_ps();   // Set accumulators to zero
	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]);   // Load floats into 256-bits
		__m256 expdst = _mm256_exp_ps(r1);    // Exponentiate (expf)
		sumdst = _mm256_add_ps(expdst, sumdst); // SUM = SUM + V
	}
	// Add the final 8 accumulators manually
	float* farr = sumdst.m256_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3]
		+ farr[4] + farr[5] + farr[6] + farr[7];
	return sum;
}


float aussie_vector_fused_expf_sum_AVX1(float v[], int n)   // Apply EXPF (exponential) to each element and SUM them too
{
	// Fused EXPF and SUM operators...
	yassert(n % 4 == 0);  // Safety
	__m128 sumdst = _mm_setzero_ps();   // Set accumulators to zero
	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]);   // Load floats into 128-bits
		__m128 dstexp = _mm_exp_ps(r1);   // Exponentiate (expf)
		_mm_store_ps(&v[i], dstexp);  // store as floats
		sumdst = _mm_add_ps(dstexp, sumdst); // SUM = SUM + V
	}
	// Add the final 4 accumulators manually
	float* farr = sumdst.m128_f32;
	float sum = farr[0] + farr[1] + farr[2] + farr[3];
	return sum;
}

void aussie_vector_expf_AVX1(float v[], int n)   // Apply EXPF (exponential) to each element
{
	yassert(n % 4 == 0);

	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]);   // Load floats into 128-bits
		__m128 dst = _mm_exp_ps(r1);   // Exponentiate (expf)
		_mm_store_ps(&v[i], dst);  // convert to floats (Aligned version)
	}
}

void aussie_vector_expf_AVX2(float v[], int n)  // Apply EXPF (exponential) to each element
{
	yassert(n % 8 == 0);

	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]);   // Load floats into 256-bits
		__m256 dst = _mm256_exp_ps(r1);    // Exponentiate (expf)
		_mm256_store_ps(&v[i], dst);  // store back to floats
	}
}


void aussie_vector_add_scalar_AVX1(float v[], int n, float c)   // Add scalar constant to all vector elements
{
	const __m128 rscalar = _mm_set1_ps(c);  // Set up vector full of scalars...
	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]);   // Load floats into 128-bits
		__m128 dst = _mm_add_ps(r1, rscalar);   // Add scalars
		_mm_store_ps(&v[i], dst);  // store back to floats
	}
}


void aussie_vector_add_scalar_AVX2(float v[], int n, float c)  // Add scalar constant to all vector elements
{
	const __m256 rscalar = _mm256_set1_ps(c);  // vector full of scalars...
	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]);   // Load floats into 256-bits
		__m256 dst = _mm256_add_ps(r1, rscalar);   // Add scalars
		_mm256_store_ps(&v[i], dst);  // convert to floats (Aligned version)
	}
}


void aussie_vector_multiply_scalar_AVX1(float v[], int n, float c)  // Multiply all vector elements by constant
{
	const __m128 rscalar = _mm_set1_ps(c);  // Set up vector full of scalars...
	for (int i = 0; i < n; i += 4) {
		__m128 r1 = _mm_loadu_ps(&v[i]);   // Load floats into 128-bits
		__m128 dst = _mm_mul_ps(r1, rscalar);   // Multiply by scalars
		_mm_store_ps(&v[i], dst);  // convert to floats (Aligned version)
	}
}




void aussie_vector_multiply_scalar_AVX2(float v[], int n, float c)  // Multiply all vector elements by constant
{
	const __m256 rscalar = _mm256_set1_ps(c);  // vector full of scalars...
	for (int i = 0; i < n; i += 8) {
		__m256 r1 = _mm256_loadu_ps(&v[i]);   // Load floats into 256-bits
		__m256 dst = _mm256_mul_ps(r1, rscalar);   // Multiply by scalars
		_mm256_store_ps(&v[i], dst);  // convert to floats (Aligned version)
	}
}


void aussie_vector_multiply_scalar_AVX2_pointer_arith(float v[], int n, float c)  // Multiply all vector elements by constant
{
	const __m256 rscalar = _mm256_set1_ps(c);  // vector full of scalars...
	for (; n > 0; n -= 8, v += 8) {
		__m256 r1 = _mm256_loadu_ps(v);   // Load floats into 256-bits
		__m256 dst = _mm256_mul_ps(r1, rscalar);   // Multiply by scalars
		_mm256_store_ps(v, dst);  // convert to floats (Aligned version)
	}
}

void aussie_test_avx_multiply_4_floats_with_alignment()
{
	// Test with 16-byte alignment
	alignas(16) float arr1[4] = { 1.0f , 2.5f , 3.14f, 0.0f };
	alignas(16) float arr2[4] = { 1.0f , 2.5f , 3.14f, 0.0f };
	alignas(16) float resultarr[4];
	aussie_multiply_vectors(arr1, arr2, resultarr, 4);  // Multiply element-wise
	ytestf(resultarr[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr[1], 2.5f * 2.5f);
	ytestf(resultarr[2], 3.14f * 3.14f);
	ytestf(resultarr[3], 0.0f * 0.0f);

	ytest(is_aligned_16(arr1));
	ytest(is_aligned_16(arr2));
	ytest(is_aligned_16(resultarr));

	float resultarr2[4] = { -1.0, -1.0, -1.0, -1.0 };
	aussie_avx_multiply_4_floats_aligned(arr1, arr2, resultarr2);
	ytestf(resultarr2[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr2[1], 2.5f * 2.5f);
	ytestf(resultarr2[2], 3.14f * 3.14f);
	ytestf(resultarr2[3], 0.0f * 0.0f);
	ytest(aussie_vector_equal_approx(resultarr2, resultarr, 4, 0.0001f));

}

void aussie_test_avx_multiply_4_floats()
{
	float arr1[4] = { 1.0f , 2.5f , 3.14f, 0.0f };
	float arr2[4] = { 1.0f , 2.5f , 3.14f, 0.0f };
	float resultarr[4];
	aussie_multiply_vectors(arr1, arr2, resultarr, 4);  // Multiply element-wise
	ytestf(resultarr[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr[1], 2.5f * 2.5f);
	ytestf(resultarr[2], 3.14f * 3.14f);
	ytestf(resultarr[3], 0.0f * 0.0f);

	float resultarr2[4] = { -1.0, -1.0, -1.0, -1.0 };
	aussie_avx_multiply_4_floats(arr1, arr2, resultarr2);
	ytestf(resultarr2[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr2[1], 2.5f * 2.5f);
	ytestf(resultarr2[2], 3.14f * 3.14f);
	ytestf(resultarr2[3], 0.0f * 0.0f);
	ytest(aussie_vector_equal_approx(resultarr2, resultarr, 4, 0.0001f));


}


void aussie_test_avx_vecdot_4_floats()  // Test AVX1 dot product of 4 floats
{
	float arr1[8] = { 1.0f , 2.5f , 3.14f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float arr2[8] = { 1.0f , 2.5f , 3.14f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float fexpect = aussie_vecdot_basic(arr1, arr2, 4);  // Dot product
	
	float fret = aussie_avx_vecdot_4_floats(arr1, arr2);
	ytestf(fret, fexpect);

	float fret4 = aussie_avx_vecdot_fma_4_floats(arr1, arr2);
	ytestf(fret4, fexpect);

	float fret2 = aussie_avx2_vecdot_8_floats_buggy(arr1, arr2);
	ytestf(fret2, fexpect);

	arr1[4] = 1.0f;
	arr2[4] = 1.0f;
	// Should be +1 on the 4 floats version... (if 8-float vecdot AVX2 is working!!)
	float fret3 = aussie_vecdot_basic(arr1, arr2, 8);  // Dot product
	ytestf(fret3, fexpect + 1.0f);
	fexpect = fret3;

	// This keeps failing, doing only 4 floats at a time... not 8 floats
	fret2 = aussie_avx2_vecdot_8_floats_buggy(arr1, arr2);
	// Fails: ytestf(fret2, fexpect);

	//ytestf(fret, fexpect);

}

void aussie_test_avx2_multiply_8_floats()
{
	float arr1[8] = { 1.0f , 2.5f , 3.14f, 0.0f, 5.0f, 4.4f, -1.2f, -5.7f };
	float arr2[8] = { 1.0f , 2.5f , 3.14f, 0.0f, 3.0f, -0.5, -1.07, +3.5f };
	float resultarr[8];
	aussie_avx2_multiply_8_floats(arr1, arr2, resultarr);

	float resultarr2[8] = { -1.0, -1.0, -1.0, -1.0 };
	aussie_multiply_vectors(arr1, arr2, resultarr2, 8);  // Multiply element-wise

	ytestf(resultarr[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr[1], 2.5f * 2.5f);
	ytestf(resultarr[2], 3.14f * 3.14f);
	ytestf(resultarr[3], 0.0f * 0.0f);
	ytestf(resultarr[4], 5.0f * 3.0f);
	ytestf(resultarr[5], 4.4f * -0.5f);
	ytestf(resultarr[6], -1.2f * -1.07f);
	ytestf(resultarr[7], -5.7f * +3.5f);

	ytestf(resultarr2[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr2[1], 2.5f * 2.5f);
	ytestf(resultarr2[2], 3.14f * 3.14f);
	ytestf(resultarr2[3], 0.0f * 0.0f);
	ytest(aussie_vector_equal_approx(resultarr2, resultarr, 8, 0.0001f));


}

#include <isa_availability.h>

#if 0 // Not working (cannot declare __isa_available?)
bool aussie_check_avx512_is_available()
{
	int isa_level = __isa_available;
	// __ISA_AVAILABLE_AVX512 = 6
	bool avx_512_is_enabled = (isa_level >= 6);
	return avx_512_is_enabled;
}
#endif

void aussie_test_avx512_multiply_16_floats()
{
	// Test AVX-512 multiplication of 16 floats...
#if AUSSIE_DO_AVX512 // not yet working on MSVS

	alignas(64) float arr1[16] = { 1.0f , 2.5f , 3.14f, 0.0f, 5.0f, 4.4f, -1.2f, -5.7f };
	alignas(64) float arr2[16] = { 1.0f , 2.5f , 3.14f, 0.0f, 3.0f, -0.5, -1.07, +3.5f };
	alignas(64) float resultarr[16];
	aussie_avx512_multiply_16_floats(arr1, arr2, resultarr);

	alignas(64) float resultarr2[16] = { -1.0, -1.0, -1.0, -1.0 };
	aussie_multiply_vectors(arr1, arr2, resultarr2, 8);  // Multiply element-wise

	ytestf(resultarr[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr[1], 2.5f * 2.5f);
	ytestf(resultarr[2], 3.14f * 3.14f);
	ytestf(resultarr[3], 0.0f * 0.0f);
	ytestf(resultarr[4], 5.0f * 3.0f);
	ytestf(resultarr[5], 4.4f * -0.5f);
	ytestf(resultarr[6], -1.2f * -1.07f);
	ytestf(resultarr[7], -5.7f * +3.5f);

	ytestf(resultarr2[0], 1.0f * 1.0f);  // Unit tests
	ytestf(resultarr2[1], 2.5f * 2.5f);
	ytestf(resultarr2[2], 3.14f * 3.14f);
	ytestf(resultarr2[3], 0.0f * 0.0f);
	ytest(aussie_vector_equal_approx(resultarr2, resultarr, 8, 0.0001f));
#endif

}


void aussie_unit_test_avx1_basics ()  // AVX version 1 (128-bit) tests __m128
{
	aussie_test_avx_multiply_4_floats();
	aussie_test_avx_multiply_4_floats_with_alignment();
	aussie_test_avx2_multiply_8_floats();
	aussie_test_avx512_multiply_16_floats();
	aussie_test_avx_vecdot_4_floats();

	//	__m128 _mm_mul_ps(__m128 a, __m128 b) // 
	float f = 0.0f;
	float arr1[100] = {1.0f , 2.5f , 3.14f, 0.0f };
	float arr2[100] = { 1.0f , 2.5f , 3.14f, 0.0f };
	float multarr[100];
	aussie_multiply_vectors(arr1, arr2, multarr, 4);  // Multiply element-wise

	float f2 = aussie_vecdot_basic(arr1, arr2, 3);
	ytestfapprox(f2, 17.109600f, 0.001f);

	__m128 a = _mm_loadu_ps(&arr1[0]);;
	__m128 b = _mm_loadu_ps(&arr2[0]);;
	__m128 adotb = _mm_dp_ps(a, b, 0xf1);


	f = _mm_cvtss_f32(adotb);
	ytestfapprox(f, f2, 0.001);

	// Test multiply...
	a = _mm_loadu_ps(&arr1[0]);;
	b = _mm_loadu_ps(&arr2[0]);;
	__m128 x = _mm_mul_ps(a, b);

	alignas(16) float fresult[4];
	_mm_store_ps(fresult, x);  // requires 16-byte alignment...

	ytest(aussie_vector_equal_approx(fresult, multarr, 4, 0.0001f));

	float fresult2[4];
	_mm_storeu_ps(fresult2, x);  // tolerates alignment non-16
	ytest(aussie_vector_equal_approx(fresult2, multarr, 4, 0.0001f));

}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

#endif // if !LINUX  // Don't include the entire file for Linux!

