// ysoftmax.cpp -- The Softmax function -- Aussie AI Base Library  
// Created Nov 12th 2023
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
#include "atest.h"
#include "avector.h"
#include "aavx.h"

#include "asoftmax.h"  // self-include

//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------
float aussie_vector_sum_of_exponentials(float v[], int n)
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		float e = expf(v[i]);  // expf is float version of exp
		yassert(e >= 0.0f);
		sum += e;
	}
	return sum;
}

void aussie_vector_softmax_basic(float v[], int n)
{
	float denom = aussie_vector_sum_of_exponentials(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	for (int i = 0; i < n; i++) {
		v[i] = expf(v[i]) / denom;   // expf is float version of exp
	}
}

//---------------------------------------------------
void aussie_vector_softmax_multiply_reciprocal(float v[], int n)
{
	float denom = aussie_vector_sum_of_exponentials(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] = expf(v[i]) * recip; 
	}
}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
void aussie_vector_softmax_exponentiate_first(float v[], int n)
{
	// Calculate the expf() values first into the vector (to avoid doing it twice)

	// Element-wise expf on vector...
	aussie_vector_expf(v, n);  // Element-wise expf...
	float denom = aussie_vector_sum(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
}

void aussie_vector_softmax_exponentiate_and_sum_AVX1(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Calculate the expf() values first into the vector (to avoid doing it twice)
	yassert(n % 4 == 0);

	// Element-wise expf on vector...
	aussie_vector_expf_AVX1(v, n);  // Element-wise expf...
	float denom = aussie_vector_sum_AVX1(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
#endif //LINUX
}



void aussie_vector_softmax_fused_exp_sum_mult_AVX1(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Calculate the expf() values first into the vector (to avoid doing it twice)
	yassert(n % 4 == 0);
	float denom = aussie_vector_fused_expf_sum_AVX1(v, n);  // Element-wise expf fused with SUM...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	aussie_vector_multiply_scalar_AVX1(v, n, recip);
#endif //LINUX
}

void aussie_vector_softmax_fused_exponentiate_sum_AVX1(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Calculate the expf() values first into the vector (to avoid doing it twice)
	yassert(n % 4 == 0);

	// Element-wise expf on vector...
	float denom = aussie_vector_fused_expf_sum_AVX1(v, n);  // Element-wise expf fused with SUM...
	//float denom = aussie_vector_sum_AVX1(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
#endif //LINUX
}

//---------------------------------------------------
void aussie_vector_softmax_exponentiate_with_AVX1(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Calculate the expf() values first into the vector (to avoid doing it twice)
	yassert(n % 4 == 0);

	// Element-wise expf on vector...
	aussie_vector_expf_AVX1(v, n);  // Element-wise expf...
	float denom = aussie_vector_sum(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
#endif //LINUX
}

//---------------------------------------------------
void aussie_vector_softmax_exponentiate_with_AVX2(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Calculate the expf() values first into the vector (to avoid doing it twice)
	yassert(n % 8 == 0);

	// Element-wise expf on vector...
	aussie_vector_expf_AVX2(v, n);  // Element-wise expf...
	float denom = aussie_vector_sum(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
#endif //LINUX
}

//---------------------------------------------------
void aussie_vector_softmax_fused_exp_sum_mult_AVX2(float v[], int n) // Softmax with EXP and SUM and MULT in AVX2
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	yassert(n % 8 == 0);
	float denom = aussie_vector_fused_expf_sum_AVX2(v, n);  // Element-wise expf...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	aussie_vector_multiply_scalar_AVX2(v, n, recip);
#endif //LINUX
}


//---------------------------------------------------
void aussie_vector_softmax_fused_exponentiate_sum_AVX2(float v[], int n) // Softmax with both EXP and SUM in AVX2
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Calculate the expf() values first into the vector (to avoid doing it twice)
	yassert(n % 8 == 0);

	// Element-wise expf on vector...
	float denom = aussie_vector_fused_expf_sum_AVX2(v, n);  // Element-wise expf...
	//float denom = aussie_vector_sum_AVX2(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
#endif //LINUX
}
//---------------------------------------------------
void aussie_vector_softmax_exponentiate_and_sum_AVX2(float v[], int n) // Softmax with both EXP and SUM in AVX2
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Calculate the expf() values first into the vector (to avoid doing it twice)
	yassert(n % 8 == 0);

	// Element-wise expf on vector...
	aussie_vector_expf_AVX2(v, n);  // Element-wise expf...
	float denom = aussie_vector_sum_AVX2(v, n);  // Denominator...
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
#endif //LINUX
}
//---------------------------------------------------
void aussie_vector_softmax_exponentiate_and_sum(float v[], int n)
{
	// Calculate the expf() values first into the vector (to avoid doing it twice)

	// Element-wise expf on vector...
	float denom = aussie_vector_expf_and_sum(v, n);  // Exponentiate & sum
	if (denom == 0.0) {
		yassert(denom != 0.0);
		return;  // fail (should not occur)
	}
	float recip = 1.0f / denom;
	for (int i = 0; i < n; i++) {
		v[i] *= recip;  // NOTE: v[i] is already expf'd
	}
}

//---------------------------------------------------
//---------------------------------------------------


void aussie_softmax_unit_tests()
{

	int n = 16;
	float v1[1000];
	float v2[1000];
	float f = 0.0f;

	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);

	aussie_vector_softmax_basic(v1, n);
	ytestf(aussie_vector_sum(v1, n), 1.0);  // Should add up to 1 after softmax

	aussie_vector_softmax_basic(v2, n);
	ytestf(aussie_vector_sum(v2, n), 1.0); // Should add up to 1 after softmax

	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);
	aussie_vector_softmax_multiply_reciprocal(v2, n);
	ytestf(aussie_vector_sum(v2, n), 1.0); // Should add up to 1 after softmax

	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);
	aussie_vector_softmax_exponentiate_first(v2, n);
	ytestf(aussie_vector_sum(v2, n), 1.0); // Should add up to 1 after softmax
	
#if !LINUX
	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);
	aussie_vector_softmax_exponentiate_with_AVX1(v2, n);
	ytestf(aussie_vector_sum(v2, n), 1.0); // Should add up to 1 after softmax
	
	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);
	aussie_vector_softmax_exponentiate_with_AVX2(v2, n);
	ytestf(aussie_vector_sum(v2, n), 1.0); // Should add up to 1 after softmax
#endif //LINUX

	
	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);
	aussie_vector_softmax_exponentiate_and_sum(v2, n);
	ytestf(aussie_vector_sum(v2, n), 1.0); // Should add up to 1 after softmax


	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);
	aussie_vector_softmax_multiply_reciprocal(v2, n);
	ytestf(aussie_vector_sum(v2, n), 1.0); // Should add up to 1 after softmax

	// Try all zeros...
	aussie_vector_clear(v1, n);
	aussie_vector_clear(v2, n);
	f = aussie_vector_sum_of_exponentials(v1, n);
	ytestfapprox(f, (float)n, 0.0001);
	aussie_vector_softmax_multiply_reciprocal(v2, n);
	ytestfapprox(aussie_vector_sum(v2, n), 1.0, 0.00001); // Should add up to 1 after softmax

}
//---------------------------------------------------
//---------------------------------------------------

