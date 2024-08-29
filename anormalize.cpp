// ynormalize.cpp -- Normalizations (e.g. LayerNorm/BatchNorm) -- Aussie AI Base Library  
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

#include "anormalize.h"  // self-include

//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

// Normalize 
// -- DONE z-score (normal distribution)
// -- minmax normalize
// -- layerNorm
// -- batchNorm

void aussie_vector_normalize_zscore_fused(float v[], int n)
{
	// Fused version which stores "v[i]-mean" in the array...
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev_fused(v, n, vmean);
	float frecip = 1.0f / stddev;  // Before the loop
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - vmean) * frecip; // Multiply by reciprocal
	}
}

void aussie_vector_normalize_zscore_sum_AVX1(float v[], int n)  // Use AVX1 for the sum only
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Fused version which stores "v[i]-mean" in the array...
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev_fused_AVX1(v, n, vmean);
	float frecip = 1.0f / stddev;  // Before the loop
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - vmean) * frecip; // Multiply by reciprocal
	}
#endif //LINUX
}

void aussie_vector_normalize_zscore_sum_AVX2(float v[], int n)  // Use AVX1 for the sum only
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Fused version which stores "v[i]-mean" in the array...
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev_fused_AVX2(v, n, vmean);
	float frecip = 1.0f / stddev;  // Before the loop
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - vmean) * frecip; // Multiply by reciprocal
	}
#endif //LINUX
}

void aussie_vector_normalize_zscore_sum_mult_AVX1(float v[], int n)  // Use AVX1 for the sum and multiply-by-reciprocal
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Fused version which stores "v[i]-mean" in the array...
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev_fused_AVX1(v, n, vmean);
	float frecip = 1.0f / stddev;  // Before the loop
	aussie_vector_multiply_scalar_AVX1(v, n, frecip);  // Multiply by reciprocal
#endif //LINUX
}

void aussie_vector_normalize_zscore_all_AVX1(float v[], int n)  // Use AVX1 for the sum and multiply-by-reciprocal
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Fused version which stores "v[i]-mean" in the array...
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev_all_AVX1(v, n, vmean);
	float frecip = 1.0f / stddev;  // Before the loop
	aussie_vector_multiply_scalar_AVX1(v, n, frecip);  // Multiply by reciprocal
#endif //LINUX
}

void aussie_vector_normalize_zscore_all_AVX2(float v[], int n)  // Use AVX1 for the sum and multiply-by-reciprocal
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Fused version which stores "v[i]-mean" in the array...
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev_all_AVX2(v, n, vmean);
	float frecip = 1.0f / stddev;  // Before the loop
	aussie_vector_multiply_scalar_AVX2(v, n, frecip);  // Multiply by reciprocal
#endif //LINUX
}

void aussie_vector_normalize_zscore_sum_mult_AVX2(float v[], int n)  // Use AVX1 for the sum and multiply-by-reciprocal
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Fused version which stores "v[i]-mean" in the array...
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev_fused_AVX2(v, n, vmean);
	float frecip = 1.0f / stddev;  // Before the loop
	aussie_vector_multiply_scalar_AVX2(v, n, frecip);  // Multiply by reciprocal
#endif //LINUX
}


void aussie_vector_normalize_zscore_reciprocal(float v[], int n)
{
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev(v, n, vmean);
	float frecip = 1.0f / stddev; 
	for (int i = 0; i < n; i++) {
		// v[i] = (v[i] - vmean) * frecip; // Multiply by reciprocal
		v[i] *= frecip; // No subtraction
	}
}

void aussie_vector_normalize_zscore_fix_mean(float v[], int n)
{
	float vmean = 0.0f;
	float stddev = aussie_vector_mean_and_stddev(v, n, vmean);
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - vmean) / stddev; // z-score = How many standard deviations away from the mean?
	}
}

void aussie_vector_normalize_zscore(float v[], int n)   // Change to z-scores from normal distribution
{
	float vmean = aussie_vector_mean(v, n);
	float stddev = aussie_vector_standard_deviation(v, n);
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - vmean) / stddev; // z-score = How many standard deviations away from the mean?
	}
}

//-------------------------------------------------------------------------
// Min-max normalization (basic scaled 0..1 normalization)
//-------------------------------------------------------------------------
void aussie_vector_normalize_min_max_basic(float v[], int n)   // Scale min..max to 0..1
{
	float fmin = aussie_vector_min(v, n);   // Minimum
	float fmax = aussie_vector_max(v, n);   // Maximum
	float frange = fmax - fmin;
	if (frange == 0.0f) {
		yassert(frange != 0.0f);
		return;  // fail
	}
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - fmin) / frange;
	}
}

void aussie_vector_normalize_min_max_reciprocal(float v[], int n)   // Scale min..max to 0..1
{
	float fmin = aussie_vector_min(v, n);   // Minimum
	float fmax = aussie_vector_max(v, n);   // Maximum
	float frange = fmax - fmin;
	if (frange == 0.0f) {
		yassert(frange != 0.0f);
		return;  // fail
	}
	float frecip = 1.0f / frange;
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - fmin) * frecip;  // Reciprocal multiplication
	}
}

void aussie_vector_normalize_min_max_pointer_arith(float v[], int n)   // Scale min..max to 0..1
{
	float fmin = aussie_vector_min(v, n);   // Minimum
	float fmax = aussie_vector_max(v, n);   // Maximum
	float frange = fmax - fmin;
	if (frange == 0.0f) {
		yassert(frange != 0.0f);
		return;  // fail
	}
	float frecip = 1.0f / frange;
	float* vend = &v[n];
	for (; v != vend; v++) {
		*v = (*v - fmin) * frecip;  // Reciprocal multiplication
	}
}


void aussie_vector_normalize_min_max_fusion(float v[], int n)   // Scale min..max to 0..1
{
	float fmax = 0.0f;   // Maximum
	float fmin = aussie_vector_min_max_fused(v, n, fmax);
	float frange = fmax - fmin;
	if (frange == 0.0f) {
		yassert(frange != 0.0f);
		return;  // fail
	}
	float frecip = 1.0f / frange;
	float* vend = &v[n];
	for (; v != vend; v++) {
		*v = (*v - fmin) * frecip;  // Reciprocal multiplication
	}
}

//-------------------------------------------------------------------------
// BATCHNORM
//-------------------------------------------------------------------------

void aussie_vector_batch_normalize_basic_wrapper(float v[], int n)
{
	aussie_vector_batch_normalize_basic(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
}

void aussie_vector_batch_normalize_with_loop_fission_wrapper(float v[], int n)
{
	aussie_vector_batch_normalize_with_loop_fission(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
}

void aussie_vector_batch_normalize_with_loop_fission2_wrapper(float v[], int n)
{
	aussie_vector_batch_normalize_with_loop_fission2(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
}

void aussie_vector_batch_normalize_with_loop_fission2_wrapper_AVX1(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	aussie_vector_batch_normalize_with_loop_fission2_AVX1(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
#endif //LINUX
}

void aussie_vector_batch_normalize_with_loop_fission2_wrapper_AVX2(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	aussie_vector_batch_normalize_with_loop_fission2_AVX2(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
#endif //LINUX
}

void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX1_wrapper(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	aussie_vector_batch_normalize_with_loop_fusion_fission_AVX1(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
#endif //LINUX
}

void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX2_wrapper(float v[], int n)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	aussie_vector_batch_normalize_with_loop_fusion_fission_AVX2(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
#endif //LINUX
}

void aussie_vector_batch_normalize_with_loop_fusion_fission_wrapper(float v[], int n)
{
	aussie_vector_batch_normalize_with_loop_fusion_fission(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
}

void aussie_vector_batch_normalize_NO_PARAMS_wrapper(float v[], int n)
{
	aussie_vector_batch_normalize_NO_PARAMS(    // Basic normalization (BatchNorm)
		v, n,
		0.00005f //, // epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
		//1.0f, // lambda, // Scaling term hyper-parameter (multiplication)
		//0.0f // beta    // Bias/shift term hyper-parameter (addition)
	);
}


void aussie_vector_batch_normalize_basic(    // Basic normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
	// NOTE: epsilon smoothing term is usually 1^e-5
	float fmean = aussie_vector_mean(v, n);
	float variance = aussie_vector_variance_of_mean(v, n, fmean);

	float denom = sqrtf(variance + epsilon);  // This is like std. deviation, but adjusted/smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	for (int i = 0; i < n; i++) {
		v[i] = (v[i] - fmean) / denom; // Normalize all elements to re-center and scale
	}

	aussie_vector_multiply_scalar(v, n, lambda);  // Scale all values by lambda hyper-param
	aussie_vector_add_scalar(v, n, beta);  // Add beta hyper-param to all values 

}

float aussie_vector_batchnorm_variance_basic(   // Just compute variance of vector (from mean) (for benchmarking)
	float v[], int n )
{
	// NOTE: epsilon smoothing term is usually 1^e-5
	float fmean = aussie_vector_mean(v, n);
	float variance = aussie_vector_variance_of_mean(v, n, fmean);
	return variance;
}

void aussie_vector_batchnorm_variance_basic_wrapper(float v[], int n)   //wrap for benchmarking
{
	(void)aussie_vector_batchnorm_variance_basic(v, n);  // Throw away return value
}

void aussie_vector_batch_normalize_with_loop_fission(    // Modified slightly improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
	// NOTE: epsilon smoothing term is usually 1^e-5
	float fmean = aussie_vector_mean(v, n);
	float variance = aussie_vector_variance_of_mean(v, n, fmean);


	float negmean = -fmean;  // Use negative for addition not subtraction
	aussie_vector_add_scalar(v, n, negmean);     // Re-center using mean
	float denom = sqrtf(variance + epsilon);  // This is like std. deviation, but adjusted/smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float recip = 1.0f / denom;  // Use reciprocal multiply not division
	aussie_vector_multiply_scalar(v, n, recip);  // Re-scale
	aussie_vector_multiply_scalar(v, n, lambda);  // Scale all values by lambda hyper-param
	aussie_vector_add_scalar(v, n, beta);  // Add beta hyper-param to all values 

}

void aussie_vector_batch_normalize_with_loop_fission2_AVX1(    // Fission-improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else

	// NOTE: epsilon smoothing term is usually 1^e-5
	float mean = aussie_vector_mean_AVX1(v, n);
	float variance = aussie_vector_variance_of_mean(v, n, mean);

	aussie_vector_add_scalar_AVX1(v, n, -mean);  // Subtract the mean
	float denom = sqrtf(variance + epsilon);  // like std. deviation, but smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float scalef = lambda / denom;  // Combined scale factor
	aussie_vector_multiply_scalar_AVX1(v, n, scalef);  // Scale by both denom and lambda 
	aussie_vector_add_scalar_AVX1(v, n, beta);  // Add beta hyper-param to all values 
#endif //LINUX
}


void aussie_vector_batch_normalize_with_loop_fission2_AVX2(    // Fission-improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// NOTE: epsilon smoothing term is usually 1^e-5
	float mean = aussie_vector_mean_AVX2(v, n);
	float variance = aussie_vector_variance_of_mean(v, n, mean);

	aussie_vector_add_scalar_AVX2(v, n, -mean);  // Subtract the mean
	float denom = sqrtf(variance + epsilon);  // like std. deviation, but smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float scalef = lambda / denom;  // Combined scale factor
	aussie_vector_multiply_scalar_AVX2(v, n, scalef);  // Scale by both denom and lambda 
	aussie_vector_add_scalar_AVX2(v, n, beta);  // Add beta hyper-param to all values 
#endif //LINUX
}

void aussie_vector_batch_normalize_with_loop_fission2(    // Fission-improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
	// NOTE: epsilon smoothing term is usually 1^e-5
	float mean = aussie_vector_mean(v, n);
	float variance = aussie_vector_variance_of_mean(v, n, mean);


	aussie_vector_add_scalar(v, n, -mean);  // Subtract the mean
	float denom = sqrtf(variance + epsilon);  // like std. deviation, but smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float scalef = lambda / denom;  // Combined scale factor
	aussie_vector_multiply_scalar(v, n, scalef);  // Scale by both denom and lambda 
	aussie_vector_add_scalar(v, n, beta);  // Add beta hyper-param to all values 

}

void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX1(    // Fusion & fission (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else

	// NOTE: epsilon smoothing term is usually 1^e-5
	float mean = aussie_vector_mean_AVX1(v, n);
	float variance = aussie_vector_variance_of_mean_fused_AVX1(v, n, mean);  // FUSION: leaves DIFF from MEAN in vector...

	// NOT NEEDED! ... aussie_vector_add_scalar(v, n, -mean);  // Subtract the mean
	float denom = sqrtf(variance + epsilon);  // like std. deviation, but smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float scalef = lambda / denom;  // Combined scale factor
	aussie_vector_multiply_scalar_AVX1(v, n, scalef);  // Scale by both denom and lambda 
	aussie_vector_add_scalar_AVX1(v, n, beta);  // Add beta hyper-param to all values 

#endif //LINUX
}

void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX2(    // Fusion & fission (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else

	// NOTE: epsilon smoothing term is usually 1^e-5
	float mean = aussie_vector_mean_AVX2(v, n);
	float variance = aussie_vector_variance_of_mean_fused_AVX2(v, n, mean);  // FUSION: leaves DIFF from MEAN in vector...

	// NOT NEEDED! ... aussie_vector_add_scalar(v, n, -mean);  // Subtract the mean
	float denom = sqrtf(variance + epsilon);  // like std. deviation, but smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float scalef = lambda / denom;  // Combined scale factor
	aussie_vector_multiply_scalar_AVX2(v, n, scalef);  // Scale by both denom and lambda 
	aussie_vector_add_scalar_AVX2(v, n, beta);  // Add beta hyper-param to all values 

#endif //LINUX
}

void aussie_vector_batch_normalize_with_loop_fusion_fission(    // Fusion & fission of loops (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
)
{
	// NOTE: epsilon smoothing term is usually 1^e-5
	float mean = aussie_vector_mean(v, n);
	float variance = aussie_vector_variance_of_mean_fused(v, n, mean);  // FUSION: leaves DIFF from MEAN in vector...

	// NOT NEEDED! ... aussie_vector_add_scalar(v, n, -mean);  // Subtract the mean
	float denom = sqrtf(variance + epsilon);  // like std. deviation, but smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float scalef = lambda / denom;  // Combined scale factor
	aussie_vector_multiply_scalar(v, n, scalef);  // Scale by both denom and lambda 
	aussie_vector_add_scalar(v, n, beta);  // Add beta hyper-param to all values 
}


void aussie_vector_batch_normalize_NO_PARAMS(    // Fusion & fission of loops (BatchNorm)
	float v[], int n,
	float epsilon //, // Smoothing term -- usually 1^e-5 (0.00005)
	//float lambda, // Scaling term hyper-parameter (multiplication)
	//float beta    // Bias/shift term hyper-parameter (addition)
)
{
	// NOTE: epsilon smoothing term is usually 1^e-5
	float mean = aussie_vector_mean(v, n);
	float variance = aussie_vector_variance_of_mean_fused(v, n, mean);  // FUSION: leaves DIFF from MEAN in vector...

	// NOT NEEDED! ... aussie_vector_add_scalar(v, n, -mean);  // Subtract the mean
	float denom = sqrtf(variance + epsilon);  // like std. deviation, but smoothed by epsilon
	if (denom == 0.0) { // Avoid divide-by-zero
		yassert(denom != 0.0);  // Should not be zero if epsilon>0.0
		return;  // fail
	}
	float scalef =  1.0f / denom;  // Combined scale factor /*NOT:lambda*/
	aussie_vector_multiply_scalar(v, n, scalef);  // Scale by both denom and lambda 
	// NOT NEEDED: aussie_vector_add_scalar(v, n, beta);  // Add beta hyper-param to all values 

}


void aussie_vector_rms_normalize_basic(float v[], int n)  // Basic RMS normalization (RMSNorm)
{
	// RMS norm re-scales by dividing by the RMS factor (sqrt of the sum-of-squares)
	// .. It does NOT re-center using the mean (so it's faster by not calculating it)
	const float epsilon = 0.00005; // Smoothing term -- usually 1^e-5 (0.00005)
	float sum_squares = aussie_vector_sum_squared(v, n);  // Sum of squares
	float avg_squares = sum_squares / n;  // Average of the squares...
	float denom = sqrtf(avg_squares + epsilon);  // RMS factor
	aussie_vector_divide_scalar(v, n, denom);  // Divide all values by the RMS scale factor
}

void aussie_vector_rms_normalize_reciprocal(float v[], int n)  // Basic RMS normalization (RMSNorm)
{
	const float epsilon = 0.00005; // Smoothing term -- usually 1^e-5 (0.00005)
	float sum_squares = aussie_vector_sum_squared(v, n);  // Sum of squares
	float avg_squares = sum_squares / n;  // Average of the squares...
	float fmult = 1.0f / sqrtf(avg_squares + epsilon);  // Reciprocal of factor, so we can multiply
	aussie_vector_multiply_scalar(v, n, fmult);  // Divide all values by the RMS scale factor
}

void aussie_vector_rms_normalize_AVX1(float v[], int n)	// RMS normalization (RMSNorm)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	const float epsilon = 0.00005; // Smoothing term -- usually 1^e-5 (0.00005)
	float sum_squares = aussie_vector_sum_squares_AVX1(v, n);  // Sum of squares
	float avg_squares = sum_squares / n;  // Average of the squares...
	float fmult = 1.0f / sqrtf(avg_squares + epsilon);  // Reciprocal of factor, so we can multiply
	aussie_vector_multiply_scalar_AVX1(v, n, fmult);  // Divide all values by the RMS scale factor
#endif //LINUX
}

void aussie_vector_rms_normalize_AVX2(float v[], int n)	// RMS normalization (RMSNorm)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	const float epsilon = 0.00005; // Smoothing term -- usually 1^e-5 (0.00005)
	float sum_squares = aussie_vector_sum_squares_AVX2(v, n);  // Sum of squares
	float avg_squares = sum_squares / n;  // Average of the squares...
	float fmult = 1.0f / sqrtf(avg_squares + epsilon);  // Reciprocal of factor, so we can multiply
	aussie_vector_multiply_scalar_AVX2(v, n, fmult);  // Divide all values by the RMS scale factor
#endif //LINUX
}


//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

void aussie_unit_test_normalization()  // Test BatchNorm/LayerNorm/zscore/etc..
{

}

//---------------------------------------------------
//---------------------------------------------------

