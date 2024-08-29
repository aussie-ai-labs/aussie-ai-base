// anormalize.h -- Normalizations (e.g. LayerNorm/BatchNorm) -- Aussie AI Base Library  
// Created Nov 12th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YNORMALIZE_INCLUDE_HEADER_H
#define AUSSIE_YNORMALIZE_INCLUDE_HEADER_H



//-------------------------------------------------------------------------
// Normalizations of vectors (LayerNorm/BatchNorm)
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Min-max normalization (basic scaled 0..1 normalization)
//-------------------------------------------------------------------------
void aussie_vector_normalize_min_max_basic(float v[], int n);   // Scale min..max to 0..1
void aussie_vector_normalize_min_max_reciprocal(float v[], int n);   // Scale min..max to 0..1
void aussie_vector_normalize_min_max_pointer_arith(float v[], int n);   // Scale min..max to 0..1
void aussie_vector_normalize_min_max_fusion(float v[], int n);   // Scale min..max to 0..1


//-------------------------------------------------------------------------
// Z-score normalization
//-------------------------------------------------------------------------

void aussie_vector_normalize_zscore(float v[], int n);   // Change to z-scores from normal distribution
void aussie_vector_normalize_zscore_fix_mean(float v[], int n);
void aussie_vector_normalize_zscore_reciprocal(float v[], int n);
void aussie_vector_normalize_zscore_fused(float v[], int n);
void aussie_vector_normalize_zscore_sum_AVX1(float v[], int n);  // Use AVX1 for the sum only
void aussie_vector_normalize_zscore_sum_mult_AVX1(float v[], int n);  // Use AVX1 for the sum and multiply-by-reciprocal

void aussie_vector_normalize_zscore_sum_AVX2(float v[], int n);  // Use AVX1 for the sum only
void aussie_vector_normalize_zscore_sum_mult_AVX2(float v[], int n);  // Use AVX1 for the sum and multiply-by-reciprocal

void aussie_vector_normalize_zscore_all_AVX1(float v[], int n);  // Use AVX1 for All: sum, diff-squares & multiply-by-reciprocal
void aussie_vector_normalize_zscore_all_AVX2(float v[], int n);  // Use AVX2 for All: sum, diff-squares & multiply-by-reciprocal

//-------------------------------------------------------------------------
// RMSNorm...
//-------------------------------------------------------------------------
void aussie_vector_rms_normalize_basic(float v[], int n);	// Basic RMS normalization (RMSNorm)
void aussie_vector_rms_normalize_reciprocal(float v[], int n);  // Basic RMS normalization (RMSNorm)
void aussie_vector_rms_normalize_AVX1(float v[], int n);	// RMS normalization (RMSNorm)
void aussie_vector_rms_normalize_AVX2(float v[], int n);	// RMS normalization (RMSNorm)


//-------------------------------------------------------------------------
// BatchNorm 
//-------------------------------------------------------------------------

void aussie_vector_batch_normalize_basic(    // Basic normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);


void aussie_vector_batch_normalize_with_loop_fission(    // Modified slightly improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);
void aussie_vector_batch_normalize_with_loop_fission2(    // Fission-improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);
void aussie_vector_batch_normalize_with_loop_fission2_AVX1(    // Fission-improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);
void aussie_vector_batch_normalize_with_loop_fission2_AVX2(    // Fission-improved normalization (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);

void aussie_vector_batch_normalize_with_loop_fusion_fission(    // Fusion & fission (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);
void aussie_vector_batch_normalize_NO_PARAMS(    // Fusion & fission of loops (BatchNorm)
	float v[], int n,
	float epsilon //, // Smoothing term -- usually 1^e-5 (0.00005)
	//float lambda, // Scaling term hyper-parameter (multiplication)
	//float beta    // Bias/shift term hyper-parameter (addition)
);

void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX1(    // Fusion & fission (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);
void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX2(    // Fusion & fission (BatchNorm)
	float v[], int n,
	float epsilon, // Smoothing term -- usually 1^e-5 (0.00005)
	float lambda, // Scaling term hyper-parameter (multiplication)
	float beta    // Bias/shift term hyper-parameter (addition)
);

float aussie_vector_batchnorm_variance_basic(float v[], int n);   // variance of vector (from mean) (for benchmarking)
void aussie_vector_batchnorm_variance_basic_wrapper(float v[], int n);   // variance of vector (from mean) (for benchmarking)

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------


void aussie_unit_test_normalization();  // Test BatchNorm/LayerNorm/zscore/etc..

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

void aussie_benchmark_normalization();   // Benchmark BatchNorm (LayerNorm?)

// Wrappers for easy benchmarking
void aussie_vector_batch_normalize_basic_wrapper(float v[], int n);
void aussie_vector_batch_normalize_with_loop_fission_wrapper(float v[], int n);
void aussie_vector_batch_normalize_with_loop_fission2_wrapper(float v[], int n);
void aussie_vector_batch_normalize_with_loop_fission2_wrapper_AVX1(float v[], int n);
void aussie_vector_batch_normalize_with_loop_fission2_wrapper_AVX2(float v[], int n);
void aussie_vector_batch_normalize_with_loop_fusion_fission_wrapper(float v[], int n);
void aussie_vector_batch_normalize_NO_PARAMS_wrapper(float v[], int n);
void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX1_wrapper(float v[], int n);
void aussie_vector_batch_normalize_with_loop_fusion_fission_AVX2_wrapper(float v[], int n);

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------



#endif //AUSSIE_YNORMALIZE_INCLUDE_HEADER_H

