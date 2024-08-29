//---------------------------------------------------
// aavx.h -- AVX/AVX-2/AVX-512 SSE intrinsics code -- Aussie AI Base Library  
// Created Nov 13th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd
//---------------------------------------------------

#ifndef AUSSIE_YAVX_INCLUDE_HEADER_H
#define AUSSIE_YAVX_INCLUDE_HEADER_H

#define AUSSIE_DO_AVX512 0

//---------------------------------------------------
//---------------------------------------------------

void aussie_avx_multiply_4_floats_aligned(float v1[4], float v2[4], float vresult[4]);
void aussie_avx2_multiply_8_floats(float v1[8], float v2[8], float vresult[8]);
void aussie_avx_multiply_4_floats(float v1[4], float v2[4], float vresult[4]);

//---------------------------------------------------
//---------------------------------------------------


void aussie_unit_test_avx();  // AVX, AVX-2, AVX-512, SSE
void aussie_unit_test_avx1_basics();  // AVX version 1 (128-bit) tests __m128
void aussie_test_avx_multiply_4_floats();
void aussie_test_avx_multiply_4_floats_with_alignment();

// AVX-2
void aussie_test_aussie_avx2_multiply_8_floats();
void aussie_test_avx2_multiply_8_floats();

// AVX-512
void aussie_avx512_multiply_16_floats(float v1[16], float v2[16], float vresult[16]);
void aussie_test_avx512_multiply_16_floats();

// Vector dot product
float aussie_avx_vecdot_4_floats(float v1[4], float v2[4]);
void aussie_test_avx_vecdot_4_floats();  // Test AVX1 dot product of 4 floats
float aussie_avx2_vecdot_8_floats_buggy(float v1[8], float v2[8]); // AVX-2 (256-bit) dot product
float aussie_avx_vecdot_fma_4_floats(float v1[4], float v2[4]);  // AVX1 vecdot using FMA (Fused Multiply-Add) primitives
float aussie_vecdot_FMA_unroll_AVX1(const float v1[], const float v2[], int n);   // AVX1 vecdot using FMA (Fused Multiply-Add) primitives
float aussie_vecdot_FMA_unroll_AVX2(const float v1[], const float v2[], int n);   // AVX2 vecdot using FMA (Fused Multiply-Add) primitives


void aussie_vector_multiply_scalar_AVX1(float v[], int n, float c);  // Multiply all vector elements by constant
void aussie_vector_multiply_scalar_AVX2(float v[], int n, float c);  // Multiply all vector elements by constant
void aussie_vector_multiply_scalar_AVX2_pointer_arith(float v[], int n, float c);  // Multiply all vector elements by constant

void aussie_vector_add_scalar_AVX1(float v[], int n, float c);   // Add scalar constant to all vector elements
void aussie_vector_add_scalar_AVX2(float v[], int n, float c);   // Add scalar constant to all vector elements

void aussie_vector_expf_AVX1(float v[], int n);   // Apply EXPF (exponential) to each element
void aussie_vector_expf_AVX2(float v[], int n);   // Apply EXPF (exponential) to each element

float aussie_vector_sum_AVX1(float v[], int n);   // Summation (horizontal) of a single vector
float aussie_vector_sum_AVX2(float v[], int n);   // Summation (horizontal) of a single vector

float aussie_vector_sum_squares_AVX1(float v[], int n);  // Summation of squares of all elements
float aussie_vector_sum_squares_AVX2(float v[], int n);  // Summation of squares of all elements

float aussie_vector_sum_diff_squared_fused_AVX1(float v[], int n, float meanval);
float aussie_vector_sum_diff_squared_fused_AVX2(float v[], int n, float meanval);


//---------------------------------------------------
//---------------------------------------------------

float aussie_vector_min(float v[], int n);  // Mininum
float aussie_vector_min_AVX1(float v[], int n);   // Minimum (horizontal) of a single vector
float aussie_vector_min_AVX2(float v[], int n);   // Minimum (horizontal) of a single vector
float aussie_vector_min_AVX1b(float v[], int n);   // Minimum (horizontal) of a single vector
float aussie_vector_min_AVX2b(float v[], int n);   // Minimum (horizontal) of a single vector

float aussie_vector_max(float v[], int n);  // Maximum
float aussie_vector_max_AVX1(float v[], int n);   // Maximum (horizontal) of a single vector
float aussie_vector_max_AVX1b(float v[], int n);   // Maximum (horizontal) of a single vector
float aussie_vector_max_AVX2(float v[], int n);   // Maximum (horizontal) of a single vector

//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_reluize_AVX1(float v[], int n);   // Apply RELU to each element (sets negatives to zero)
void aussie_vector_reluize_AVX2(float v[], int n);   // Apply RELU to each element (sets negatives to zero)

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------


#endif //AUSSIE_YAVX_INCLUDE_HEADER_H

