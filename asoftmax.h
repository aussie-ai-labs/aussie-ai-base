// asoftmax.h -- The Softmax function -- Aussie AI Base Library  
// Created Nov 12th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YSOFTMAX_INCLUDE_HEADER_H
#define AUSSIE_YSOFTMAX_INCLUDE_HEADER_H


#endif //AUSSIE_YSOFTMAX_INCLUDE_HEADER_H

//---------------------------------------------------
//---------------------------------------------------


// Softmax vector computations
float aussie_vector_sum_of_exponentials(float v[], int n);
void aussie_vector_softmax_basic(float v[], int n);
void aussie_vector_softmax_multiply_reciprocal(float v[], int n);
void aussie_vector_softmax_exponentiate_first(float v[], int n);
void aussie_vector_softmax_exponentiate_and_sum(float v[], int n);

// Softmax AVX1
void aussie_vector_softmax_exponentiate_with_AVX1(float v[], int n);
void aussie_vector_softmax_fused_exponentiate_sum_AVX1(float v[], int n);
void aussie_vector_softmax_exponentiate_and_sum_AVX1(float v[], int n);
float aussie_vector_fused_expf_sum_AVX1(float v[], int n);   // Apply EXPF (exponential) to each element and SUM them
void aussie_vector_softmax_fused_exp_sum_mult_AVX1(float v[], int n);



// Softmax AVX2
void aussie_vector_softmax_exponentiate_with_AVX2(float v[], int n);
void aussie_vector_softmax_exponentiate_and_sum_AVX2(float v[], int n);
void aussie_vector_softmax_fused_exponentiate_sum_AVX2(float v[], int n);
float aussie_vector_fused_expf_sum_AVX2(float v[], int n);   // Apply EXPF (exponential) to each element and SUM them
void aussie_vector_softmax_fused_exp_sum_mult_AVX2(float v[], int n);


void aussie_softmax_unit_tests();
void aussie_benchmark_softmax();


//---------------------------------------------------
//---------------------------------------------------



