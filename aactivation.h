// aactivation.h -- Activation Functions -- Aussie AI Base Library  
// Created Oct 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YACTIVATION_INCLUDE_HEADER_H
#define AUSSIE_YACTIVATION_INCLUDE_HEADER_H

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

#define AUSSIE_PI 3.1415926535f

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

// RELU activation function
float aussie_RELU_basic(float f);   // Basic RELU (inefficient)
#define AUSSIE_RELU_MACRO(f)  ( (f) <= 0.0f ? 0.0f : (f) )

void aussie_vector_reluize(float v[], int n);   // Apply RELU to each element (sets negatives to zero)
void aussie_vector_reluize_AVX1(float v[], int n);   // Apply RELU to each element (sets negatives to zero)
void aussie_vector_reluize_AVX2(float v[], int n);   // Apply RELU to each element (sets negatives to zero)

#define RELU(x) aussie_RELU_basic(x)
#define RELUIZE1(x) ( (x) = AUSSIE_RELU_MACRO(x))       // Slower version
#define RELUIZE2(x) if ( (x) < 0.0f) { (x) = 0.0f; }    // If-then version
#define RELUIZE3(x) ( (x) < 0.0f && ( (x) = 0.0f) )     // Short-circuited operator version

//-------------------------------------------------------------------------
// ELU activaton function
//-------------------------------------------------------------------------
// ... x  if x > 0 .0
// ... alpha * ( exp(x) - 1) if x <= 0.0
// ... alpha is a hyper-parameter controlling the version of ELU
//-------------------------------------------------------------------------

float aussie_ELU_basic(float x, float alpha_hyperparam);   // Basic ELU activation (inefficient)


//-------------------------------------------------------------------------
// GELU activation function
//-------------------------------------------------------------------------

// GELU paper: https://arxiv.org/abs/1606.08415 -- Hendrycks & Gimpel 2016/revised 2023
float aussie_GELU_basic(float x);   // Basic Gaussian GELU (inefficient)
float aussie_GELU_basic2(float x);   // Basic Gaussian GELU (still inefficient)
// GELU paper approx #1 = 0.5 * x * ( 1 + tanh ( sqrt(2/PI) * (x + 0.44715 * x^3)  ) ) 
float aussie_GELU_approx1(float f);   // Approximated Gaussian GELU
float aussie_GELU_approx1_optimized(float f);   // Approximated Gaussian GELU (with minor optimizations)
float aussie_GELU_approx1_optimized2(float f);   // Approximated Gaussian GELU (with 2nd minor optimizations)

float aussie_sigmoid(float f);

float aussie_GELU_approx2(float f);   // Approximated Gaussian GELU
float aussie_GELU_approx2b(float x);   // Approximated Gaussian GELU

float gelu_fast_FP32(float x);   // Table lookup GELU
void aussie_GELU_setup_table_FP32(); // Initialize GELU precomputed table

void aussie_GELU_setup_table_FP16(); // Initialize GELU precomputed table for FP16

// GELU precomputation in 24-bit lookup table...
void aussie_GELU_setup_table_FP32_24bits_PRINT_SOURCE(char* nickname, char *outfname); // Initialize 24-bits GELU precomputed table
void aussie_GELU_setup_table_FP32_24bits(); // Initialize 24-bits GELU precomputed table
float gelu_fast_FP32_24bits(float f);    // Table lookup GELU (using 24 bits)

//-------------------------------------------------------------------------
// SiLU activation function
//-------------------------------------------------------------------------

float aussie_SiLU_basic(float x);   // Basic SiLU (inefficient)

//-------------------------------------------------------------------------

void aussie_precompute_tests();  // Test precompute of activations example

//-------------------------------------------------------------------------
// TODO: Mish activation
// TODO: Sigmoid (basic)
// TODO: Step function (basic)
//-------------------------------------------------------------------------

float aussie_step_basic(float f);

#define  AUSSIE_STEP_FUNCTION(f)  ( (f) < 0.0f ? 0.0f : 1.0f )

#define AUSSIE_SIGN_FUNCTION(x)  ( ( (x) > 0.0f ? 1.0f : ( (x) < 0.0f ? -1.0f : 0.0f )  ) )

#define AUSSIE_SIGNBIT_FP32(f)  (( *(unsigned*)&(f)) >> 31u )  // Sign bit
#define  AUSSIE_STEP_FUNCTION2(f)  ( AUSSIE_SIGNBIT_FP32(f) ? 0.0f : 1.0f )
#define  AUSSIE_STEP_FUNCTION3(f)  ((float)!( AUSSIE_SIGNBIT_FP32(f)))

#define AUSSIE_ISNEGATIVE_FAST_FP32(f)  (( *(unsigned*)&(f)) >> 31u )  // Sign bit
#if 0 // Old versions
#define  AUSSIE_STEP_FUNCTION2(f)  ( AUSSIE_ISNEGATIVE_FAST_FP32(f) ? 0.0f : 1.0f )
#define  AUSSIE_STEP_FUNCTION3(f)  (float)!( AUSSIE_ISNEGATIVE_FAST_FP32(f))
#endif

//-------------------------------------------------------------------------
// SOFTPLUS = log(1+e^x)
// ... scaled softplus = 1/beta * log(1 + exp(beta * x)
//-------------------------------------------------------------------------


//-------------------------------------------------------------------------
// Unit testing for Activations...
//-------------------------------------------------------------------------

void aussie_activation_unit_tests();
void aussie_test_GELU(float f);

//-------------------------------------------------------------------------

#endif //AUSSIE_YACTIVATION_INCLUDE_HEADER_H

