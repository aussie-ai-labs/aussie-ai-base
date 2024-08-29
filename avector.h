// avector.h --  Vector computations (various) -- Aussie AI Base Library
// Created 6th October 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef YVECTOR_INCLUDE_HEADER_H
#define YVECTOR_INCLUDE_HEADER_H

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

// Simple vector dot products...

// Optimized vector dot product
float aussie_vecdot_pointer_arithmetic(float v1[], float v2[], int n);   // Pointer arithmetic vector dot product

// Loop unrolling -- vector dot product
float aussie_vecdot_unroll4_basic(float v1[], float v2[], int n);  // Loop-unrolled x 4 -- Vector dot product 
float aussie_vecdot_unroll4_better(float v1[], float v2[], int n);  // Loop-unrolled x 4 -- Vector dot product
float aussie_vecdot_unroll_AVX2(const float v1[], const float v2[], int n);  // AVX-2 loop-unrolled (8 floats, 256-bits) Vector dot product 
float aussie_vecdot_unroll_AVX1(const float v1[], const float v2[], int n);  // AVX-1 loop-unrolled (4 floats, 128-bit) Vector dot product 
float aussie_vecdot_FMA_unroll_AVX1(const float v1[], const float v2[], int n);   // AVX1 vecdot using FMA (Fused Multiply-Add) primitives
float aussie_vecdot_FMA_unroll_AVX2(const float v1[], const float v2[], int n);   // AVX2 vecdot using FMA (Fused Multiply-Add) primitives

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
void aussie_vector_do_sqrt(float v[], int n);
void aussie_vector_do_sqrt_loop_splitting(float v[], int n);

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

// Unit testing wrapper
void aussie_yvector_unit_tests();
void aussie_yvector_test_dot_products(float v1[], float v2[], int n, float expected);
void aussie_yvector_test_dot_products_BIG(float v1[], float v2[], int n, float expected);

//-------------------------------------------------------------------------

// Basic vector set APIs
void aussie_vector_zero(float v1[], int n);   // Clear vector to all zeros
void aussie_vector_set_1_N(float v1[], int n);   // Set vector values to 1..N 
void aussie_vector_set_1_N_reverse(float v1[], int n);   // Set vector values to N..1

void aussie_vector_copy_basic(float vdest[], float vsrc[], int n);

//-------------------------------------------------------------------------
// Reversed dot products (loop reversal)
//-------------------------------------------------------------------------

float aussie_vecdot_reverse_basic(float v1[], float v2[], int n);   // REVERSED basic vector dot product
float aussie_vecdot_reverse_zerotest(float v1[], float v2[], int n);   // Reversed-with-zero-test vector dot product
float aussie_vecdot_reverse_basic2(float v1[], float v2[], int n);   // REVERSED basic vector dot product #2
float aussie_vecdot_unroll4_basic(float v1[], float v2[], int n);  // Loop-unrolled Vector dot product 
float aussie_vecdot_unroll4_better(float v1[], float v2[], int n); // Loop-unrolled Vector dot product 
float aussie_vecdot_unroll4_duffs_device(float v1[], float v2[], int n); // Loop-unrolled Vector dot product 


//-------------------------------------------------------------------------
// Loop perforation (stochastic/probabilistic iteration skipping)
//-------------------------------------------------------------------------

float aussie_vecdot_perforated_slow(float v1[], float v2[], int n, int percent_perforation);   // Loop perforation -- vector dot product

float aussie_vecdot_basic(const float v1[], const float v2[], int n);   // Basic vector dot product

int aussie_vecdot_integer_fixed_point(int v1[], int v2[], int n);   // Integer vector dot product
int aussie_vecdot_integer_bitshift(int v1[], int v2[], int n);   // Integer vector dot product



float aussie_vecdot_section512(float v1[], float v2[]);  // Simulate parallel 512 section 
float aussie_vecdot_parallel_basic(float v1[], float v2[], int n);   // Simulated parallel vector dot product
float aussie_vecdot_parallel_odd_sizes(float v1[], float v2[], int n);   // Simulated parallel vector dot product
float aussie_vecdot_parallel_padding(float v1[], float v2[], int n);   // Padding used for extra leftover array...
float aussie_vecdot_pointer_arithmetic(float v1[], float v2[], int n);   // Pointer arithmetic vector dot product
float aussie_vecdot_reverse_basic(float v1[], float v2[], int n);  // REVERSED basic vector dot product
float aussie_vecdot_reverse_basic2(float v1[], float v2[], int n);   // REVERSED basic vector dot product #2
float aussie_vecdot_reverse_zerotest(float v1[], float v2[], int n);   // Reversed-with-zero-test vector dot product
float aussie_vecdot_zero_skipping(const float v1[], const float v2[], int n);   // Zero skipping vector dot product

// INT vecdot...
int aussie_vecdot_int_basic(int v1[], int v2[], int n);   // Basic INT vector dot product

//-------------------------------------------------------------------------
// Element-wise operations on each vector element
//-------------------------------------------------------------------------
void aussie_vector_clear(float v[], int n);  // Clear to zero vector
void aussie_vector_set_constant(float v[], int n, float c); // Set all vector elements to constant
void aussie_vector_set_1_N(float v1[], int n);   // Set vector values to 1..N 
void aussie_vector_set_1_N_MAXN(float v1[], int n, int maxn);   // Set vector values to 1..N, but cycling
void aussie_vector_set_range(float v1[], int n, int start, int end);   // Set vector values to START..END

// int vectors..
void aussie_ivector_set_1_N(int v1[], int n);   // Set vector values to 1..N 

//-------------------------------------------------------------------------
// Statistical measures of a vector
//-------------------------------------------------------------------------
float aussie_vector_variance(float v[], int n);  // Variance (square of std. dev.)
float aussie_vector_variance_of_mean(float v[], int n, float fmean);  // Variance from already-calculated mean (square of std. dev.)
float aussie_vector_standard_deviation(float v[], int n);  // Std. dev (sqrt of variance)
float aussie_vector_sum_diff_squared(float v[], int n, float meanval);
float aussie_vector_mean(float v[], int n);  // Mean (same as average)
float aussie_vector_avg(float v[], int n);  // Average
float aussie_vector_sum_squared(float v[], int n);  // Sum of squared elements...
float aussie_vector_sum_diff_squared_fused(float v[], int n, float meanval); // Fusion version that leaves the DIFF in the vector..
float aussie_vector_sum_diff_squared_fission(float v[], int n, float meanval); // FISSION version that leaves the DIFF in the vector..
float aussie_vector_sum_diff_squared_fissionB(float v[], int n, float meanval);  // FISSION version made concise.. (slower)
float aussie_vector_sum_diff_squared_fission(float v[], int n, float meanval);
float aussie_vector_sum_diff_squared_fission_AVX1(float v[], int n, float meanval);
float aussie_vector_sum_diff_squared_fission_AVX2(float v[], int n, float meanval);

float aussie_vector_sum_squares_basic(float v[], int n); // Summation of squares of all elements
float aussie_vector_sum_squares_AVX1(float v[], int n);  // Summation of squares of all elements
float aussie_vector_sum_squares_AVX2(float v[], int n);  // Summation of squares of all elements

float aussie_vector_variance_of_mean_fused(float v[], int n, float fmean);  // Variance with Fusion (leaves DIFF from MEAN in the vector)
float aussie_vector_variance_of_mean_fused_AVX1(float v[], int n, float fmean);  // Variance with Fusion (leaves DIFF from MEAN in the vector)
float aussie_vector_variance_of_mean_fused_AVX2(float v[], int n, float fmean);  // Variance with Fusion (leaves DIFF from MEAN in the vector)

float aussie_vector_mean_and_variance(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)
float aussie_vector_mean_and_stddev(float v[], int n, float& meanout);  // Std. dev (sqrt of variance)
float aussie_vector_mean_and_stddev_fused(float v[], int n, float& meanout);  // Std. dev (sqrt of variance)
float aussie_vector_mean_and_variance_fused(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)
float aussie_vector_mean_and_variance_fused_AVX1(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)
float aussie_vector_mean_and_variance_fused_AVX2(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)


float aussie_vector_mean_and_variance_fused_AVX1(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)
float aussie_vector_mean_and_variance_all_AVX1(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)

float aussie_vector_mean_AVX1(float v[], int n);  // Mean (same as average)
float aussie_vector_mean_AVX2(float v[], int n);  // Mean (same as average)
float aussie_vector_mean_and_stddev_fused_AVX1(float v[], int n, float& meanout);  // Std. dev (sqrt of variance)
float aussie_vector_mean_and_stddev_fused_AVX2(float v[], int n, float& meanout);  // Std. dev (sqrt of variance)
float aussie_vector_mean_and_variance_fused_AVX2(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)
float aussie_vector_mean_and_stddev_all_AVX1(float v[], int n, float& meanout);  // Std. dev (sqrt of variance)
float aussie_vector_mean_and_stddev_all_AVX2(float v[], int n, float& meanout);  // Std. dev (sqrt of variance)

float aussie_vector_mean_and_variance_all_AVX1(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)
float aussie_vector_mean_and_variance_all_AVX2(float v[], int n, float& fmean_out);  // Variance (square of std. dev.)


//-------------------------------------------------------------------------
// Math computations on a vector 
//-------------------------------------------------------------------------
float aussie_vector_sum(float v[], int n);  // Summation (addition)
float aussie_vector_sum_pointer_arith(float v[], int n);  // Summation
float aussie_vector_sum_AVX1(float v[], int n);   // Summation (horizontal) of a single vector
float aussie_vector_sum_AVX2(float v[], int n);   // Summation (horizontal) of a single vector

float aussie_vector_min_max_fused(float v[], int n, float &fmax);  // Mininum returned, maximum in parameter

float aussie_vector_min(float v[], int n);  // Mininum
float aussie_vector_max(float v[], int n);  // Maximum

float aussie_vector_product(float v[], int n);  // Product (multiply)
float aussie_vector_sum_squares(float v[], int n);  // Summation of squares of all elements
float aussie_vector_distance(float v[], int n);  // Euclidean distance/Magnitude of vector -- Sqrt of Summation of squares of all elements

//-------------------------------------------------------------------------
// Vector comparisons...
//-------------------------------------------------------------------------
bool aussie_vector_is_equal(float v1[], float v2[], int n);   // Test if 2 vectors are identical/equal (all elements)


//-------------------------------------------------------------------------
// Count elements of a vector (by some criteria)
//-------------------------------------------------------------------------
int aussie_vector_count_negatives(float v[], int n);
int aussie_vector_count_zeros(float v[], int n);
int aussie_vector_count_positives(float v[], int n);
int aussie_vector_count_nonzeros(float v[], int n);
int aussie_vector_count_in_range(float v[], int n, float minrange, float maxrange);
int aussie_vector_count_outside_range(float v[], int n, float minrange, float maxrange);
int aussie_vector_count_greater(float v[], int n, float fval);
int aussie_vector_count_greater_equal(float v[], int n, float fval);
int aussie_vector_count_less_equal(float v[], int n, float fval);
int aussie_vector_count_less(float v[], int n, float fval);
int aussie_vector_count_equal(float v[], int n, float fval);
int aussie_vector_count_notequal(float v[], int n, float fval);

//-------------------------------------------------------------------------
// Element-wise actions on all vector elements
//-------------------------------------------------------------------------
void aussie_vector_expize_basic(float v[], int n);  // Exponentiate all vector elements with "exp"
void aussie_vector_expize(float v[], int n);   // Apply EXP to each element
#define aussie_vector_add_constant aussie_vector_add_scalar  // alias
//#define aussie_vector_add_scalar aussie_vector_add_constant  // alias
void aussie_vector_add_scalar(float v[], int n, float c);  // Add constant to all vector elements
void aussie_vector_add_scalar_AVX1(float v[], int n, float c);   // Add scalar constant to all vector elements
void aussie_vector_add_scalar_AVX2(float v[], int n, float c);   // Add scalar constant to all vector elements

//#define aussie_vector_multiply_scalar aussie_vector_multiply_constant  // alias

void aussie_vector_multiply_scalar(float v[], int n, float c);  // Multiply all vector elements by constant
void aussie_vector_multiply_scalar_AVX1(float v[], int n, float c);  // Multiply all vector elements by constant
void aussie_vector_multiply_scalar_AVX2(float v[], int n, float c);  // Multiply all vector elements by constant
void aussie_vector_multiply_scalar_AVX2_pointer_arith(float v[], int n, float c);  // Multiply all vector elements by constant

void aussie_vector_divide_scalar(float v[], int n, float fc);  // Divide all vector elements by constant


void aussie_vector_multiply_constant(float v[], int n, float c);  // Multiply all vector elements by constant
void aussie_vector_logize(float v[], int n);   // Apply LOG to each element
void aussie_vector_log10ize(float v[], int n);   // Apply LOG10 to each element
void aussie_vector_square(float v[], int n);   // Square each element
void aussie_vector_sqrt(float v[], int n);   // Apply SQRT (square root) to each element
void aussie_vector_fabs(float v[], int n);   // Apply FABS (absolute value) to each element
void aussie_vector_tanh(float v[], int n);   // Apply TANH (hyperbolic tangent) to each element
void aussie_vector_step_function(float v[], int n);    // Apply Step Function to each element
void aussie_vector_sign_function(float v[], int n);    // Apply Sign Function to each element

void aussie_vector_expf(float v[], int n);   // Apply EXPF (exponential) to each element
void aussie_vector_expf_pointer_arith(float v[], int n);   // Apply EXPF (exponential) to each element


float aussie_vector_expf_and_sum(float v[], int n);   // Apply EXPF (exponential) to each element

void aussie_vector_multiply_constant(float v[], int n, float c);  // Multiply all vector elements by constant
void aussie_vector_multiply_scalar_pointer_arith(float v[], int n, float c);  // Multiply all vector elements by constant


//-------------------------------------------------------------------------

#define AUSSIE_SQUARE(x)  ( (x) * (x) )


//-------------------------------------------------------------------------

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Arithmetic approximate multiplications...
//-------------------------------------------------------------------------
float aussie_vecdot_add_as_int_mogami(float v1[], float v2[], int n);   // Add as integer

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Kernel fusion
//-------------------------------------------------------------------------
float aussie_fused_vecdot_RELU_basic(float v1[], float v2[], int n);   // Basic fused dot product + RELU
float aussie_nonfused_vecdot_RELU_basic(float v1[], float v2[], int n);   // Basic fused dot product + RELU
void aussie_vector_addition_slow(float v[], int n, bool do_addition, float scalar);

//-------------------------------------------------------------------------
// Loop optimizations
//-------------------------------------------------------------------------
void aussie_vector_addition_loop_distribution(float v[], int n, bool do_addition, float scalar);

//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Vector binary operations -- modifies the LHS vector
//-------------------------------------------------------------------------
void aussie_vector_assign_vector(float v1[], float v2[], int n);   // v1 = v2 -- Put V2 into V1 (changing V1)
void aussie_vector_add_vector(float v1[], float v2[], int n);   // += -- Add V2 into V1 (changing V1)
void aussie_vector_multiply_vector(float v1[], float v2[], int n);   // v1 *= v2 -- Multiply V2 into V1 (changing V1)
void aussie_vector_subtract_vector(float v1[], float v2[], int n);   // v1 -= v2 -- Multiply V2 into V1 (changing V1)
void aussie_vector_divide_vector(float v1[], float v2[], int n);   // v1 /= v2 -- Multiply V2 into V1 (changing V1)
void aussie_vector_bitand_vector(float v1[], float v2[], int n);   // v1 &= v2 -- Multiply V2 into V1 (changing V1)
void aussie_vector_bitor_vector(float v1[], float v2[], int n);   // v1 |= v2 -- Multiply V2 into V1 (changing V1)
void aussie_vector_bitxor_vector(float v1[], float v2[], int n);   // v1 |= v2 -- Multiply V2 into V1 (changing V1)

//-------------------------------------------------------------------------
// Vector scan for negatives (mostly to use as examples of loop optimizations)
//-------------------------------------------------------------------------
bool aussie_vector_has_negative_basic(float v[], int n);
bool aussie_vector_has_negative_sentinel(float v[], int n);
bool aussie_vector_has_negative_sentinel2(float v[], int n);
bool aussie_vector_has_negative_sentinel3(float v[], int n);
bool aussie_vector_has_negative_pointer_arithmetic(float v[], int n);

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

void aussie_vector_setall(float v1[], int n, float fval);
void aussie_vector_setall_intarr(int v1[], int n, int ival);
void aussie_print_vector(float v[], int n);

void aussie_multiply_vectors(float v1[], float v2[], float result[], int n);
bool aussie_vector_equal(float v1[], float v2[], int n);
bool aussie_vector_equal_approx(float v1[], float v2[], int n, float err, bool warn=false);

//-------------------------------------------------------------------------
// Vector dot product benchmarking
void aussie_benchmark_vecdot();  // vector dot product benchmarks...
void run_vector_float_N(char* name, long int niter, long int nvecsize, 
	void (*voidvectorfnptr)(const float v[], int n),
	float (*floatvectorfnptr)(const float v[], const float v2[], int n) = NULL);
void run_vector_float_N_non_const(char* name, long int niter, long int nvecsize,
	void (*voidvectorfnptr)(float v[], int n),
	float (*floatvectorfnptr)(float v[], float v2[], int n) = NULL
);

#define aussie_is_aligned_16(ptr)  ((((unsigned long)(ptr)) &15ul) == 0)
#define aussie_is_aligned_32(ptr)  ((((unsigned long)(ptr)) &31ul) == 0)

void aussie_test_vector_sum(float v[], int n, float fexpected);

#endif //YVECTOR_INCLUDE_HEADER_H

