// yvector.cpp -- Vector computations (various) -- Aussie AI Base Library
// Created 6th October 2023
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

//#include "gelu_precomp_24bits.cpp"  // Test it compiles! (500kb CPP file)




//---------------------------------------------------
//---------------------------------------------------

#include "aport.h"
#include "aussieai.h"
#include "atest.h"
#include "aassert.h"
#include "afloat.h"
#include "aactivation.h"
#include "anorms.h"
#include "asoftmax.h"
#include "anormalize.h"
#include "atopk.h"
#include "aavx.h"

#include "avector.h"  // Self-include


//---------------------------------------------------
//---------------------------------------------------

#include <assert.h>

float vector_sum_assert_example1(float v[], int n)
{
	assert(v != NULL);
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}

//---------------------------------------------------
//---------------------------------------------------

float vector_sum_assert_example2(float v[], int n)
{
	yassert(v != NULL);
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}

float vector_sum_assert_example3(float v[], int n)
{
	if (v != NULL) {
		yassert(v != NULL);
		return 0.0;
	}
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}

float vector_sum_assert_example5(float v[], int n)
{
	yassert_param_tolerate_null(v, 0.0f);
	yassert_param_tolerate_null2(v, 0.0f);
	yassert_and_return(v != NULL, 0.0f);

	yassert(v != NULL);
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}


float vector_sum_assert_example4(float v[], int n)
{
	if (v == NULL) {
		yassert(v != NULL);
		return 0.0;
	}
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}


//---------------------------------------------------
//---------------------------------------------------

float aussie_vector_sum(float v[], int n)  // Summation
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}

float aussie_vector_sum_pointer_arith(float v[], int n)  // Summation
{
	float sum = 0.0;
	for (; n > 0; n--, v++) {
		sum += *v;
	}
	return sum;
}


float aussie_vector_product(float v[], int n)  // Product
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum *= v[i];
	}
	return sum;
}


float aussie_vector_sum_squares(float v[], int n)  // Summation of squares of all elements
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += v[i] * v[i];
	}
	return sum;
}

float aussie_vector_distance(float v[], int n)  // Euclidean distance/Magnitude of vector -- Sqrt of Summation of squares of all elements
{
	float f = aussie_vector_sum_squares(v, n);
	f = sqrtf(f);  // Square root (euclidean distance formula)
	return f;
}

float aussie_vector_max(float v[], int n)  // Maximum
{
	float vmax = v[0];
	for (int i = 1 /*not 0*/; i < n; i++) {
		if (v[i] > vmax) vmax = v[i];
	}
	return vmax;
}

float aussie_vector_min_max_fused(float v[], int n, float& vmax)  // Mininum returned, maximum in parameter
{
	vmax = v[0];
	float vmin = v[0];
	for (int i = 1 /*not 0*/; i < n; i++) {
		if (v[i] < vmin) vmin = v[i];
		if (v[i] > vmax) vmax = v[i];
	}
	return vmin;
}

float aussie_vector_min(float v[], int n)  // Mininum
{
	float vmin = v[0];
	for (int i = 1 /*not 0*/; i < n; i++) {
		if (v[i] < vmin) vmin = v[i];
	}
	return vmin;
}


void aussie_print_vector(float v[], int n)
{
	printf("[ ");
	for (int i = 0; i < n; i++) {
		printf("%3.2f ", v[i]);
	}
	printf("]\n");
}


float aussie_vector_avg(float v[], int n)  // Average
{
	if (n == 0) {
		yassert(n != 0);
		return 0.0;  // fail internal error
	}
	float sum = aussie_vector_sum(v, n);
	return sum / (float)n;
}



float aussie_vector_mean(float v[], int n)  // Mean (same as average)
{
	if (n == 0) {
		yassert(n != 0);
		return 0.0;  // fail internal error
	}
	float sum = aussie_vector_sum(v, n);
	return sum / (float)n;
}

float aussie_vector_mean_AVX1(float v[], int n)  // Mean (same as average)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	if (n == 0) {
		yassert(n != 0);
		return 0.0;  // fail internal error
	}
	float sum = aussie_vector_sum_AVX1(v, n);
	return sum / (float)n;
#endif //LINUX
}

float aussie_vector_mean_AVX2(float v[], int n)  // Mean (same as average)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	if (n == 0) {
		yassert(n != 0);
		return 0.0;  // fail internal error
	}
	float sum = aussie_vector_sum_AVX2(v, n);
	return sum / (float)n;
#endif //LINUX
}

float aussie_vector_sum_diff_squared(float v[], int n, float meanval)
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		float fdiff = v[i] - meanval;
		sum += (fdiff * fdiff);   // add the squared differences..
	}
	return sum;
}

float aussie_vector_sum_diff_squared_fused(float v[], int n, float meanval)
{
	// Fused version of "sum diff squared" that leaves the DIFF in the vector...
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		float fdiff = v[i] - meanval;
		sum += (fdiff * fdiff);   // add the squared differences..
		v[i] = fdiff;  // Fusion: store the diff there (for another loop to use later)
	}
	return sum;
}

float aussie_vector_sum_diff_squared_fission(float v[], int n, float meanval)
{
	// FISSION version of "sum diff squared" 
	// Loop 1. Calculate DIFF in the vector...
	for (int i = 0; i < n; i++) {
		v[i] = v[i] - meanval;
	}
	// Loop 2. Calculate the sum-of-squares...
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		float f = v[i];
		sum += (f * f);   // add the squares..
	}
	return sum;
}

float aussie_vector_sum_diff_squared_fissionB(float v[], int n, float meanval)
{
	// FISSION version of "sum diff squared" 
	aussie_vector_add_scalar(v, n, -meanval);  // Loop 1. DIFFs
	float sum = aussie_vector_sum_squares(v, n);  // Loop 2. Sum-of-squares...
	return sum;
}

float aussie_vector_sum_diff_squared_fission_AVX1(float v[], int n, float meanval)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// FISSION version of "sum diff squared" 
	aussie_vector_add_scalar_AVX1(v, n, -meanval);  // Loop 1. DIFFs
	float sum = aussie_vector_sum_squares_AVX1(v, n);  // Loop 2. Sum-of-squares...
	return sum;
#endif //LINUX
}

float aussie_vector_sum_diff_squared_fission_AVX2(float v[], int n, float meanval)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// FISSION version of "sum diff squared" 
	aussie_vector_add_scalar_AVX2(v, n, -meanval);  // Loop 1. DIFFs
	float sum = aussie_vector_sum_squares_AVX2(v, n);  // Loop 2. Sum-of-squares...
	return sum;
#endif //LINUX
}

float aussie_vector_sum_squared(float v[], int n)
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += AUSSIE_SQUARE(v[i]);   // sum the squared elements..
	}
	return sum;
}



float aussie_vector_mean_and_variance_fused(float v[], int n, float& fmean_out)  // Variance (square of std. dev.)
{
	fmean_out = aussie_vector_mean(v, n);  // Get the mean/average
	float sumsquares = aussie_vector_sum_diff_squared_fused(v, n, fmean_out);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
}

float aussie_vector_mean_and_variance_fused_AVX1(float v[], int n, float& fmean_out)  // Variance (square of std. dev.)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	fmean_out = aussie_vector_mean_AVX1(v, n);  // Get the mean/average
	float sumsquares = aussie_vector_sum_diff_squared_fused_AVX1(v, n, fmean_out);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
#endif //LINUX
}

float aussie_vector_mean_and_variance_fused_AVX2(float v[], int n, float& fmean_out)  // Variance (square of std. dev.)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	fmean_out = aussie_vector_mean_AVX1(v, n);  // Get the mean/average
	float sumsquares = aussie_vector_sum_diff_squared_fused_AVX2(v, n, fmean_out);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
#endif //LINUX
}

float aussie_vector_mean_and_variance_all_AVX1(float v[], int n, float& fmean_out)  // Variance (square of std. dev.)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	fmean_out = aussie_vector_mean_AVX1(v, n);  // Get the mean/average
	float sumsquares = aussie_vector_sum_diff_squared_fission_AVX1(v, n, fmean_out);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
#endif //LINUX
}

float aussie_vector_mean_and_variance_all_AVX2(float v[], int n, float& fmean_out)  // Variance (square of std. dev.)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	fmean_out = aussie_vector_mean_AVX2(v, n);  // Get the mean/average
	float sumsquares = aussie_vector_sum_diff_squared_fission_AVX2(v, n, fmean_out);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
#endif //LINUX
}


float aussie_vector_mean_and_variance(float v[], int n, float &fmean_out)  // Variance (square of std. dev.)
{
	fmean_out = aussie_vector_mean(v, n);  // Get the mean/average

	float sumsquares = aussie_vector_sum_diff_squared(v, n, fmean_out);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
}

float aussie_vector_variance(float v[], int n)  // Variance (square of std. dev.)
{
	float vmean = aussie_vector_mean(v, n);  // Get the mean/average

	float sumsquares = aussie_vector_sum_diff_squared(v, n, vmean);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
}

float aussie_vector_variance_of_mean(float v[], int n, float fmean)  // Variance from already-calculated mean (square of std. dev.)
{
	float sumsquares = aussie_vector_sum_diff_squared(v, n, fmean);  // Sum of squared-diffs from mean
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
}

float aussie_vector_variance_of_mean_fused(float v[], int n, float fmean)  // Variance with Fusion (leaves DIFF from MEAN in the vector)
{
	// Fusion version that leaves the DIFF in the vector..
	float sumsquares = aussie_vector_sum_diff_squared_fused(v, n, fmean);  // Sum of squared-diffs from mean (leaves DIFF in vector)
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
}

float aussie_vector_variance_of_mean_fused_AVX1(float v[], int n, float fmean)  // Variance with Fusion (leaves DIFF from MEAN in the vector)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// Fusion version that leaves the DIFF in the vector..
	float sumsquares = aussie_vector_sum_diff_squared_fused_AVX1(v, n, fmean);  // Sum of squared-diffs from mean (leaves DIFF in vector)
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
#endif //LINUX
}

float aussie_vector_variance_of_mean_fused_AVX2(float v[], int n, float fmean)  // Variance with Fusion (leaves DIFF from MEAN in the vector)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// Fusion version that leaves the DIFF in the vector..
	float sumsquares = aussie_vector_sum_diff_squared_fused_AVX2(v, n, fmean);  // Sum of squared-diffs from mean (leaves DIFF in vector)
	return sumsquares / (float)n;  // Divide the sum-of-squares by N...
#endif //LINUX
}

void aussie_multiply_vectors(float v1[], float v2[], float result[], int n)
{
	for (int i = 0; i < n; i++) result[i] = v1[i] * v2[i];
}

bool aussie_vector_equal(float v1[], float v2[], int n)
{
	for (int i = 0; i < n; i++) {
		if (v1[i] != v2[i]) return false;
	}
	return true;
}

bool aussie_vector_equal_approx(float v1[], float v2[], int n, float err, bool warn)
{
	for (int i = 0; i < n; i++) {
		if (v1[i] != v2[i] && fabs(v1[i] - v2[i]) > err) {
			if (warn) ytestf(v1[i], v2[i]);
			return false;
		}
	}
	return true;
}


float aussie_vector_mean_and_stddev(float v[], int n, float &meanout)  // Std. dev (sqrt of variance)
{
	float vvariance = aussie_vector_mean_and_variance(v, n, meanout);  // Get the Variance
	return sqrtf(vvariance);   // Std.dev is just the sqrt of variance...
}

float aussie_vector_mean_and_stddev_fused(float v[], int n, float& meanout)  // Std. dev (sqrt of variance)
{
	float vvariance = aussie_vector_mean_and_variance_fused(v, n, meanout);  // Get the Variance
	return sqrtf(vvariance);   // Std.dev is just the sqrt of variance...
}

float aussie_vector_mean_and_stddev_all_AVX1(float v[], int n, float& meanout)  // Std. dev (sqrt of variance)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	float vvariance = aussie_vector_mean_and_variance_all_AVX1(v, n, meanout);  // Get the Variance
	return sqrtf(vvariance);   // Std.dev is just the sqrt of variance...
#endif //LINUX
}



float aussie_vector_mean_and_stddev_fused_AVX1(float v[], int n, float& meanout)  // Std. dev (sqrt of variance)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	float vvariance = aussie_vector_mean_and_variance_fused_AVX1(v, n, meanout);  // Get the Variance
	return sqrtf(vvariance);   // Std.dev is just the sqrt of variance...
#endif //LINUX
}

float aussie_vector_mean_and_stddev_all_AVX2(float v[], int n, float& meanout)  // Std. dev (sqrt of variance)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	float vvariance = aussie_vector_mean_and_variance_all_AVX2(v, n, meanout);  // Get the Variance
	return sqrtf(vvariance);   // Std.dev is just the sqrt of variance...
#endif //LINUX
}

float aussie_vector_mean_and_stddev_fused_AVX2(float v[], int n, float& meanout)  // Std. dev (sqrt of variance)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	float vvariance = aussie_vector_mean_and_variance_fused_AVX2(v, n, meanout);  // Get the Variance
	return sqrtf(vvariance);   // Std.dev is just the sqrt of variance...
#endif //LINUX
}

float aussie_vector_standard_deviation(float v[], int n)  // Std. dev (sqrt of variance)
{
	float vvariance = aussie_vector_variance(v, n);  // Get the Variance
	return sqrtf(vvariance);   // Std.dev is just the sqrt of variance...
}

void aussie_vector_clear(float v[], int n)  // Clear to zero vector
{
	for (int i = 0; i < n; i++) {
		v[i] = 0.0;  // Clear to zero
	}
}

int aussie_vector_count_negatives(float v[], int n)  
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] < 0.0) ct++;
	}
	return ct;
}

int aussie_vector_count_zeros(float v[], int n)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] == 0.0) ct++;
	}
	return ct;
}

int aussie_vector_count_nonzeros(float v[], int n)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] != 0.0) ct++;
	}
	return ct;
}

int aussie_vector_count_positives(float v[], int n)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] > 0.0) ct++;
	}
	return ct;
}

int aussie_vector_count_in_range(float v[], int n, float minrange, float maxrange)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] >= minrange && v[i] <= maxrange) ct++;
	}
	return ct;
}

int aussie_vector_count_outside_range(float v[], int n, float minrange, float maxrange)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if ( ! (v[i] >= minrange && v[i] <= maxrange)) ct++;
	}
	return ct;
}

int aussie_vector_count_greater(float v[], int n, float fval)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] > fval) ct++;
	}
	return ct;
}

int aussie_vector_count_greater_equal(float v[], int n, float fval)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] >= fval) ct++;
	}
	return ct;
}

int aussie_vector_count_less_equal(float v[], int n, float fval)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] <= fval) ct++;
	}
	return ct;
}

int aussie_vector_count_less(float v[], int n, float fval)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] < fval) ct++;
	}
	return ct;
}

int aussie_vector_count_equal(float v[], int n, float fval)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] == fval) ct++;
	}
	return ct;
}

int aussie_vector_count_notequal(float v[], int n, float fval)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		if (v[i] != fval) ct++;
	}
	return ct;
}

//---------------------------------------------------
//---------------------------------------------------
void aussie_vector_expize_basic(float v[], int n)  // Exponentiate all vector elements with "exp"
{
	for (int i = 0; i < n; i++) {
		float eval = expf(v[i]);  // expf is float version of exp
		yassert(eval > 0.0);
		v[i] = eval;
	}
}

//---------------------------------------------------
//---------------------------------------------------
void aussie_vector_add_scalar(float v[], int n, float c)  // Add constant to all vector elements
{
	for (int i = 0; i < n; i++) {
		v[i] += c;
	}
}




//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_multiply_constant(float v[], int n, float c)  // Multiply all vector elements by constant
{
	for (int i = 0; i < n; i++) {
		v[i] *= c;
	}
}

void aussie_vector_multiply_scalar(float v[], int n, float c)  // Multiply all vector elements by constant
{
	for (int i = 0; i < n; i++) {
		v[i] *= c;
	}
}
void aussie_vector_divide_scalar(float v[], int n, float fc)  // Divide all vector elements by constant
{
	for (int i = 0; i < n; i++) {
		v[i] /= fc;  // Float division
	}
}




void aussie_vector_multiply_scalar_pointer_arith(float v[], int n, float c)  // Multiply all vector elements by constant
{
	for (; n > 0; n--, v++) {
		*v *= c;
	}
}



//---------------------------------------------------
//---------------------------------------------------
void aussie_vector_set_constant(float v[], int n, float c)  // Set all vector elements to constant
{
	for (int i = 0; i < n; i++) {
		v[i] = c;
	}
}

//---------------------------------------------------


void aussie_vector_expize(float v[], int n)   // Apply EXP to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = expf(v[i]);
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_logize(float v[], int n)   // Apply LOG to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = logf(v[i]);
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_log10ize(float v[], int n)   // Apply LOG10 to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = log10f(v[i]);
	}
}

void aussie_vector_square(float v[], int n)   // Square each element
{
	for (int i = 0; i < n; i++) {
		v[i] = AUSSIE_SQUARE(v[i]);
	}
}

void aussie_vector_step_function(float v[], int n)    // Apply Step Function to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = AUSSIE_STEP_FUNCTION(v[i]);
	}
}

void aussie_vector_sign_function(float v[], int n)    // Apply Sign Function to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = AUSSIE_SIGN_FUNCTION(v[i]);
	}
}

void aussie_vector_expf(float v[], int n)   // Apply EXPF (exponential) to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = expf(v[i]);
	}
}

void aussie_vector_expf_pointer_arith(float v[], int n)   // Apply EXPF (exponential) to each element
{
	for (; n > 0; n--, v++) {
		*v = expf(*v);
	}
}

float aussie_vector_expf_and_sum(float v[], int n)   // Apply EXPF (exponential) to each element
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		v[i] = expf(v[i]);
		sum += v[i];
	}
	return sum;
}

void aussie_vector_sqrt(float v[], int n)   // Apply SQRT (square root) to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = sqrtf(v[i]);
	}
}

void aussie_vector_fabs(float v[], int n)   // Apply FABS (absolute value) to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = fabsf(v[i]);
	}
}

void aussie_vector_tanh(float v[], int n)   // Apply TANH (hyperbolic tangent) to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = tanhf(v[i]);
	}
}


//---------------------------------------------------
//---------------------------------------------------


float aussie_vecdot_add_as_int_mogami(float v1[], float v2[], int n)   // Add as integer
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		int c = *(int*)&(v1[i]) + *(int*)&(v2[i]) - 0x3f800000;  // Mogami(2020)
		sum += *(float*)&c;
	}
	return sum;
}

//---------------------------------------------------

float aussie_fused_vecdot_RELU_basic(float v1[], float v2[], int n)   // Basic fused dot product + RELU
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += v1[i] * v2[i];
	}
	return AUSSIE_RELU_MACRO(sum);
}

float aussie_nonfused_vecdot_RELU_basic(float v1[], float v2[], int n)   // Basic fused dot product + RELU
{
	float f = aussie_vecdot_basic(v1, v2, n);  // Basic vector dot product
	f = AUSSIE_RELU_MACRO(f);
	return f;
}

void aussie_vector_addition_slow(float v[], int n, bool do_addition, float scalar)
{
	for (int i = 0; i < n; i++) {
		if (do_addition) v[i] += scalar;  // Add scalar
		else v[i] -= scalar;   // Subtraction
	}
}

void aussie_vector_addition_loop_distribution(float v[], int n, bool do_addition, float scalar)
{
	if (do_addition) { // Add scalar
		for (int i = 0; i < n; i++) {
			v[i] += scalar;  // Addition
		}
	}
	else {  // Subtract scalar
		for (int i = 0; i < n; i++) {
			v[i] -= scalar;   // Subtraction
		}
	}
}

//---------------------------------------------------
//---------------------------------------------------

float aussie_vecdot_basic(const float v1[], const float v2[], int n)   // Basic FLOAT vector dot product
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += v1[i] * v2[i];
	}
	return sum;
}

float aussie_vecdot_zero_skipping(const float v1[], const float vweights[], int n) 
{
	// Vector dot product with simple zero skipping test
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		if (vweights[i] != 0.0f) { // Skip zero weights
			sum += v1[i] * vweights[i];
		}
	}
	return sum;
}


int aussie_vecdot_int_basic(int v1[], int v2[], int n)   // Basic INT vector dot product
{
	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v1[i] * v2[i];
	}
	return sum;
}

float aussie_vecdot_section512(float v1[], float v2[])   // Simulate parallel 512 section 
{
	float sum = 0.0;
	const int chunksize = 512;  // Simulate fixed size GPU array...
	for (int i = 0; i < chunksize/*512*/; i++) {
		sum += v1[i] * v2[i];
	}
	return sum;

}

//---------------------------------------------------
//---------------------------------------------------

float aussie_vecdot_parallel_basic(float v1[], float v2[], int n)   // Simulated parallel vector dot product
{
	// Simulate parallel vecdot via sections 512 length at a time...

	if (n % 512 != 0) {  // Assumes exact multiple...
		yassert(n % 512 == 0);
		return 0.0;  // fail
	}

	// Split into 512-item size chunks...
	float sum = 0.0;
	int chunks = n / 512;
	for (int i = 0; i < chunks; i++) {
		sum += aussie_vecdot_section512(&v1[i * 512], &v2[i * 512]);
	}

	return sum;
}

//---------------------------------------------------

float aussie_vecdot_parallel_odd_sizes(float v1[], float v2[], int n)   // Simulated parallel vector dot product
{
	// Simulate parallel vecdot via sections 512 length at a time...

	// Split into 512-item size chunks...
	const int sizechunk = 512;
	int chunks = n / sizechunk;
	float sum = 0.0;
	int i = 0;
	for (; i < chunks; i++) {
		// Process a 512 chunk (vectorized)
		sum += aussie_vecdot_section512(&v1[i * 512], &v2[i * 512]);
	}

	if (i* sizechunk < n) {
		// We have an odd size, with extra leftovers at the end...
		int start = i * sizechunk;
		int leftover = n - start;
		sum += aussie_vecdot_basic(&v1[start], &v2[start], leftover);
	}

	return sum;
}


float aussie_vecdot_parallel_padding(float v1[], float v2[], int n)   // Use padding for leftovers...
{
	// Simulate parallel vecdot via sections 512 length at a time...

	// Split into 512-item size chunks...
	const int sizechunk = 512;
	int chunks = n / sizechunk;
	float sum = 0.0;
	int i = 0;
	for (; i < chunks; i++) {
		// Process a 512 chunk (vectorized)
		sum += aussie_vecdot_section512(&v1[i * 512], &v2[i * 512]);
	}

	if (i * sizechunk < n) {
		// We have an odd size, with extra leftovers at the end...
		// Copy extras to temporary vectors of size 512...
		int start = i * sizechunk;
		int leftover = n - start;
		float vtemp1[512];
		float vtemp2[512];
		memcpy(vtemp1, &v1[start], leftover * sizeof(float));  // Put leftovers at front
		memcpy(vtemp2, &v2[start], leftover * sizeof(float));  // Put leftovers at front
		memset(vtemp1 + leftover, 0, sizeof(float) * (512 - leftover));  // Padding zeros at end
		memset(vtemp2 + leftover, 0, sizeof(float) * (512 - leftover));  // Padding zeros at end
		// Process a 512 chunk (vectorized)
		sum += aussie_vecdot_section512(vtemp1, vtemp2);
	}

	return sum;
}


//---------------------------------------------------
//---------------------------------------------------

float aussie_vecdot_pointer_arithmetic(float v1[], float v2[], int n)   // Pointer arithmetic vector dot product
{
	float sum = 0.0;
	float* endv1 = v1 + n;  // v1 start plus n*4 bytes
	for (; v1 < endv1; v1++,v2++) {
		sum += (*v1) * (*v2);
	}
	return sum;
}



float aussie_vecdot_reverse_basic(float v1[], float v2[], int n)   // REVERSED basic vector dot product
{
	float sum = 0.0;
	for (int i = n - 1; i >= 0; i--) {  // Note: i cannot be "unsigned" or "size_t" type
		sum += v1[i] * v2[i];
	}
	return sum;
}

float aussie_vecdot_reverse_basic2(float v1[], float v2[], int n)   // REVERSED basic vector dot product #2
{
	float sum = 0.0;
	n--;  // Use "n" not "i"
	for (; n >= 0; n--) {  // Note: n cannot be "unsigned" or "size_t" type
		sum += v1[n] * v2[n];
	}
	return sum;
}

float aussie_vecdot_reverse_zerotest(float v1[], float v2[], int n)   // Reversed-with-zero-test vector dot product
{
	float sum = 0.0;
	int i = n - 1;
	do {
		sum += v1[i] * v2[i];
		i--;
	} while (i != 0);   // Zero-test is faster than ">=" test...
	sum += v1[0] * v2[0];  // Don't skip the last one!
	return sum;
}

float aussie_vecdot_unroll4_basic(float v1[], float v2[], int n)  // Loop-unrolled Vector dot product 
{
	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return 0.0; // fail
	}
	float sum = 0.0;
	for (int i = 0; i < n; ) {
		sum += v1[i] * v2[i]; i++;
		sum += v1[i] * v2[i]; i++;
		sum += v1[i] * v2[i]; i++;
		sum += v1[i] * v2[i]; i++;
	}
	return sum;
}

#if !LINUX
#include <intrin.h>
#endif //LINUX

// aussie_vecdot_AVX1 ... 
float aussie_vecdot_unroll_AVX1(const float v1[], const float v2[], int n)  // AVX-1 loop-unrolled (4 floats) Vector dot product 
{		
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// Code sequence from aussie_vecdot_unroll_AVX1

	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return 0.0; // fail
	}
	float sum = 0.0;
	for (int i = 0; i < n; i += 4) {
		// AVX1: Vector dot product of 2 vectors 
		//  ... process 4x32-bit floats in 128 bits
		__m128 r1 = _mm_loadu_ps(&v1[i]);   // Load floats into 128-bits
		__m128 r2 = _mm_loadu_ps(&v2[i]);
		__m128 dst = _mm_dp_ps(r1, r2, 0xf1); // Dot product
		sum += _mm_cvtss_f32(dst);
	}
	return sum;
#endif //LINUX
}

// aussie_vecdot_AVX2
float aussie_vecdot_FMA_unroll_AVX2(const float v1[], const float v2[], int n)   // AVX2 vecdot using FMA (Fused Multiply-Add) primitives
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else

	if (n % 8 != 0) {
		yassert(n % 8 == 0);
		return 0.0; // fail
	}

	__m256 sumdst = _mm256_setzero_ps();   // Set accumulators to zero
#if 0
	float* farrtest = (float*)&sumdst;
	float sumtest1 = farrtest[0] + farrtest[1] + farrtest[2] + farrtest[3];
	ytestf(sumtest1, 0.0f);
	float sumtest2 =
		farrtest[4] + farrtest[5] + farrtest[6] + farrtest[7];
	float sumtest = sumtest1 + sumtest2;
	ytestf(sumtest2, 0.0f);
	ytestf(sumtest, 0.0f);
#endif

	for (int i = 0; i < n; i += 8) {
		// AVX2: Vector dot product of 2 vectors 
		//  ... process 8x32-bit floats in 256 bits
		__m256 r1 = _mm256_loadu_ps(&v1[i]);   // Load floats into 128-bits
		__m256 r2 = _mm256_loadu_ps(&v2[i]);
		sumdst = _mm256_fmadd_ps(r1, r2, sumdst); // FMA of 3 vectors
	}

	// Add the final 8 accumulators manually
	float* farr = (float*)&sumdst;
	float sum = farr[0] + farr[1] + farr[2] + farr[3]
		+ farr[4] + farr[5] + farr[6] + farr[7];
	return sum;
#endif //LINUX
}

float aussie_vecdot_FMA_unroll_AVX1(const float v1[], const float v2[], int n)   // AVX1 vecdot using FMA (Fused Multiply-Add) primitives
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// Code sequence from aussie_vecdot_unroll_AVX1

	if (n % 4 != 0) {
		yassert(n % 4 == 0);
		return 0.0; // fail
	}

	__m128 sumdst = _mm_setzero_ps();   // Set accumulators to zero
	for (int i = 0; i < n; i += 4) {
		// AVX1: Vector dot product of 2 vectors 
		//  ... process 4x32-bit floats in 128 bits
		__m128 r1 = _mm_loadu_ps(&v1[i]);   // Load floats into 128-bits
		__m128 r2 = _mm_loadu_ps(&v2[i]);
		sumdst = _mm_fmadd_ps(r1, r2, sumdst); // FMA of 3 vectors
	}

	// Add the final 4 accumulators manually
	float* farr = (float*)&sumdst;
	float sum = farr[0] + farr[1] + farr[2] + farr[3];  // Manual add the 4 floats...
	return sum;
#endif //LINUX
}



float aussie_vecdot_unroll_AVX2(const float v1[], const float v2[], int n)  // AVX-2 loop-unrolled (8 floats) Vector dot product 
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	if (n % 8 != 0) { 
		yassert(n % 8 == 0);
		return 0.0; // fail
	}
#if 0 // manual enable
	yassert(aussie_is_aligned_16(v1));
	yassert(aussie_is_aligned_16(v2));
	yassert(aussie_is_aligned_32(v1));
	yassert(aussie_is_aligned_32(v2));
#endif

	float sum = 0.0;
	for (int i = 0; i < n; i += 4 /*4 not 8! problem..*/) {
		// AVX2 dot product of 8 float's in 256-bits
		// Code sequence from aussie_vecdot_unroll_AVX2
		__m256 r1 = _mm256_loadu_ps(&v1[i]);   // Load floats into 256-bits ...
		__m256 r2 = _mm256_loadu_ps(&v2[i]);
		__m256 dst = _mm256_dp_ps(r1, r2, 0xFF);   // Dot product (dotp)
		sum += _mm256_cvtss_f32(dst);
	}
	return sum;
#endif //LINUX
}


float aussie_vecdot_unroll4_better(float v1[], float v2[], int n)  // Loop-unrolled Vector dot product 
{
	int i = 0;
	float sum = 0.0;
	if (n % 4 != 0) {
		// Handle the extra cases...
		switch (n % 4) {
		case 1: 
			sum += v1[i] * v2[i]; i++; 
			break;
		case 2: 
			sum += v1[i] * v2[i]; i++;
			sum += v1[i] * v2[i]; i++;
			break;
		case 3:
			sum += v1[i] * v2[i]; i++;
			sum += v1[i] * v2[i]; i++;
			sum += v1[i] * v2[i]; i++;
			break;
		default: yassert_not_reached(); break;
		} // end switch
		// Keep going below with the rest of the vector
	}
	for (; i < n; ) {  // Unrolled 4 times...
		sum += v1[i] * v2[i]; i++;
		sum += v1[i] * v2[i]; i++;
		sum += v1[i] * v2[i]; i++;
		sum += v1[i] * v2[i]; i++;
	}
	return sum;
}
#define aussie_sqrt_optimized sqrtf // Dummy for example

void aussie_vector_do_sqrt(float v[], int n)
{
	for (int i = 0; i < n; i++) {
		if (i < 100) { // Fast case
			v[i] = aussie_sqrt_optimized(v[i]);
		}
		else {  // General case
			v[i] = sqrtf(v[i]);
		}
	}
}

void aussie_vector_do_sqrt_loop_splitting(float v[], int n)
{
	for (int i = 0; i < 100; i++) { // Fast cases		
		v[i] = aussie_sqrt_optimized(v[i]);
	}
	for (int i = 100; i < n; i++) { // General cases		
		v[i] = sqrtf(v[i]);
	}
}

float aussie_vecdot_unroll4_duffs_device(float v1[], float v2[], int n)  
{
	// Unrolled dot product with Duff's Device 
	int i = 0;
	float sum = 0.0;
	switch (n % 4) {
		for (; i < n; ) {
			case 0:	sum += v1[i] * v2[i]; i++;
			case 3: sum += v1[i] * v2[i]; i++;
			case 2: sum += v1[i] * v2[i]; i++;
			case 1: sum += v1[i] * v2[i]; i++;
			default:;
		} // end for
	} // end switch

	// Note that this trick is not actually very useful for vectorization,
	// because a switch cannot branch into the middle of a vectorized intrinsic.
	// Note that none of the cases has a break statement and instead relies on fallthrough of switch cases.
	// Similarly, the default clause is mainly just to avoid getting a spurious compilation warning,
	// and that it also has only a semicolon, not a break.
	// Note that this code is buggy for n==0, because it incorrectly does 4 iterations.
	// Note that the cases are in reverse order, except 0 at the top
	// Note that this sounds like a hack, but actually uses standardized C++ semantics
	// and should work on all platforms. Branching into the middle of a loop with a switch
	// is legal, provided it doesn't bypass a local variable initialization in C++, which 
	// it doesn't in this case. Also, the case fallthrough semantics are standard for C and C++ since inception.
	// If you like this kind of coding trickery, search up Jensen's device and Pigeon's device.
	// 
	return sum;
}



bool aussie_vector_has_negative_basic(float v[], int n)
{
	for (int i = 0; i < n; i++) {
		if (v[i] < 0.0) return true;  // Found negative
	}
	return false;  // No negatives
}

bool aussie_vector_has_negative_pointer_arithmetic(float v[], int n)
{
	float* endv = &v[n];
	for ( ; v != endv; v++) {
		if (*v < 0.0) return true;  // Found negative
	}
	return false;  // No negatives
}

bool aussie_vector_has_negative_sentinel(float v[], int n)
{
	v[n] = -99.0;  // Dummy negative (BUG!)
	int i = 0;
	for ( ; /*GONE!*/; i++) {
		if (v[i] < 0.0) break;  // Found negative
	}
	if (i == n) return false;  // At the dummy (fake success)
	return true;  // Found a negative (for real)
}

bool aussie_vector_has_negative_sentinel2(float v[], int n)
{
	float save = v[n - 1];  // Save it!
	v[n - 1] = -99.0;  // Dummy negative at end
	int i = 0;
	for ( ; /*GONE!*/; i++) {
		if (v[i] < 0.0) break;  // Found negative
	}
	v[n - 1] = save;  // Restore it!
	if (i == n - 1) {
		// At the dummy (fake success)
		if (save < 0.0) return true;  // Must check it!
		return false;  
	}
	return true;  // Found a negative (for real)
}

bool aussie_vector_has_negative_sentinel3(float v[], int n)
{
	// Sentinel + Pointer Arithmetic
	float *savev = &v[n - 1];
	float save = *savev;  // Save it!
	*savev = -99.0;  // Dummy negative at end v[n - 1]
	for (; /*GONE!*/; v++) {
		if (*v < 0.0) break;  // Found negative
	}
	*savev = save;  // Restore it!
	if (v == savev) {
		// At the dummy (fake success)
		if (save < 0.0) return true;  // Must check it!
		return false;
	}
	return true;  // Found a negative (for real)
}

//---------------------------------------------------
//---------------------------------------------------


float aussie_vecdot_perforated_slow(float v1[], float v2[], int n, int percent_perforation)   // Loop perforation -- vector dot product
{
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		if ( (int)( rand() % 100 ) + 1 <= percent_perforation) {
			// This iteration is perforated...
			continue; // Skip it...
		}
		sum += v1[i] * v2[i];
	}
	return sum;
}


//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_zero(float v1[], int n)   // Clear vector to all zeros
{
	for (int i = 0; i < n; i++) {
		v1[i] = 0.0;
	}
}

void aussie_vector_copy_basic(float vdest[], float vsrc[], int n)
{
	for (int i = 0; i < n; i++) {
		vdest[i] = vsrc[i];
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_setall(float v1[], int n, float fval)
{
	for (int i = 0; i < n; i++) {
		v1[i] = fval;
	}
}

void aussie_vector_setall_intarr(int v1[], int n, int ival)
{
	for (int i = 0; i < n; i++) {
		v1[i] = ival;
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_set_1_N(float v1[], int n)   // Set vector values to 1..N 
{
	for (int i = 0; i < n; i++) {
		v1[i] = (float)(i + 1);   // set to 1..N ...
	}
}

void aussie_vector_set_range(float v[], int n, int start, int end)   // Set vector values to START..END
{
	yassert(end > start);
	float fstart = (float)start;
	float fend = (float)end;
	float finc = (fend - fstart) / (float)n;

	int i = 0;
	for (float f = fstart; i < n; i++, f += finc) {
		v[i] = f;
	}

}


void aussie_ivector_set_1_N(int v1[], int n)   // Set vector values to 1..N 
{
	for (int i = 0; i < n; i++) {
		v1[i] = (i + 1);   // set to 1..N ...
	}
}

void aussie_vector_set_1_N_reverse(float v1[], int n)   // Set vector values to N..1
{
	for (int i = 0; i < n; i++) {
		v1[i] = (float)(n - i);   // set to N..1...
	}
}


void aussie_vector_set_1_N_MAXN(float v1[], int n, int maxn)   // Set vector values to 1..N, but cycling
{
	float fval = 0.0f;
	for (int i = 0; i < n; i++) {
		fval += 1.0f;
		v1[i] = fval; // set to 1..N ...
		if (i % maxn == 0) fval = 0.0f;  // reset cycle at maximum
	}
}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------

void aussie_vector_assign_vector(float v1[], float v2[], int n)   // v1 = v2 -- Put V2 into V1 (changing V1)
{
	// TODO: Faster version with memcpy -- aussie_vector_assign_vector
	for (int i = 0; i < n; i++) {
		v1[i] = v2[i];
	}
}

void aussie_vector_add_vector(float v1[], float v2[], int n)   // v1 += v2 -- Add V2 into V1 (changing V1)
{
	for (int i = 0; i < n; i++) {
		v1[i] += v2[i];
	}
}

//---------------------------------------------------

void aussie_vector_multiply_vector(float v1[], float v2[], int n)   // v1 *= v2 -- Multiply V2 into V1 (changing V1)
{
	for (int i = 0; i < n; i++) {
		v1[i] *= v2[i];
	}
}

void aussie_vector_subtract_vector(float v1[], float v2[], int n)   // v1 -= v2 -- Multiply V2 into V1 (changing V1)
{
	for (int i = 0; i < n; i++) {
		v1[i] -= v2[i];
	}
}

void aussie_vector_divide_vector(float v1[], float v2[], int n)   // v1 /= v2 -- Multiply V2 into V1 (changing V1)
{
	for (int i = 0; i < n; i++) {
		if (v2[i] == 0.0) {
			v1[i] = 0.0; // Avoid divide by zero
			continue;
		}
		v1[i] /= v2[i];
	}
}

void aussie_vector_bitand_vector(float v1[], float v2[], int n)   // v1 &= v2 -- Multiply V2 into V1 (changing V1)
{
	for (int i = 0; i < n; i++) {
		unsigned int ui1 = (unsigned int)v1[i];
		unsigned int ui2 = (unsigned int)v2[i];
		unsigned int ui3 = ui1 & ui2;  // Bitwise AND integer operation
		v1[i] = (float)ui3;
	}
}

void aussie_vector_bitor_vector(float v1[], float v2[], int n)   // v1 |= v2 -- Multiply V2 into V1 (changing V1)
{
	for (int i = 0; i < n; i++) {
		unsigned int ui1 = (unsigned int)v1[i];
		unsigned int ui2 = (unsigned int)v2[i];
		unsigned int ui3 = ui1 | ui2;  // Bitwise OR integer operation
		v1[i] = (float)ui3;
	}
}

void aussie_vector_bitxor_vector(float v1[], float v2[], int n)   // v1 |= v2 -- Multiply V2 into V1 (changing V1)
{
	for (int i = 0; i < n; i++) {
		unsigned int ui1 = (unsigned int)v1[i];
		unsigned int ui2 = (unsigned int)v2[i];
		unsigned int ui3 = ui1 ^ ui2;  // Bitwise XOR integer operation
		v1[i] = (float)ui3;
	}
}


bool aussie_vector_is_equal(float v1[], float v2[], int n)   // Test if 2 vectors are identical/equal (all elements)
{
	// TODO: Faster version possible with memcmp? -- aussie_vector_is_equal 
	// (Or can 2 floats differ in bits but have the same value, e.g. negative zero?)
	for (int i = 0; i < n; i++) {
		if (v1[i] != v2[i]) return false;  // Elements differ
	}
	return true;  // All elements were the same
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_yvector_test_dot_products(float v1[], float v2[], int n, float expected)
{
	float f = aussie_vecdot_basic(v1, v2, n);

	ytest(f == expected);

	float f2 = 0.0;
	
	if (n % 4 == 0) {
		f2 = aussie_vecdot_unroll4_basic(v1, v2, n);
		ytestf(f2, f);

#if !LINUX
		f2 = aussie_vecdot_unroll_AVX1(v1, v2, n);
		ytestf(f2, f);

		f2 = aussie_vecdot_FMA_unroll_AVX1(v1, v2, n);
		ytestf(f2, f);

		f2 = aussie_vecdot_FMA_unroll_AVX2(v1, v2, n);
		ytestf(f2, f);
#endif //LINUX


	}
#if !LINUX
	if (n % 8 == 0) {
		f2 = aussie_vecdot_unroll_AVX2(v1, v2, n);
		ytestf(f2 , f);
	}
#endif //LINUX

	f2 = aussie_vecdot_unroll4_better(v1, v2, n);
	ytestf(f2, f);

	f2 = aussie_vecdot_pointer_arithmetic(v1, v2, n);
	ytestf(f2, f);

	
	// Test Reversed Loops...
	f2 = aussie_vecdot_reverse_basic(v1, v2, n);
	ytestf(f2, f);
	f2 = aussie_vecdot_reverse_zerotest(v1, v2, n);
	ytestf(f2, f);
	f2 = aussie_vecdot_reverse_basic2(v1, v2, n);
	ytestf(f2, f);

	
	f2 = aussie_vecdot_unroll4_duffs_device(v1, v2, n);
	ytestf(f2 , f);

	// Test Loop Perforation
	f2 = aussie_vecdot_perforated_slow(v1, v2, n, 50 /*50% perforation*/);
	ytest(f2 <= f);  // Must be less, some additions are skipped...
	if (f == 0.0) ytest(f2 == 0);  // Must also be zero...
	if (f > 0.0 && n > 5) ytest(f2 < f);  // Should be strictly less...

	f2 = aussie_vecdot_perforated_slow(v1, v2, n, 100 /*100% perforation*/);
	ytest(f2 == 0.0);  // Should skip 100% of them!...

}

void aussie_yvector_test_dot_products_BIG(float v1[], float v2[], int n, float expected)
{
	float f = aussie_vecdot_basic(v1, v2, n);
	ytestf(f, expected);

	if (n % 512 == 0) {
		float f2 = aussie_vecdot_parallel_basic(v1, v2, n);   // Simulated 512 parallel vector dot product
		ytestf(f2, expected);
	}


	float f3 = aussie_vecdot_parallel_odd_sizes(v1, v2, n);
	ytestf(f3, expected);

	f3 = aussie_vecdot_parallel_padding(v1, v2, n);
	ytestf(f3, expected);

}


void aussie_precompute_tests()
{
	return;  // Disable manually...
	aussie_GELU_setup_table_FP32_24bits_PRINT_SOURCE("GELU", "gelu_precomp_24bits.cpp");
}

void aussie_test_vector_sum(float v[], int n, float fexpected)
{
	float f = aussie_vector_sum(v, n);
	ytestf(f, fexpected);

	float f2 = aussie_vector_sum_pointer_arith(v, n);
	ytestf(f2, fexpected);

#if !LINUX
	if (n % 4 == 0) {
		f2 = aussie_vector_sum_AVX1(v, n);
		ytestf(f2, fexpected);
	}
#endif //LINUX

#if !LINUX
	if (n % 8 == 0) {
		f2 = aussie_vector_sum_AVX2(v, n);
		ytestf(f2, fexpected);
	}
#endif //LINUX

}

void aussie_vector_max_test_one(float v[], int n)
{
	float fmax = aussie_vector_max(v, n);

#if !LINUX
	float f2 = aussie_vector_max_AVX1(v, n);
	ytestf(f2, fmax);
#endif //LINUX

#if !LINUX
	f2 = aussie_vector_max_AVX2(v, n);
	ytestf(f2, fmax);
#endif //LINUX

}

void aussie_vector_min_test_one(float v[], int n)
{
	float fmax = aussie_vector_min(v, n);

#if !LINUX
	float f2 = aussie_vector_min_AVX1(v, n);
	ytestf(f2, fmax);
#endif //LINUX

#if !LINUX
    f2 = aussie_vector_min_AVX1b(v, n);
	ytestf(f2, fmax);
	f2 = aussie_vector_min_AVX2b(v, n);
	ytestf(f2, fmax);

	f2 = aussie_vector_min_AVX2(v, n);
	ytestf(f2, fmax);
#endif //LINUX

}


void aussie_vector_max_tests()
{
	float v[1024] = { 0 };

	int n = 128;
	aussie_vector_set_1_N(v, n);
	aussie_vector_max_test_one(v, n);
	aussie_vector_min_test_one(v, n);

	aussie_vector_setall(v, n, -100);
	aussie_vector_max_test_one(v, n);
	aussie_vector_min_test_one(v, n);

	aussie_vector_setall(v, n, -100);
	v[1] = +99;
	aussie_vector_max_test_one(v, n);
	aussie_vector_min_test_one(v, n);

	for (int i = 0; i < n; i++) {
		aussie_vector_setall(v, n, -100);
		v[i] = +99;
		aussie_vector_max_test_one(v, n);
		aussie_vector_min_test_one(v, n);
	}

	for (int i = 0; i < n; i++) {
		aussie_vector_setall(v, n, +100);
		v[i] = +199;
		aussie_vector_max_test_one(v, n);
		aussie_vector_min_test_one(v, n);
	}

	for (int i = 0; i < n; i++) {
		aussie_vector_setall(v, n, +100);
		v[i] = 33;
		aussie_vector_max_test_one(v, n);
		aussie_vector_min_test_one(v, n);
	}

	for (int i = 0; i < n; i++) {
		aussie_vector_setall(v, n, -100);
		v[i] = -222;
		aussie_vector_max_test_one(v, n);
		aussie_vector_min_test_one(v, n);
	}

	aussie_vector_setall(v, n, -100);
	v[33] = +99;
	aussie_vector_max_test_one(v, n);
	aussie_vector_min_test_one(v, n);

}


// Unit testing wrapper
void aussie_yvector_unit_tests()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	aussie_vector_max_tests();

	aussie_yvector_norm_unit_tests(); // Vector Norms L1/L2/etc.
	aussie_vector_topk_tests();   // Top-K
	aussie_softmax_unit_tests();  // Softmax
	aussie_unit_test_normalization();  // Test BatchNorm/LayerNorm/z-score/etc.

	int n = 10;
	float v1[10];
	float v2[10];
	float f = 0.0f;

	aussie_vector_set_1_N(v1, n);
	aussie_vector_set_1_N(v2, n);

	aussie_yvector_test_dot_products(v1, v2, 10, 385.0);

	ytestf(aussie_vector_min(v1, 10), 1.0);
	ytestf(aussie_vector_max(v1, 10), 10.0);
	ytestf(aussie_vector_sum(v1, 10), 55.0);
	ytestf(aussie_vector_sum(v2, 10), 55.0);

	aussie_test_vector_sum(v1, 10, 55.0);
	aussie_test_vector_sum(v2, 10, 55.0);

	// Test negatives (sentinel version)
	ytest(!aussie_vector_has_negative_basic(v1, n-1)); 
	ytest(!aussie_vector_has_negative_sentinel(v1, n-1)); //  Use n-1 to avoid arrays bounds/Valgrind error
	ytest(!aussie_vector_has_negative_sentinel2(v1, n-1));
	ytest(!aussie_vector_has_negative_sentinel3(v1, n - 1));
	ytest(!aussie_vector_has_negative_pointer_arithmetic(v1, n - 1));

	v1[5] = -55.0;  // Now there's a negative..
	ytest(aussie_vector_has_negative_basic(v1, n-1));
	ytest(aussie_vector_has_negative_sentinel(v1, n-1));  //  Use n-1 to avoid arrays bounds/Valgrind error
	ytest(aussie_vector_has_negative_sentinel2(v1, n-1));
	ytest(aussie_vector_has_negative_sentinel3(v1, n - 1));
	ytest(aussie_vector_has_negative_pointer_arithmetic(v1, n - 1));

	// Test reverse N..1
	aussie_vector_set_1_N_reverse(v1, 10);  // Set to 10..1
	ytestf(v1[0], 10.0f);
	ytestf(v1[1], 9.0f);
	ytestf(v1[2], 8.0f);
	aussie_test_vector_sum(v1, 10, 55.0);


	//-------------------------------------------
	// Big vector tests 
	// ... Test aussie_vecdot_parallel_basic()
	//-------------------------------------------
	float v3[10 * 512];
	float v4[10 * 512];
	n = 10 * 512;  // Bigger size...
	float expected = 197021.0f;
	aussie_vector_set_1_N_MAXN(v3, n, 10/*maxn*/);  // Set to 1..N
	aussie_vector_set_1_N_MAXN(v4, n, 10/*maxn*/);  // Set to 1..N
	f = aussie_vecdot_basic(v3, v4, n);
	ytestf(f, expected /*197021.0f*/);  // Sum of squares: 1^2 + 2^2 + ... + 512^2
	aussie_yvector_test_dot_products_BIG(v3, v4, n, expected);

	aussie_test_vector_sum(v3, n, aussie_vector_sum(v3,n));

	n = 9 * 512 + 13;   // odd size...
	expected = aussie_vecdot_basic(v3, v4, n);
	aussie_yvector_test_dot_products_BIG(v3, v4, n, expected);


}


void aussie_memset_wrapper(char* addr, int c, int sz)
{
	if (sz == sizeof(float*)) {
		yassert(sz == sizeof(float*));  // Probable error!
	}
	if (sz == 0) {
		yassert(sz != 0);  // Wrongly reversed parameters?
	}
	memset(addr, c, sz);  // Call the real deal
}
