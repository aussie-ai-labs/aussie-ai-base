// ynorms.cpp -- Vector Norms (not Normalization) -- Aussie AI Base Library  
// Created 12th Nov 2023
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
#include "afloat.h"

#include "anorms.h"  // self-include

//---------------------------------------------------


float aussie_vector_L1_norm(float v[], int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		sum += fabsf(v[i]);
	}
	return sum;
}

float aussie_vector_L1_norm_if_test(float v[], int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		if (v[i] < 0.0) sum += -v[i];
		else sum += v[i];
	}
	return sum;
}

float aussie_vector_L1_norm_bitwise_fabs(float v[], int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		unsigned int u = (AUSSIE_FLOAT_TO_UINT((v[i])) & ~(1u << 31u));  // Clear sign bit...
		sum += AUSSIE_UINT_TO_FLOAT(u);  // Convert back to float & add.

	}
	return sum;
}


float aussie_vector_L2_norm(float v[], int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		sum += (v[i] * v[i]);
	}
	return sqrtf(sum);
}

float aussie_vector_L2_squared_norm(float v[], int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		sum += (v[i] * v[i]);
	}
	return sum;  // NOT sqrtf(sum);
}


float aussie_vector_L3_norm(float v[], int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n; i++) {
		sum += (v[i] * v[i] * v[i]);
		yassert(sum >= 0);
	}
	const float frac_third = 1.0f / 3.0f;
	yassert(sum >= 0);
	return powf(sum, frac_third);
}


//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------
void aussie_yvector_norm_unit_tests()  // Test NORMs
{

	int n = 16;
	alignas(32) float v1[100];
	alignas(32) float v2[100];
	yassert(n < 100);

	aussie_vector_zero(v1, n);
	aussie_vector_zero(v2, n);

	ytestf(aussie_vector_L1_norm(v1, n), 0.0f);
	ytestf(aussie_vector_L1_norm(v2, n), 0.0f);
	ytestf(aussie_vector_L2_norm(v1, n), 0.0f);
	ytestf(aussie_vector_L2_norm(v2, n), 0.0f);
	ytestf(aussie_vector_L3_norm(v1, n), 0.0f);
	ytestf(aussie_vector_L3_norm(v2, n), 0.0f);

	float f = aussie_vecdot_basic(v1, v2, n);

	ytest(f == 0.0);

	aussie_vector_set_1_N(v1, n);  // Set to 1..N
	aussie_vector_set_1_N_reverse(v2, n);  // Set to N..1

	float vecdotexpected = aussie_vecdot_basic(v1, v2, n);
	aussie_yvector_test_dot_products(v1, v2, n, vecdotexpected);

	n = 10;
	aussie_vector_set_1_N(v1, n);  // Set to 1..10
	aussie_vector_set_1_N(v2, n);  // Set to 1..10
	f = aussie_vecdot_basic(v1, v2, n);
	ytest(f == 385.0);  // Sum of squares: 1^2 + 2^2 + ... + 10^2

	n = 10;
	ytestf(aussie_vector_L1_norm(v1, n), 55.0f);
	ytestfapprox(aussie_vector_L2_norm(v1, n), 19.621416, 0.001f);
	ytestfapprox(aussie_vector_L2_squared_norm(v1, n), 19.621416 * 19.62146, 0.001f);
	ytestfapprox(aussie_vector_L3_norm(v1, n), 14.462f, 0.001f);

}
//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

