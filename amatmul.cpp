// ymatmul.cpp -- Matrix multiplication (MatMul/GEMM) -- Aussie AI Base Library  
// Created Oct 22nd 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

//---------------------------------------------------
//---------------------------------------------------

#include "aport.h"
#include "aussieai.h"
#include "aassert.h"
#include "avector.h"
#include "atest.h"
#include "aactivation.h"

#if !LINUX
#include <xmmintrin.h>  // AVX
#include <intrin.h>
#endif //LINUX

#include "amatmul.h"  // self-include

//---------------------------------------------------
int aussie_count_zero(ymatrix m)
{
	int ct = 0;
	for (int i = 0; i < AUSSIE_MATRIX_ROWS; i++) {
		for (int j = 0; j < AUSSIE_MATRIX_COLUMNS; j++) {
			if (m[i][j] == 0.0f) ct++;
		}
	}
	return ct;
}

int aussie_count_nonzero(ymatrix m)
{
	int ct = 0;
	for (int i = 0; i < AUSSIE_MATRIX_ROWS; i++) {
		for (int j = 0; j < AUSSIE_MATRIX_COLUMNS; j++) {
			if (m[i][j] != 0.0f) ct++;
		}
	}
	return ct;
}

void aussie_clear_matrix(ymatrix m)
{
	for (int i = 0; i < AUSSIE_MATRIX_ROWS; i++) {
		for (int j = 0; j < AUSSIE_MATRIX_COLUMNS; j++) {
			m[i][j] = 0.0;
		}
	}
}

void aussie_clear_matrix_n(ymatrix m, int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			m[i][j] = 0.0;
		}
	}
}

void aussie_identity_matrix(ymatrix m, int n)
{
	aussie_clear_matrix_n(m, n);
	for (int i = 0; i < n; i++) {  // Identity diagonal
		m[i][i] = 1.0f;
	}
}


void aussie_clear_matrix_tiled(ymatrix m)
{
	const int TILEX = 4; // 4x4 tile size
	const int TILEY = 4;
	static_assert(AUSSIE_MATRIX_ROWS % TILEX == 0, "Exact X");
	static_assert(AUSSIE_MATRIX_COLUMNS % TILEY == 0, "Exact Y");
	for (int i = 0; i < AUSSIE_MATRIX_ROWS; i += TILEX) {
		for (int j = 0; j < AUSSIE_MATRIX_COLUMNS; j += TILEY) {
			// Do the 2x2 tile...
			for (int tilex = i; tilex < i + TILEX; tilex++) {
				for (int tiley = j; tiley < j + TILEY; tiley++) {
					m[tilex][tiley] = 0.0f;
				}
			}
		}
	}
}

void aussie_set_matrix_1_N_max(ymatrix m, int n, int imax)  // 1..N with rotation at max...
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			++ct;
			m[i][j] = (float)ct;
			if (ct > imax) ct = 1;  // Don't go too high
		}
	}
}

void aussie_set_matrix_1_N(ymatrix m, int n)
{
	int ct = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			++ct;
			m[i][j] = (float)ct;
		}
	}
}


float aussie_sum_matrix(ymatrix m)
{
	float sum = 0.0f;
	for (int i = 0; i < AUSSIE_MATRIX_ROWS; i++) {
		for (int j = 0; j < AUSSIE_MATRIX_COLUMNS; j++) {
			sum += m[i][j];
		}
	}
	return sum;
}

void aussie_matrix_vector_test_one(const ymatrix m, const float v[], int n)
{
	float vtemp[5000];
	float vresult[5000];
	yassert(n < 5000);
	if (n >= 5000) return;

	// Basic result... assume it's correct, test the rest are identical...
	aussie_matmul_vector_basic_out1(m, v, n, vresult);
	aussie_vector_copy_basic(vtemp, vresult, n);  // Save a copy...
	yassert(aussie_vector_equal(vtemp, vresult, n));

	// Test each version does the same thing...
	aussie_matmul_vector_basic_out1(m, v, n, vresult);
	yassert(aussie_vector_equal(vtemp, vresult, n));

	aussie_matmul_vector_basic_out2(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp, n), aussie_vector_sum(vresult, n));
	yassert(aussie_vector_equal(vtemp, vresult, n));

	aussie_matmul_vector_basic_out3(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp, n), aussie_vector_sum(vresult, n));
	yassert(aussie_vector_equal(vtemp, vresult, n));


	aussie_matmul_vector_tiled_2x2(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp, n), aussie_vector_sum(vresult, n));
	yassert(aussie_vector_equal(vtemp, vresult, n));

	aussie_matmul_vector_tiled_4x4(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp, n), aussie_vector_sum(vresult, n));
	yassert(aussie_vector_equal(vtemp, vresult, n));
	

	aussie_matmul_vector_unrolled4(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp,n), aussie_vector_sum(vresult,n));
	yassert(aussie_vector_equal_approx(vtemp, vresult, n, 0.001/*error*/, true/*warn*/));

	aussie_matmul_vector_unrolled8(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp, n), aussie_vector_sum(vresult, n));
	yassert(aussie_vector_equal_approx(vtemp, vresult, n, 0.001/*error*/, true/*warn*/));

	aussie_matmul_vector_unrolled4b(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp, n), aussie_vector_sum(vresult, n));
	yassert(aussie_vector_equal_approx(vtemp, vresult, n, 0.001/*error*/, true/*warn*/));

	// aussie_matmul_vector_unrolled8b
	aussie_matmul_vector_unrolled8b(m, v, n, vresult);
	ytestf(aussie_vector_sum(vtemp, n), aussie_vector_sum(vresult, n));
	yassert(aussie_vector_equal_approx(vtemp, vresult, n, 0.001/*error*/, true/*warn*/));



}

void aussie_matrix_tests_basic()  // Matrix-vector unit tests...
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	static ymatrix m;
	int n = AUSSIE_MATRIX_ROWS;

	aussie_clear_matrix(m);
	ytestf(aussie_sum_matrix(m), 0.0f);
	ytesti(aussie_count_zero(m), AUSSIE_TOTAL_ELEMENTS);
	ytesti(aussie_count_nonzero(m), 0);

	aussie_set_matrix_1_N(m, n);
	if (AUSSIE_TOTAL_ELEMENTS < 1000) ytestf(aussie_sum_matrix(m), AUSSIE_SUM_1_N(AUSSIE_TOTAL_ELEMENTS) );
	ytesti(aussie_count_zero(m), 0);
	ytesti(aussie_count_nonzero(m), AUSSIE_TOTAL_ELEMENTS);


	aussie_set_matrix_1_N(m, n);
	aussie_clear_matrix(m);
	ytestf(aussie_sum_matrix(m), 0.0f);
	ytesti(aussie_count_zero(m), AUSSIE_TOTAL_ELEMENTS);
	ytesti(aussie_count_nonzero(m), 0);

	aussie_set_matrix_1_N(m, n);
	aussie_clear_matrix_tiled(m);
	ytestf(aussie_sum_matrix(m), 0.0f);
	ytesti(aussie_count_zero(m), AUSSIE_TOTAL_ELEMENTS);
	ytesti(aussie_count_nonzero(m), 0);


	// Check the default row-by-row memory layout in C++
	yassert((char*)&m[0][0] == (char*)&m);
	yassert((char*)&m[0][1] == (char*)&m + sizeof(float));
	yassert((char*)&m[0][2] == (char*)&m + 2 * sizeof(float));
	yassert((char*)&m[1][0] == (char*)&m + AUSSIE_MATRIX_COLUMNS * sizeof(float));

	n = AUSSIE_MATRIX_ROWS;
	yvector v1;
	yvector v2;
	yvector vout;
	float fsum = 0.0f;
	aussie_vector_clear(v1, n);
	aussie_vector_clear(v2, n);

	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));
	ytestf(v1[0], 1.0);
	ytestf(v1[9], 10.0);
	float fexpected = AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/);

	aussie_set_matrix_1_N_max(m, n, 5);
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 

	aussie_matrix_vector_test_one(m, v1, n);


	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	aussie_test_vector_sum(v1, n, fexpected);

	aussie_matrix_set_identity(m);  // Set M to identity matrix I

	// This actually still works with the buggy code...
	aussie_matrix_set_identity(m);  // Set M to identity matrix I
	aussie_matrix_vector_test_one(m, v1, n);

	aussie_matmul_vector_basic1_buggy(m, v1, n);  // Multiply vector by identity matrix
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));

	aussie_matmul_vector_basic2_buggy(m, v1, n);  // Multiply vector by identity matrix
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));

	aussie_matrix_set_identity(m);  // Set M to identity matrix I
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	aussie_vector_set_1_N(vout, n);   // Set vector values to 1..N 
	aussie_matrix_vector_test_one(m, v1, n);

	aussie_matmul_vector_basic_out1(m, v1, n, vout);  // Multiply vector by identity matrix
	ytestf(aussie_vector_sum(vout, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));

	aussie_matrix_set_identity(m);  // Set M to identity matrix I
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	aussie_vector_set_1_N(vout, n);   // Set vector values to 1..N 
	aussie_matmul_vector_basic_out2(m, v1, n, vout);  // Multiply vector by identity matrix
	ytestf(aussie_vector_sum(vout, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));

	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Vector is { 1..N }
	aussie_vector_set_1_N(vout, n);   // Set vector values to 1..N 
	aussie_matmul_vector_basic1_buggy(m, v1, n);  // Multiply vector by zero matrix
	ytestf(aussie_vector_sum(v1, n), 0.0f);
	aussie_test_vector_sum(v1, n, 0.0f);

	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Vector is { 1..N }
	aussie_matmul_vector_basic2_buggy(m, v1, n);  // Multiply vector by zero matrix
	ytestf(aussie_vector_sum(v1, n), 0.0f);
	aussie_test_vector_sum(v1, n, 0.0f);

	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Vector is { 1..N }
	aussie_matmul_vector_basic_out1(m, v1, n, vout);  // Multiply vector by zero matrix
	ytestf(aussie_vector_sum(vout, n), 0.0f);
	aussie_test_vector_sum(vout, n, 0.0f);

	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Vector is { 1..N }
	aussie_matmul_vector_basic_out2(m, v1, n, vout);  // Multiply vector by zero matrix
	ytestf(aussie_vector_sum(vout, n), 0.0f);
	aussie_test_vector_sum(vout, n, 0.0f);


	// Test to detect problem with output vector modified too early...
	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	if (n==12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));
	for (int i = 0; i < n; i++) m[i][0] = 1.0f;  // Left column all 1's...
	ytestf(m[0][0], 1.0f);
	m[0][0] = 0.0f;  // Make first row all-zeros...
	ytestf(v1[0], 1.0f);
	ytestf(v1[1], 2.0f);
	aussie_matrix_vector_test_one(m, v1, n);
	aussie_matmul_vector_basic1_buggy(m, v1, n);  // Multiply vector by zero matrix
#if 0 // Buggy!!!
	ytest(aussie_vector_sum(v1, n) != 0.0f);  // BUG if zero!!
	ytestf(v1[0], 0.0f);
	ytest(v1[1] != 0.0f);
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));
#endif

	// Test to detect problem with output vector modified too early...
	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));
	for (int i = 0; i < n; i++) m[i][0] = 1.0f;  // Left column all 1's...
	ytestf(m[0][0], 1.0f);
	m[0][0] = 0.0f;  // Make first row all-zeros...
	ytestf(v1[0], 1.0f);
	ytestf(v1[1], 2.0f);
	fsum = aussie_vector_sum(v1, n);
	aussie_matrix_vector_test_one(m, v1, n);
	aussie_matmul_vector_basic_out2(m, v1, n, vout);  // Multiply vector by zero matrix
	ytestf(aussie_vector_sum(v1, n), fsum);  // Should be unchanged
#if 1 // Set to 0 if buggy!!!
	ytest(aussie_vector_sum(v1, n) != 0.0f);  // BUG if zero!!
	ytestf(vout[0], 0.0f);
	ytest(vout[1] != 0.0f);
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));
#endif


	// Test to detect problem with output vector modified too early...
	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));
	for (int i = 0; i < n; i++) m[i][0] = 1.0f;  // Left column all 1's...
	ytestf(m[0][0], 1.0f);
	m[0][0] = 0.0f;  // Make first row all-zeros...
	ytestf(v1[0], 1.0f);
	ytestf(v1[1], 2.0f);
	aussie_matrix_vector_test_one(m, v1, n);
	fsum = aussie_vector_sum(v1, n);
	aussie_matmul_vector_basic_out1(m, v1, n, vout);  // Multiply vector by zero matrix
	ytestf(aussie_vector_sum(v1, n), fsum);  // Should be unchanged
#if 1 // Set to 0 if buggy!!!
	ytest(aussie_vector_sum(v1, n) != 0.0f);  // BUG if zero!!
	ytestf(vout[0], 0.0f);
	ytest(vout[1] != 0.0f);
	ytestf(aussie_vector_sum(v1, n), AUSSIE_SUM_1_N(AUSSIE_MATRIX_COLUMNS/*55.0*/));
#endif

	// Test correctly that it's not buggy...
	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	for (int i = 0; i < n; i++) m[i][0] = 1.0f;  // Left column all 1's...
	ytestf(m[0][0], 1.0f);
	m[0][0] = 0.0f;  // Make first row all-zeros...
	aussie_matrix_vector_test_one(m, v1, n);
	fsum = aussie_vector_sum(v1, n);
	aussie_matmul_vector_basic_out1(m, v1, n, vout);    // M x V1 -> VOUT (3 operands matmul)
	ytestf(aussie_vector_sum(v1, n), fsum);  // Should be unchanged
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);  // Unchanged
	ytest(aussie_vector_sum(vout, n) != 0.0f);  // BUG if zero!!
	ytestf(vout[0], 0.0f);
	ytestf(vout[1], 1.0f);
	ytestf(aussie_vector_sum(vout, n), (n - 1) * 1.0f);  // All 1's...except first element

	// Test correctly that it's not buggy...
	aussie_clear_matrix(m);  // Zero matrix...
	aussie_vector_set_1_N(v1, n);   // Set vector values to 1..N 
	if (n == 12) ytestf(aussie_vector_sum(v1, n), 78.0f);
	for (int i = 0; i < n; i++) m[i][0] = 1.0f;  // Left column all 1's...
	ytestf(m[0][0], 1.0f);
	m[0][0] = 0.0f;  // Make first row all-zeros...
	aussie_matmul_vector_basic_out2(m, v1, n, vout);    // M x V1 -> VOUT (3 operands matmul)
	float fsum2 = aussie_vector_sum(vout, n);
	ytestf(fsum2, (n - 1) * 1.0f); 
	ytest(aussie_vector_sum(vout, n) != 0.0f);  // BUG if zero!!
	ytestf(vout[0], 0.0f);
	ytestf(vout[1], 1.0f);
	ytestf(aussie_vector_sum(vout, n), (n - 1) * 1.0f);  // All 1's...except first element


	aussie_clear_matrix(m);  // Zero matrix...


}

void aussie_matmul_vector_basic1_buggy(ymatrix m, float v[], int n)
{
	// Basic matrix-by-vector using vector dot products..
	// This is buggy!! 
	// Changing v[] during computations messes up the subsequent ones!
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int i = 0; i < n; i++) {
		float* rowvector = &m[i][0];
		float sum = aussie_vecdot_basic(rowvector, v, n);  // Dot product
		v[i] = sum;
	}
}


void aussie_matmul_vector_basic_out1(const ymatrix m, const float v[], int n, float vout[])
{
	// Basic matrix-by-vector using vector dot products..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int i = 0; i < n; i++) {
		const float* rowvector = &m[i][0];
		float sum = aussie_vecdot_basic(rowvector, v, n);  // Dot product
		vout[i] = sum;
	}
}


void aussie_matmul_vector_basic_vecdot_RELU_fused(const ymatrix m, const float v[], int n, float vout[])
{
	// Basic fused matrix-by-vector RELU using vector dot products..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int i = 0; i < n; i++) {
		const float* rowvector = &m[i][0];
		float sum = aussie_vecdot_basic(rowvector, v, n);  // Dot product
		vout[i] = AUSSIE_RELU_MACRO(sum);
	}
}

void aussie_VMM_vector_basic_vecdot_nonfused(const ymatrix m, const float v[], int n, float vout[])
{
	// Basic matrix-by-vector using vector dot products..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int i = 0; i < n; i++) {
		const float* rowvector = &m[i][0];
		float sum = aussie_vecdot_basic(rowvector, v, n);  // Dot product
		vout[i] = sum;
	}
}


void aussie_VMM_vector_vecdot_AVX1(const ymatrix m, const float v[], int n, float vout[])
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// VMM matrix-by-vector using AVX1 vector dot product
	for (int i = 0; i < n; i++) {
		const float* rowvector = &m[i][0];
		float sum = aussie_vecdot_unroll_AVX1(rowvector, v, n);  // Dot product
		vout[i] = sum;
	}
#endif //LINUX
}

void aussie_VMM_vecdot_RELU_AVX1(const ymatrix m, const float v[], int n, float vout[])
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	aussie_VMM_vector_vecdot_AVX1(m, v, n, vout);  // Matrix-vector multiply
	aussie_vector_reluize_AVX1(vout, n);  // Apply RELU on the output
#endif //LINUX
}

void aussie_matmul_vector_basic_vecdot_RELU_nonfused(const ymatrix m, const float v[], int n, float vout[])
{
	aussie_VMM_vector_basic_vecdot_nonfused(m, v, n, vout);  // Matrix-vector multiply
	aussie_vector_reluize(vout, n);  // Apply RELU on the output
}

void aussie_matmul_vector_basic2_buggy(ymatrix m, float v[], int n)
{
	// Basic matrix-by-vector using vector dot products..
	// This is buggy!! 
	// Changing v[] during computations messes up the subsequent ones!
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int row = 0; row < n; row++) {
		float sum = 0.0f;
		for (int col = 0; col < n; col++) {
			sum += (m[row][col] * v[col]);
		}
		v[row] = sum;
	}
}


void aussie_matmul_vector_basic_out2_rowwise(const ymatrix m, const float v[], int n, float vout[])
{
	// Hoist rowvector initialization out of the loop..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int row = 0; row < n; row++) {
		const float* rowvector = &m[row][0];   // Hoist m[row]
		float sum = 0.0f;
		for (int col = 0; col < n; col++) {
			sum += (rowvector[col] * v[col]);
		}
		vout[row] = sum;
	}
}

void aussie_matmul_vector_basic_out2_pointer_arith(const ymatrix m, const float v[], int n, float vout[])
{
	// Pointer Arithmetic matrix-by-vector ..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	float* rowvector = (float*) & m[0][0];
	const float* endrow = rowvector + (n * n);
	const float* endcol = v + n;
	for (; rowvector != endrow; /*rowvector += n*/) {
		float sum = 0.0f;
		for (const float* vcolumn = v; vcolumn != endcol; vcolumn++,rowvector++) {
			sum += ( (*rowvector) * (*vcolumn) );
		}
		*vout++ = sum;
	}
}

void aussie_matmul_vector_vecdot_AVX1(const ymatrix m, const float v[], int n, float vout[])
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Matrix-by-vector using AVX1 vector dot products..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int i = 0; i < n; i++) {
		const float* rowvector = &m[i][0];
		float sum = aussie_vecdot_unroll_AVX1(rowvector, v, n);  // Dot product
		vout[i] = sum;
	}
#endif //LINUX
}




void aussie_matmul_vector_vecdot_AVX2(const ymatrix m, const float v[], int n, float vout[])
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// Matrix-by-vector using AVX2 vector dot products with AVX2 FMA intrinsic.
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int i = 0; i < n; i++) {
		const float* rowvector = &m[i][0];
		// NOTE: Use FMA not DP version: aussie_vecdot_FMA_unroll_AVX2 not aussie_vecdot_unroll_AVX2
		float sum = aussie_vecdot_FMA_unroll_AVX2(rowvector, v, n);  // Dot product
		vout[i] = sum;
	}
#endif //LINUX
}



void aussie_matmul_vector_basic_out2(const ymatrix m, const float v[], int n, float vout[])
{
	// Basic matrix-by-vector using nested loops..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int row = 0; row < n; row++) {
		float sum = 0.0f;
		for (int col = 0; col < n; col++) {
			sum += (m[row][col] * v[col]);
		}
		vout[row] = sum;
	}
}

void aussie_matmul_vector_basic_out3(const ymatrix m, const float v[], int n, float vout[])
{
	// Basic matrix-by-vector using nested loops..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int row = 0; row < n; row++) {
		vout[row] = 0.0f;
		for (int col = 0; col < n; col++) {
			vout[row] += (m[row][col] * v[col]);
		}
	}
}

void aussie_matmul_vector_basic_interchange(const ymatrix m, const float v[], int n, float vout[])
{
	// Interchanged loops without any hoisting -- matrix-by-vector using nested loops..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	memset(vout, 0, sizeof(float) * n);  // Zero vout[]
	for (int col = 0; col < n; col++) {
		for (int row = 0; row < n; row++) {
			vout[row] += (m[row][col] * v[col]);
		}
	}
}

void aussie_matmul_vector_hoisted_interchange(const ymatrix m, const float v[], int n, float vout[])
{
	// Interchanged loops without v[col] hoisted -- matrix-by-vector using nested loops..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	memset(vout, 0, sizeof(float) * n);  // Zero vout[]
	for (int col = 0; col < n; col++) {
		float fcol = v[col];
		for (int row = 0; row < n; row++) {
			vout[row] += (m[row][col] * fcol);
		}
	}
}



void aussie_matmul_vector_tiled_2x2(const ymatrix m, const float v[], int n, float vout[])
{
	// Tiled/blocked matrix-by-vector using 2x2 tiling..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 2 == 0);
	for (int row = 0; row < n; row += 2) {
		vout[row] = 0.0f;
		vout[row + 1] = 0.0f;
		for (int col = 0; col < n; col += 2) {
			vout[row] += (m[row][col] * v[col]); // row+0, col + 0
			vout[row] += (m[row][col + 1] * v[col + 1]); // row+0, col + 1
			vout[row + 1] += (m[row + 1][col] * v[col]); // row+1, col + 0
			vout[row + 1] += (m[row + 1][col + 1] * v[col + 1]); // row+1, col + 1
		}
	}
}

void aussie_matmul_vector_tiled_2x2_better(const ymatrix m, const float v[], int n, float vout[])
{
	// Tiled/blocked matrix-by-vector using 2x2 tiling.. (faster sub-kernel)
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 2 == 0);
	for (int row = 0; row < n; row += 2) {
		vout[row] = 0.0f;
		vout[row + 1] = 0.0f;
		for (int col = 0; col < n; col += 2) {
			vout[row] += (m[row][col] * v[col])  // row+0, col + 0
			    + (m[row][col + 1] * v[col + 1]) // row+0, col + 1
				;
			vout[row + 1] += (m[row + 1][col] * v[col]) // row+1, col + 0
				+ (m[row + 1][col + 1] * v[col + 1])    // row+1, col + 1
				; 
		}
	}
}


void aussie_matmul_vector_tiled_2x2_better_hoisted(const ymatrix m, const float v[], int n, float vout[])
{
	// Tiled/blocked matrix-by-vector using 2x2 tiling.. (faster sub-kernel)
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 2 == 0);
	for (int row = 0; row < n; row += 2) {
		vout[row] = 0.0f;
		vout[row + 1] = 0.0f;
		for (int col = 0; col < n; col += 2) {
			float fcol0 = v[col];
			float fcol1 = v[col + 1];
			vout[row] += (m[row][col] * fcol0)  // row+0, col + 0
				+ (m[row][col + 1] * fcol1) // row+0, col + 1
				;
			vout[row + 1] += (m[row + 1][col] * fcol0) // row+1, col + 0
				+ (m[row + 1][col + 1] * fcol1)    // row+1, col + 1
				;
		}
	}
}



void aussie_matmul_vector_tiled_4x4(const ymatrix m, const float v[], int n, float vout[])
{
	// Tiled/blocked matrix-by-vector using 4x4 tiling..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 4 == 0);
	memset(vout, 0, sizeof(float) * n);
	for (int row = 0; row < n; row += 4) {
		//vout[row] = 0.0f;
		//vout[row + 1] = 0.0f;
		//vout[row + 2] = 0.0f;
		//vout[row + 3] = 0.0f;
		for (int col = 0; col < n; col += 4) {
			vout[row] +=
				(m[row][col] * v[col]) // row+0, col + 0
				+ (m[row][col + 1] * v[col + 1]) // row+0, col + 1
				+ (m[row][col + 2] * v[col + 2]) // row+0, col + 2
				+ (m[row][col + 3] * v[col + 3]) // row+0, col + 3
				;
			vout[row + 1] +=
				(m[row+1][col] * v[col]) // row+1, col + 0
				+ (m[row + 1][col + 1] * v[col + 1]) // row+1, col + 1
				+ (m[row + 1][col + 2] * v[col + 2]) // row+1, col + 2
				+ (m[row + 1][col + 3] * v[col + 3]) // row+1, col + 3
				;
			vout[row + 2] +=
				(m[row + 2][col] * v[col]) // row+2, col + 0
				+ (m[row + 2][col + 1] * v[col + 1]) // row+2, col + 1
				+ (m[row + 2][col + 2] * v[col + 2]) // row+2, col + 2
				+ (m[row + 2][col + 3] * v[col + 3]) // row+2, col + 3
				;
			vout[row + 3] +=
				(m[row + 3][col] * v[col]) // row+3, col + 0
				+ (m[row + 3][col + 1] * v[col + 1]) // row+3, col + 1
				+ (m[row + 3][col + 2] * v[col + 2]) // row+3, col + 2
				+ (m[row + 3][col + 3] * v[col + 3]) // row+3, col + 3
				;
		}
	}
}

void aussie_matmul_vector_tiled_4x4_CSE(const ymatrix m, const float v[], int n, float vout[])
{
	// Tiled/blocked matrix-by-vector using 4x4 tiling..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 4 == 0);
	memset(vout, 0, sizeof(float) * n);
	for (int row = 0; row < n; row += 4) {
		//vout[row] = 0.0f;
		//vout[row + 1] = 0.0f;
		//vout[row + 2] = 0.0f;
		//vout[row + 3] = 0.0f;
		for (int col = 0; col < n; col += 4) {
			float fcol0 = v[col];
			float fcol1 = v[col + 1];
			float fcol2 = v[col + 2];
			float fcol3 = v[col + 3];
			vout[row] +=
				(m[row][col] * fcol0) // row+0, col + 0
				+ (m[row][col + 1] * fcol1) // row+0, col + 1
				+ (m[row][col + 2] * fcol2) // row+0, col + 2
				+ (m[row][col + 3] * fcol3) // row+0, col + 3
				;
			vout[row + 1] +=
				(m[row + 1][col] * fcol0) // row+1, col + 0
				+ (m[row + 1][col + 1] * fcol1) // row+1, col + 1
				+ (m[row + 1][col + 2] * fcol2) // row+1, col + 2
				+ (m[row + 1][col + 3] * fcol3) // row+1, col + 3
				;
			vout[row + 2] +=
				(m[row + 2][col] * fcol0) // row+2, col + 0
				+ (m[row + 2][col + 1] * fcol1) // row+2, col + 1
				+ (m[row + 2][col + 2] * fcol2) // row+2, col + 2
				+ (m[row + 2][col + 3] * fcol3) // row+2, col + 3
				;
			vout[row + 3] +=
				(m[row + 3][col] * fcol0) // row+3, col + 0
				+ (m[row + 3][col + 1] * fcol1) // row+3, col + 1
				+ (m[row + 3][col + 2] * fcol2) // row+3, col + 2
				+ (m[row + 3][col + 3] * fcol3) // row+3, col + 3
				;
		}
	}
}

void aussie_matmul_vector_tiled_4x4_CSE2(const ymatrix m, const float v[], int n, float vout[])
{
	// Tiled/blocked matrix-by-vector using 4x4 tiling..
	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 4 == 0);
	memset(vout, 0, sizeof(float) * n);
	for (int row = 0; row < n; row += 4) {
		//vout[row] = 0.0f;
		//vout[row + 1] = 0.0f;
		//vout[row + 2] = 0.0f;
		//vout[row + 3] = 0.0f;
		const float* rowvec = &m[row][0];
		const float* rowvec1 = &m[row + 1][0];
		const float* rowvec2 = &m[row + 2][0];
		const float* rowvec3 = &m[row + 3][0];
		for (int col = 0; col < n; col += 4) {
			float fcol0 = v[col];
			float fcol1 = v[col + 1];
			float fcol2 = v[col + 2];
			float fcol3 = v[col + 3];
			vout[row] +=
				(rowvec[col] * fcol0) // row+0, col + 0
				+ (rowvec[col + 1] * fcol1) // row+0, col + 1
				+ (rowvec[col + 2] * fcol2) // row+0, col + 2
				+ (rowvec[col + 3] * fcol3) // row+0, col + 3
				;
			vout[row + 1] +=
				(rowvec1[col] * fcol0) // row+1, col + 0
				+ (rowvec1[col + 1] * fcol1) // row+1, col + 1
				+ (rowvec1[col + 2] * fcol2) // row+1, col + 2
				+ (rowvec1[col + 3] * fcol3) // row+1, col + 3
				;
			vout[row + 2] +=
				(rowvec2[col] * fcol0) // row+2, col + 0
				+ (rowvec2[col + 1] * fcol1) // row+2, col + 1
				+ (rowvec2[col + 2] * fcol2) // row+2, col + 2
				+ (rowvec2[col + 3] * fcol3) // row+2, col + 3
				;
			vout[row + 3] +=
				(rowvec3[col] * fcol0) // row+3, col + 0
				+ (rowvec3[col + 1] * fcol1) // row+3, col + 1
				+ (rowvec3[col + 2] * fcol2) // row+3, col + 2
				+ (rowvec3[col + 3] * fcol3) // row+3, col + 3
				;
		}
	}
}

void aussie_matmul_vector_unrolled4(const ymatrix m, const float v[], int n, float vout[])
{
	// Unrolled inner-loop matrix-by-vector multiply..
	//yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 4 == 0);
	for (int row = 0; row < n; row++) {
		vout[row] = 0.0f;
		for (int col = 0; col < n; col+= 4) {
			vout[row] += 
				  (m[row][col] * v[col])
				+ (m[row][col+1] * v[col+1])
				+ (m[row][col+2] * v[col+2])
				+ (m[row][col+3] * v[col+3])
				;
		}
	}

}


void aussie_matmul_vector_unrolled4b(const ymatrix m, const float v[], int n, float vout[])
{
	// Unrolled inner-loop matrix-by-vector multiply..
	//yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 4 == 0);
	for (int row = 0; row < n; row++) {
		const float* rowvec = &m[row][0];
		vout[row] = 0.0f;
		for (int col = 0; col < n; /*col += 4*/) {
			vout[row] += (rowvec[col] * v[col]); col++;
			vout[row] += (rowvec[col] * v[col]); col++;
			vout[row] += (rowvec[col] * v[col]); col++;
			vout[row] += (rowvec[col] * v[col]); col++;
		}
	}

}

void aussie_matmul_vector_unrolled8(const ymatrix m, const float v[], int n, float vout[])
{
	// Unrolled inner-loop matrix-by-vector multiply..
	// Hoist m[row] out of the loop...
	//yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 8 == 0);
	for (int row = 0; row < n; row++) {
		const float* rowvec = &m[row][0];
		vout[row] = 0.0f;
		for (int col = 0; col < n; col += 8) {
			vout[row] +=
				(rowvec[col] * v[col])
			+(rowvec[col + 1] * v[col + 1])
			+(rowvec[col + 2] * v[col + 2])
			+(rowvec[col + 3] * v[col + 3])
			+(rowvec[col + 4] * v[col + 4])
			+(rowvec[col + 5] * v[col + 5])
			+(rowvec[col + 6] * v[col + 6])
			+(rowvec[col + 7] * v[col + 7])
				;
		}
	}
}


void aussie_matmul_vector_unrolled8b(const ymatrix m, const float v[], int n, float vout[])
{
	// Unrolled inner-loop matrix-by-vector multiply..
	//yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 8 == 0);
	for (int row = 0; row < n; row++) {
		vout[row] = 0.0f;
		for (int col = 0; col < n; col += 8) {
			vout[row] +=
				(m[row][col] * v[col])
			+(m[row][col + 1] * v[col + 1])
			+(m[row][col + 2] * v[col + 2])
			+(m[row][col + 3] * v[col + 3])
			+(m[row][col + 4] * v[col + 4])
			+(m[row][col + 5] * v[col + 5])
			+(m[row][col + 6] * v[col + 6])
			+(m[row][col + 7] * v[col + 7])
				;
		}
	}

}

void aussie_matrix_set_identity(ymatrix m)
{
	// Set a matrix to the identity matrix
	aussie_clear_matrix(m);
	for (int i = 0; i < AUSSIE_MATRIX_ROWS; i++) {
		m[i][i] = 1.0;  // Put 1's in the diagonal...
	}

}


//---------------------------------------------------
//---------------------------------------------------

void aussie_matmul_matrix_basic(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// Matrix-Matrix multiplication basic naive n^3 algorithm...

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			float sum = 0.0f;
			for (int k = 0; k < n; k++) {
				sum += (m1[row][k] * m2[k][col]);
			}
			mout[row][col] = sum;
		}
	}
}


void aussie_matmul_matrix_hoisted(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// Matrix-Matrix multiplication basic naive n^3 algorithm...

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			float sum = 0.0f;
			for (int k = 0; k < n; k++) {
				sum += (rowvec[k] * m2[k][col]);
			}
			mout[row][col] = sum;
		}
	}
}


void aussie_matmul_matrix_unrolled4(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// Matrix-Matrix multiplication basic naive n^3 algorithm...

	//yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 4 == 0);

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			float sum = 0.0f;
			for (int k = 0; k < n; k += 4) {
				sum += 
					(rowvec[k] * m2[k][col])
					+ (rowvec[k + 1] * m2[k + 1][col])
					+ (rowvec[k + 2] * m2[k + 2][col])
					+ (rowvec[k + 3] * m2[k + 3][col])
					;
			}
			mout[row][col] = sum;
		}
	}
}

void aussie_matmul_matrix_fake_transpose(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// Matrix-Matrix naive n^3 algorithm on a TRANSPOSE...

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			float sum = 0.0f;
			const float* colvec = &m2[col][0];
			for (int k = 0; k < n; k++) {
				sum += (rowvec[k] * colvec[k]);
			}
			mout[row][col] = sum;
		}
	}
}


void aussie_matmul_matrix_fake_transpose_unrolled4(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// Transpose Matrix-Matrix multiplication with 4 iteration unrolling...

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 4 == 0);

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			float sum = 0.0f;
			const float* colvec = &m2[col][0];
			for (int k = 0; k < n; k += 4) {
				sum += (rowvec[k] * colvec[k])
					+ (rowvec[k + 1] * colvec[k + 1])
					+ (rowvec[k + 2] * colvec[k + 2])
					+ (rowvec[k + 3] * colvec[k + 3])
					;
			}
			mout[row][col] = sum;
		}
	}
}


void aussie_matmul_matrix_fake_transpose_unrolled8(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// Transpose Matrix-Matrix multiplication with 8 iteration unrolling...

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 8 == 0);

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			float sum = 0.0f;
			const float* colvec = &m2[col][0];
			for (int k = 0; k < n; k += 8) {
				sum += (rowvec[k] * colvec[k])
					+ (rowvec[k + 1] * colvec[k + 1])
					+ (rowvec[k + 2] * colvec[k + 2])
					+ (rowvec[k + 3] * colvec[k + 3])
					+ (rowvec[k + 4] * colvec[k + 4])
					+ (rowvec[k + 5] * colvec[k + 5])
					+ (rowvec[k + 6] * colvec[k + 6])
					+ (rowvec[k + 7] * colvec[k + 7])
					;
			}
			mout[row][col] = sum;
		}
	}
}

void aussie_matmul_matrix_fake_transpose_vecdot_AVX1_inlined(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// AVX1 Matrix-Matrix multiplication 
	yassert(n % 8 == 0);
	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			const float* colvec = &m2[col][0];
			float sum = 0.0;
			for (int i = 0; i < n; i += 4) {
				// AVX1: Vector dot product of 2 vectors 
				//  ... process 4x32-bit floats in 128 bits
				__m128 r1 = _mm_loadu_ps(&rowvec[i]);   // Load floats into 128-bits
				__m128 r2 = _mm_loadu_ps(&colvec[i]);
				__m128 dst = _mm_dp_ps(r1, r2, 0xf1); // Dot product
				sum += _mm_cvtss_f32(dst);
			}
			mout[row][col] = sum;
		}
	}
#endif //LINUX
}


void aussie_matmul_matrix_fake_transpose_vecdot_AVX1_inlined_unrolled4(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// AVX1 Matrix-Matrix multiplication 
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	yassert(n % 16 == 0);
	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			const float* colvec = &m2[col][0];
			float sum = 0.0;
			for (int i = 0; i < n; i += 16) {
				// AVX1: Vector dot product of 2 vectors 
				//  ... process 4x32-bit floats in 128 bits
				__m128 r1 = _mm_loadu_ps(&rowvec[i]);   // Load floats into 128-bits
				__m128 r2 = _mm_loadu_ps(&colvec[i]);
				__m128 dst = _mm_dp_ps(r1, r2, 0xf1); // Dot product
				sum += _mm_cvtss_f32(dst);

				r1 = _mm_loadu_ps(&rowvec[i+4]);   // Load floats into 128-bits
				r2 = _mm_loadu_ps(&colvec[i+4]);
				dst = _mm_dp_ps(r1, r2, 0xf1); // Dot product
				sum += _mm_cvtss_f32(dst);

				r1 = _mm_loadu_ps(&rowvec[i+8]);   // Load floats into 128-bits
				r2 = _mm_loadu_ps(&colvec[i+8]);
				dst = _mm_dp_ps(r1, r2, 0xf1); // Dot product
				sum += _mm_cvtss_f32(dst);

				r1 = _mm_loadu_ps(&rowvec[i+12]);   // Load floats into 128-bits
				r2 = _mm_loadu_ps(&colvec[i+12]);
				dst = _mm_dp_ps(r1, r2, 0xf1); // Dot product
				sum += _mm_cvtss_f32(dst);
			}
			mout[row][col] = sum;
		}
	}
#endif //LINUX
}


void aussie_matmul_matrix_fake_transpose_vecdot_AVX1(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return;
#else
	// AVX1 Matrix-Matrix multiplication 

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 8 == 0);

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			const float* colvec = &m2[col][0];
			mout[row][col] = aussie_vecdot_unroll_AVX1(rowvec, colvec, n);
		}
	}
#endif //LINUX
}


void aussie_matmul_matrix_fake_transpose_vecdot_AVX2_inlined(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// AVX2 Matrix-Matrix multiplication .

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 8 == 0);

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			const float* colvec = &m2[col][0];
			__m256 sumdst = _mm256_setzero_ps();   // Set accumulators to zero
			for (int i = 0; i < n; i += 8) {
				// AVX2: Vector dot product of 2 vectors 
				//  ... process 4x32-bit floats in 128 bits
				__m256 r1 = _mm256_loadu_ps(&rowvec[i]);   // Load floats into 256-bits
				__m256 r2 = _mm256_loadu_ps(&colvec[i]);
				sumdst = _mm256_fmadd_ps(r1, r2, sumdst); // FMA of 3 vectors
			}
			// Add the final 8 accumulators manually
			float* farr = (float*)&sumdst;
			float sum = farr[0] + farr[1] + farr[2] + farr[3]
				+ farr[4] + farr[5] + farr[6] + farr[7];
			mout[row][col] = sum;
		}
	}
#endif //LINUX
}


void aussie_matmul_matrix_fake_transpose_vecdot_AVX2(const ymatrix m1, const ymatrix m2, int n, ymatrix mout)
{
	// AVX2 Matrix-Matrix multiplication .

	yassert(n == AUSSIE_MATRIX_ROWS);  // Matrix & vector dimensions must match
	yassert(n % 8 == 0);

	for (int row = 0; row < n; row++) {
		const float* rowvec = &m1[row][0];
		for (int col = 0; col < n; col++) {
			const float* colvec = &m2[col][0];
			mout[row][col] = aussie_vecdot_FMA_unroll_AVX2(rowvec, colvec, n);
		}
	}
}

void aussie_matrix_transpose_basic(const ymatrix m1, int n, ymatrix transpose)
{
	// Transpose: put the transposed matrix into the output matrix (square matrix)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			transpose[j][i] = m1[i][j];
		}
	}
}

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

