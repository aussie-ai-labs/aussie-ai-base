// amatmul.h -- Matrix multiplication (MatMul/GEMM) -- Aussie AI Base Library  
// Created Oct 22nd 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YMATMUL_INCLUDE_HEADER_H
#define AUSSIE_YMATMUL_INCLUDE_HEADER_H



//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// MatMul basic types...
//-------------------------------------------------------------------------
#define AUSSIE_MATRIX_ROWS 2048 // 12
#define AUSSIE_MATRIX_COLUMNS 2048 //  12
#define AUSSIE_TOTAL_ELEMENTS (AUSSIE_MATRIX_ROWS * AUSSIE_MATRIX_COLUMNS) 
#define AUSSIE_SUM_1_N(N) ( (N * (N + 1)) / 2.0 ) 
typedef float ymatrix[AUSSIE_MATRIX_ROWS][AUSSIE_MATRIX_COLUMNS];
typedef float yvector[AUSSIE_MATRIX_ROWS];

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------


//-------------------------------------------------------------------------
// MatMul basic APIs...
//-------------------------------------------------------------------------
void aussie_clear_matrix(ymatrix m);
void aussie_clear_matrix_n(ymatrix m, int n);
void aussie_clear_matrix_tiled(ymatrix m);  // Tiling/blocking version of clear matrix

void aussie_identity_matrix(ymatrix m, int n);  // Diagonal of 1s
void aussie_matrix_set_identity(ymatrix m);

void aussie_set_matrix_1_N(ymatrix m, int n);
void aussie_set_matrix_1_N_max(ymatrix m, int n, int imax);  // 1..N with rotation at max...

//-------------------------------------------------------------------------
// Counting matrix elements
//-------------------------------------------------------------------------
int aussie_count_zero(ymatrix m);
int aussie_count_nonzero(ymatrix m);

//-------------------------------------------------------------------------
// Matrix Reductions (matrix to float)
//-------------------------------------------------------------------------
float aussie_sum_matrix(ymatrix m);

//-------------------------------------------------------------------------
// Unit tests of MatMul
//-------------------------------------------------------------------------
void aussie_matrix_tests_basic();

//-------------------------------------------------------------------------
// MATRIX-MATRIX MULTIPLY
//-------------------------------------------------------------------------
void aussie_matmul_matrix_basic(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_hoisted(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_fake_transpose(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_unrolled4(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_fake_transpose_unrolled4(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_fake_transpose_unrolled8(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_fake_transpose_vecdot_AVX1(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_fake_transpose_vecdot_AVX2(const ymatrix m, const ymatrix m2, int n, ymatrix mout);

void aussie_matmul_matrix_fake_transpose_vecdot_AVX1_inlined(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_fake_transpose_vecdot_AVX2_inlined(const ymatrix m, const ymatrix m2, int n, ymatrix mout);
void aussie_matmul_matrix_fake_transpose_vecdot_AVX1_inlined_unrolled4(const ymatrix m, const ymatrix m2, int n, ymatrix mout);



void aussie_matrix_transpose_basic(const ymatrix m1, int n, ymatrix transpose);  // Transpose created...

//-------------------------------------------------------------------------
// MATRIX-VECTOR MULTIPLY
//-------------------------------------------------------------------------

void aussie_matmul_vector_basic1_buggy(ymatrix m, float v[], int n);
void aussie_matmul_vector_basic2_buggy(ymatrix m, float v[], int n);

void aussie_matmul_vector_basic_out1(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_basic_out2(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_basic_out3(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_tiled_2x2(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_unrolled4(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_unrolled8(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_unrolled4b(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_unrolled8b(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_basic_interchange(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_hoisted_interchange(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_tiled_4x4(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_tiled_2x2_better(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_tiled_2x2_better_hoisted(const ymatrix m, const float v[], int n, float vout[]);

void aussie_matmul_vector_tiled_4x4_CSE(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_tiled_4x4_CSE2(const ymatrix m, const float v[], int n, float vout[]);


void aussie_matmul_vector_basic_out2_rowwise(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_basic_out2_pointer_arith(const ymatrix m, const float v[], int n, float vout[]);


void aussie_matmul_vector_vecdot_AVX1(const ymatrix m, const float v[], int n, float vout[]);
void aussie_matmul_vector_vecdot_AVX2(const ymatrix m, const float v[], int n, float vout[]);

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------


//-------------------------------------------------------------------------
//-------------------------------------------------------------------------


//-------------------------------------------------------------------------


#endif //AUSSIE_YMATMUL_INCLUDE_HEADER_H

