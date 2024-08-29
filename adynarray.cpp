//---------------------------------------------------
// adynarray.h -- Dynamic arrays (allocated vectors/matrices/tensors) -- Aussie AI Base Library  
// Created March 2024
// Copyright (c) 2024 Aussie AI Labs Pty Ltd
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

#include "adynarray.h"  // self-include

//---------------------------------------------------

float* aussie_dynamic_vector_allocate(int n)
{
	// Allocate 1-D vector...
	float* arr = (float*)calloc(sizeof(float), n);
	return arr;
}

void aussie_dynamic_vector_deallocate(float* farr)
{
	// Free dynamic vector
	if (!farr) {
		yassert(farr);
		return;
	}
	free(farr);
}

//---------------------------------------------------
//---------------------------------------------------

float** aussie_dynamic_matrix_basic_allocate(int rows, int cols)
{
	// Allocate 2-D matrix... basic lots-of-blocks version
	// ... The MAIN ADVANTAGE is that you can use the basic ARR[i][j] syntax,
	// ... without having to do a LINEARIZED macro trick with i*SIZE+j. 
	// Basic version where each Matrix row is a separate block (of data)
	// ... as compared to the indexed version with a single block of data.
	// Advantage: each row is its own block, so can detect row vector array overruns (e.g. with Valgrind)
	//    .. and each row is still in contiguous memory, so can do vector-level operations faster..
	// Disadvantage: lots of memory allocations is slow...
	// ... Rows are not in contiguous memory (with each other), each row in a separate memory block
	// ... which makes it hard to do operations over multiple rows (e.g. initialize whole matrix with memset)
	// ... or advanced cross-row matrix optimizations like 'tiling'

	// Index pointers block
	float**arr = (float**)calloc(sizeof(float*), rows);
	if (!arr) { // alloc failure
		yassert(arr);
		return NULL;
	}

	// Allocate a block for each row...
	for (int i = 0; i < rows; i++) {
		arr[i] = (float*)calloc(sizeof(float), cols);  // Alloc a row...
	}
	return arr;
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_dynamic_matrix_basic_deallocate(float **arr, int rows, int cols_unused /*= 0*/)
{
	// Deallocate basic lots-of-blocks 2D dynamic matrix
	// NOTE: to deallocate, we only need to know the 'rows' count...
	// TODO: We should store the rows & cols count in an object with the array pointer...
	// De-allocate all the rows....
	for (int i = 0; i < rows; i++) {  // For all rows
		free(arr[i]);  // Free each row
	}
	free(arr);  // Free the main index block
}

//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

float** aussie_dynamic_matrix_indexed_allocate(int rows, int cols)
{
	// Allocate Indexed dynamic 2-D matrix...
	// ... Indexed version has only 2 allocated blocks:
	// ... 1) The main data block with ALL the data is a single contiguous block
	// ... 2) The index block which has a pointer for each ROW...
	// NOTE: This is similar to a LINEARIZED array, but with an extra INDEX block
	// ... you can still scan the data block as a linear array of the whole Matrix...
	// Main data block
	int dataelems = rows * cols;
	float* dataarr = (float*)calloc(sizeof(float), dataelems);
	if (!dataarr) {
		yassert(dataarr);
		return NULL;
	}
	float* enddataarr = dataarr + dataelems;  // Note: pointer arithmetic

	// Index pointers block
	float** arr = (float**)calloc(sizeof(float*), rows);
	if (!arr) { // alloc failure
		yassert(arr);
		return NULL;
	}

	// Each row's index is just the next row of data in the big data block...
	// Don't allocate more blocks, just point into the middle of the block

	float* rowarr = &dataarr[0];
	for (int i = 0; i < rows; i++, rowarr += cols) {
		arr[i] = rowarr;
	}
	return arr;
}

void aussie_dynamic_matrix_indexed_deallocate(float** marr, int rows /*= 0*/, int cols /* = 0*/)
{
	// Deallocate Indexed 2D matrix: 2 memory blocks: big data array & index array of pointer...
	float* dataarr = marr[0];  // Start of the data is also the block's first row start..
	yassert(dataarr);
	if (dataarr) {
		free(dataarr);
	}
	free(marr);  // Free the index block (pointers to rows)
}

//---------------------------------------------------
//---------------------------------------------------


float*** aussie_dynamic_tensor3D_basic_allocate(int slices, int rows, int cols)
{
	// Allocate 3-D tensor... basic lots-of-blocks version
	// Basic version where each Matrix row is a separate block (of data)
	// ... as compared to the indexed version with a single big block of data.
	// 
	// Advantage: each row is its own block, so can detect row vector array overruns (e.g. with Valgrind)
	//    .. and each row is still in contiguous memory, so can do vector-level operations faster..
	// Disadvantage: lots of memory allocations is slow...
	// ... Rows are not in contiguous memory (with each other), each row in a separate memory block
	// ... which makes it hard to do operations over multiple rows (e.g. initialize whole matrix with memset)
	// ... or advanced cross-row matrix optimizations like 'tiling'

	float*** tensorarr = (float***)calloc(sizeof(float**), slices);  // Tensor top-level (SLICES, level=3)

	for (int slice = 0; slice < slices; slice++) { // Slices=Matrices ...(Level 3)

		// Level 2: Index pointers block (Matrix row-pointers, level=2)
		float** matrixarr = (float**)calloc(sizeof(float*), rows);
		if (!matrixarr) { // alloc failure
			yassert(matrixarr);
			return NULL;
		}

		tensorarr[slice] = matrixarr;  // Store the matrix as a tensor slice... (into level 3)

		// Level 1: Allocate a low-level block for each row in each matrix... (Row, level=1)
		for (int i = 0; i < rows; i++) {
			matrixarr[i] = (float*)calloc(sizeof(float), cols);  // Alloc a row...
		}
	}

	return tensorarr;
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_dynamic_tensor3D_basic_deallocate(float*** tensorarr, int slices, int rows, int cols_unused /*= 0*/)
{
	// Deallocate 3D tensor -- basic lots-of-blocks 3D dynamic tensor
	// TODO: We should store the rows & cols count in an object with the array pointer...
	// De-allocate all the rows....
	for (int slice = 0; slice < slices; slice++) {
		float** marr = tensorarr[slice];  // Matrix (a single slice)
		for (int i = 0; i < rows; i++) {  // For all rows
			free(marr[i]);  // Free each row
		}
		free(marr);  // Free the main MATRIX level-2 index block
	}
	free(tensorarr);
}


//---------------------------------------------------
//---------------------------------------------------


float*** aussie_dynamic_tensor3D_indexed_allocate(int slices, int rows, int cols)
{
	// Allocate 3-D tensor... INDEXED few-blocks version (has 3 blocks)
	// The dynamic 3D tensor is 3 allocated blocks:
	//   1. The big data block (linearized array representing 3D data)
	//   2. The matrix-level indexes, lots of pointers to rows of each matrix
	//   3. The slice-level index, pointer to a matrix for each tensor slice...

	// Advantage: each row is its own block, so can detect row vector array overruns (e.g. with Valgrind)
	//    .. and each row is still in contiguous memory, so can do vector-level operations faster..
	// Disadvantage: lots of memory allocations is slow...
	// ... Rows are not in contiguous memory (with each other), each row in a separate memory block
	// ... which makes it hard to do operations over multiple rows (e.g. initialize whole matrix with memset)
	// ... or advanced cross-row matrix optimizations like 'tiling'

	int ndata = slices * rows * cols;
	int nmatrix = slices * rows;

	// Allocate the 3 blocks...
	float* data = (float*)calloc(sizeof(float), ndata);   // Big DATA array (level=1)
	yassert(data);
	float** marr = (float**)calloc(sizeof(float*), nmatrix);  // Matrix index of ptrs-to-rows.. (level=2).
	yassert(marr);
	float*** tensorarr = (float***)calloc(sizeof(float**), slices);  // Tensor top-level (SLICES, level=3)
	yassert(tensorarr);

	// Set up the INDEXING in the 2 higher-level arrays...
	float** matrixarr = &marr[0];  // First matrix index entry start
	float** matrixend = matrixarr + (nmatrix /*slices*rows*/);
	float* rowarr = &data[0];  // First data row of first matrix is also the start of the big data block...
	float* dataend = data + (ndata /*slices*rows*cols*/);
	for (int slice = 0; slice < slices; slice++) { // Slices=Matrices ...(Level 3)

		// Top-level index (level=3) ... slices point into the matrix-index block...
		tensorarr[slice] = matrixarr;  // Store the matrix as a tensor slice... (into level 3)

		// Level 2: Index pointers block (Matrix row-pointers, level=2)
		for (int i = 0; i < rows; i++) {
			*matrixarr = rowarr;
			matrixarr++;
			rowarr += cols;  // Note: pointer aritmetic
		}
	}

	yassert(rowarr == dataend);
	yassert(matrixarr == matrixend);

	return tensorarr;
}

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

void aussie_dynamic_tensor3D_indexed_deallocate(float*** tensorarr, int slices_unused /*= 0*/, int rows_unused /*=0*/, int cols_unused /*= 0*/)
{
	// Deallocate 3D tensor -- INDEXED -- 3 blocks of memory
	// ... We can find them without knowing the slices/rows/cols dimensions
	
	yassert(tensorarr);
	float** marray = tensorarr[0];   // First matrix is the start of the MATRIX INDEX block
	yassert(marray);
	if (marray) {
		float* data = marray[0];   // First row of the first matrix is the start of the big DATA block
		yassert(data);
		free(data);  // Free the data block
		free(marray);  // Free the MATRIX level-2 index block
	}
	if (tensorarr) {
		free(tensorarr);  // Free the TENSOR level-3 slice index block
	}
}


//---------------------------------------------------
//---------------------------------------------------

void aussie_test_dynarray()  // Unit tests for dynamic arrays
{
	int cols = 20;
	float* farr = aussie_dynamic_vector_allocate(cols /*20*/);
	// Visit every vector element so can Valgrind check
	for (int i = 0; i < cols; i++) {
		farr[i] = (float)i;
	}
	aussie_dynamic_vector_deallocate(farr);
	farr = NULL;

	// Test the BASIC LOTS-OF-BLOCKS dynamic matrix -- lots of blocks...
	// ... each matrix row is a separate block (1 vector)
	cols = 20;
	int rows = 10;  // Matrix rows: 10x20 matrix
	float ** marr = aussie_dynamic_matrix_basic_allocate(10, 20);
	// Visit every matrix element so can Valgrind check
	int ct = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			marr[i][j] = (float)ct++;
		}
	}
	aussie_dynamic_matrix_basic_deallocate(marr, rows, cols);
	marr = NULL;


	// Test the INDEXED dynamic matrix -- 2 block only...
	// ... each matrix row is pointer into the data block...
	cols = 20;
	rows = 10;  // Matrix rows: 10x20 matrix
	marr = aussie_dynamic_matrix_indexed_allocate(10, 20);
	// Visit every matrix element so can Valgrind check
	ct = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			marr[i][j] = (float)ct++;
		}
	}
	aussie_dynamic_matrix_indexed_deallocate(marr); // Free the 2 blocks
	marr = NULL;

	//----------------------------------------------------
	// 3D tensor basic lots-of-blocks version...
	//----------------------------------------------------
	float*** tensorarr = NULL;  // 3D tensor...
	int slices = 5;
	cols = 20;
	rows = 10;  // Matrix rows: 10x20 matrix
	tensorarr = aussie_dynamic_tensor3D_basic_allocate(slices, rows, cols);
	// Visit every matrix element so can Valgrind check
	ct = 0;
	for (int slice = 0; slice < slices; slice++) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				tensorarr[slice][i][j] = (float)ct++;
			}
		}
	}
	aussie_dynamic_tensor3D_basic_deallocate(tensorarr, slices, rows, cols); // Free the 2 blocks
	marr = NULL;

	//----------------------------------------------------
	// 3D tensor INDEXED few-blocks version... (3 blocks)
	//----------------------------------------------------
	tensorarr = NULL;  // 3D tensor...
	slices = 5;
	cols = 20;
	rows = 10;  // Matrix rows: 10x20 matrix
	tensorarr = aussie_dynamic_tensor3D_indexed_allocate(slices, rows, cols);
	// Visit every matrix element so can Valgrind check
	ct = 0;
	for (int slice = 0; slice < slices; slice++) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				tensorarr[slice][i][j] = (float)ct++;
			}
		}
	}
	aussie_dynamic_tensor3D_indexed_deallocate(tensorarr, slices, rows, cols); // Free the 2 blocks
	marr = NULL;


}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

