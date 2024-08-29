//---------------------------------------------------
// adynarray.h -- Dynamic arrays (allocated vectors/matrices/tensors) -- Aussie AI Base Library  
// Created March 2024
// Copyright (c) 2024 Aussie AI Labs Pty Ltd
//---------------------------------------------------

#ifndef AUSSIE_YDYNARRAY_INCLUDE_HEADER_H
#define AUSSIE_YDYNARRAY_INCLUDE_HEADER_H

//---------------------------------------------------

void aussie_test_dynarray();  // Unit tests for dynamic arrays

//---------------------------------------------------
//---------------------------------------------------

float* aussie_dynamic_vector_allocate(int n);
void aussie_dynamic_vector_deallocate(float* farr);

//---------------------------------------------------
//---------------------------------------------------

float** aussie_dynamic_matrix_basic_allocate(int rows, int cols);
void aussie_dynamic_matrix_basic_deallocate(float** arr, int rows, int cols_unused = 0);

//---------------------------------------------------
//---------------------------------------------------
void aussie_dynamic_matrix_indexed_deallocate(float** marr, int rows = 0, int cols = 0);
float** aussie_dynamic_matrix_indexed_allocate(int rows, int cols);

//---------------------------------------------------
//---------------------------------------------------

void aussie_dynamic_tensor3D_basic_deallocate(float*** tensorarr, int slices, int rows, int cols_unused = 0);
float*** aussie_dynamic_tensor3D_basic_allocate(int slices, int rows, int cols);

//---------------------------------------------------

void aussie_dynamic_tensor3D_indexed_deallocate(float*** tensorarr, int slices_unused = 0, int rows_unused = 0, int cols_unused = 0);
float*** aussie_dynamic_tensor3D_indexed_allocate(int slices, int rows, int cols);

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------


#endif //AUSSIE_YDYNARRAY_INCLUDE_HEADER_H

