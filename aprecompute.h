//-----------------------------------------------
// yprecompute.h -- Precomputed table lookup APIs -- Aussie AI Base Library  
// Created Oct 29th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd
//-----------------------------------------------

#ifndef AUSSIE_YPRECOMPUTE_INCLUDE_HEADER_H
#define AUSSIE_YPRECOMPUTE_INCLUDE_HEADER_H

//-----------------------------------------------
//-----------------------------------------------

#define AUSSIE_TABLE_LOOKUP_FAST(arr, f)  ( (arr)[*(unsigned int*)&(f)] )

#define AUSSIE_TABLE_LOOKUP_BASIC(arr, i)  ( (arr)[(i)] )

//-----------------------------------------------
//-----------------------------------------------

void aussie_precompute_table_FP32_generic_24bits(   // Initialize precomputed table
	float arr_table[],
	unsigned int maxn,
	float (*fnptr)(float));

void aussie_generic_setup_table_FP32_24bits_PRINT_SOURCE( // Print C++ of 24-bits precomputed table 
	char* nickname,
	char* outfname,
	float (*fnptr)(float)
);

void aussie_generic_setup_table_FP32_24bits_PRINT_SOURCE( // Print C++ of 24-bits GELU precomputed table 
	char* nickname,
	char* outfname,
	float (*fnptr)(float),  // e.g. GELU
	unsigned int maxn,  // eg. 1<<24
	float arrout[]  // array to store (optional, can be NULL)
);

//-----------------------------------------------
//-----------------------------------------------

void aussie_unit_test_precompute();

//-----------------------------------------------
//-----------------------------------------------

#endif //AUSSIE_YPRECOMPUTE_INCLUDE_HEADER_H

