// atopk.h -- Top-K basic vector algorithms -- Aussie AI Base Library  
// Created Nov 12th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YTOPK_INCLUDE_HEADER_H
#define AUSSIE_YTOPK_INCLUDE_HEADER_H

void aussie_vector_top_k_2(float v[], int n, float vout[]);  // Topk with k=2
int aussie_top_k_qsort_cmp(void const* addr1, void const* addr2);
void aussie_vector_top_k_qsort(float v[], int n, int k, float vout[]);  // Top-k with general k (qsort algorithm)
void aussie_vector_top_k_qsort_permut(float v[], int n, int k, float vout[], int permut_out[]);  // Top-k with general k (permuted qsort algorithm)
void aussie_vector_top_k_shuffle(float v[], int n, int k, float vout[]);  // Top-k with general k (shuffle algorithm)
void aussie_vector_top_k_shuffle_BUGGY(float v[], int n, int k, float vout[]);  // Top-k with general k (shuffle algorithm)

// PERMUTATIONS
void aussie_permutation_identity(int permut[], int n);
int aussie_top_k_qsort_permutation_cmp(void const* addr1, void const* addr2);

void aussie_vector_topk_tests();  // Unit tests for Top-K 

#endif //AUSSIE_YTOPK_INCLUDE_HEADER_H

