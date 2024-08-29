// ytopk.cpp -- Top-K basic vector algorithms -- Aussie AI Base Library  
// Created Nov 12th 2023
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
#include "atest.h"
#include "avector.h"

#include "atopk.h"  // self-include

//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------



void aussie_vector_top_k_2(float v[], int n, float vout[])  // Topk with k=2
{
	// Order the first 2 elements
	float vmax1 = v[0], vmax2 = v[1];
	if (v[1] > v[0]) {
		vmax1 = v[1];  // Reverse them
		vmax2 = v[0];
	}
	for (int i = 2 /*not 0*/; i < n; i++) {
		if (v[i] > vmax2) {
			// Bigger than the smallest one
			if (v[i] > vmax1) {
				// Bigger than both (shuffle)
				vmax2 = vmax1;
				vmax1 = v[i];
			}
			else {
				// In the middle (fix 2nd only)
				vmax2 = v[i];
			}
		}
	}
	vout[0] = vmax1;  // Biggest
	vout[1] = vmax2;  // 2nd biggest
}


int aussie_top_k_qsort_cmp(void const* addr1, void const* addr2)
{
	float f1 = *(float*)addr1;
	float f2 = *(float*)addr2;
	if (f1 < f2) return +1;  // Reversed (descending)
	else if (f1 > f2) return -1;
	else return 0;
}

void aussie_vector_top_k_qsort(float v[], int n, int k, float vout[])  // Top-k with general k (qsort algorithm)
{
	// Sort the array
	qsort(v, n, sizeof(vout[0]), aussie_top_k_qsort_cmp);
	// Copy top-k elements
	for (int i = 0; i < k; i++) vout[i] = v[i];
}

void aussie_permutation_identity(int permut[], int n)
{
	for (int i = 0; i < n; i++) permut[i] = i;
}

static float* g_float_array_for_qsort = NULL;

int aussie_top_k_qsort_permutation_cmp(void const* addr1, void const* addr2)
{
	int index1 = *(int*)addr1;
	int index2 = *(int*)addr2;
	float f1 = g_float_array_for_qsort[index1];
	float f2 = g_float_array_for_qsort[index2];
	if (f1 < f2) return +1;  // Reversed (descending)
	else if (f1 > f2) return -1;
	else return 0;
}

void aussie_vector_top_k_qsort_permut(float v[], int n, int k, float vout[], int permut_out[])  // Top-k with general k (permuted qsort algorithm)
{
	// Create a dynamic permutation array
	int* permut_arr = ::new int[n];
	aussie_permutation_identity(permut_arr, n);  // Identity permutation

	// Sort the array (by permutation)
	g_float_array_for_qsort = v;
	qsort(permut_arr, n, sizeof(permut_arr[0]), aussie_top_k_qsort_permutation_cmp);
	// Copy top-k elements
	for (int i = 0; i < k; i++) {
		permut_out[i] = permut_arr[i];
		vout[i] = v[permut_arr[i]];
	}
	delete[] permut_arr;
}

void aussie_vector_top_k_shuffle(float v[], int n, int k, float vout[])  // Top-k with general k (shuffle algorithm)
{
	yassert(n >= k);

	// Phase 1: Shuffle sort the first k items
	vout[0] = v[0];
	int nout = 1;
	// DEBUG: aussie_print_vector(vout, nout);
	for (int i = 1 /*not 0*/; i < n; i++) {
		float fnew = v[i];
		int maxj;
		if (nout < k) {
			vout[nout++] = fnew;
			maxj = nout - 2;
		}
		else {
			maxj = nout - 1;
			// DEBUG: nout = nout;  // debug
		}
		maxj = nout - 1;
		// DEBUG: aussie_print_vector(vout, nout);
		for (int j = maxj; j >= 0; j--) {
			// DEBUG: aussie_print_vector(vout, nout);
			if (fnew > vout[j]) {
				// Shuffle & insert
				// DEBUG: aussie_print_vector(vout, nout);
				if (j + 1 < k) vout[j + 1] = vout[j];  // Shuffle down
				vout[j] = fnew;
				// DEBUG: aussie_print_vector(vout, nout);
				// Keep going
			}
			else {
				// Done.. insert it
				if (j != maxj) {
					if (j + 1 < k) vout[j + 1] = vout[j];
					vout[j] = fnew;
				}
				break;
			}
		} // end for j
		//aussie_print_vector(vout, nout);

	} // end for i

}

void aussie_vector_top_k_shuffle_BUGGY(float v[], int n, int k, float vout[])  // Top-k with general k (shuffle algorithm)
{

	// BUG WAS: vout[j+1] = v[j] ... the RHS should be vout[j]
	yassert(n >= k);
	vout[0] = v[0];
	int nout = 1;
	aussie_print_vector(vout, nout);
	for (int i = 1 /*not 0*/; i < n; i++) {
		if (v[i] > vout[nout - 1]) {
			// Bigger than the smallest one 
			// Shuffle new item into place
			float fnew = v[i];
#if 0 // bad code
			if (nout < k) {
				vout[nout] = vout[nout - 1];
				vout[nout - 1] = fnew;
				nout++;
				aussie_print_vector(vout, nout);
			}
#endif

			int maxj = nout - 1;
			for (int j = maxj; j >= 0; j--) {
				aussie_print_vector(vout, nout);
				if (fnew > vout[j]) {
					// Shuffle...
					vout[j + 1] = vout[j];  // Shuffle down
					if (j == 0) {
						vout[j] = fnew;
						break;
					}
					// Keep going
				}
				else { // Done.. insert it
					vout[j + 1] = vout[j];
					vout[j] = fnew;
					break;
				}
			} // end for j
			if (nout < k) {  // Not enough yet?
				nout++;  // We added one...
			}
			aussie_print_vector(vout, nout);
		}
		else if (nout < k) {
			// Not bigger, but don't have k elements in output array yet
			vout[nout++] = v[i];
		}
	}
}

//---------------------------------------------------
//---------------------------------------------------


void aussie_vector_topk_tests()
{
	int n = 10;
	float v1[10];
	float v2[10];
	float f = 0.0f;

	memset(v1, 0, sizeof(v1));
	memset(v2, 0, sizeof(v2));

	// Top-K tests...
	float vout[1000] = { -99.0, -99.0, -99 };
	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_top_k_2(v1, 10, vout);
	ytestf(aussie_vector_max(v1, 10), vout[0]);


	// Test top-k
	aussie_vector_set_1_N(v1, 10);  // Set to 1..10
	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_top_k_2(v1, 10, vout);
	ytest(vout[0] >= vout[1]);
	ytestf(aussie_vector_max(v1, 10), vout[0]);
	ytest(vout[1] != -99.0f);
	ytestf(vout[2], -99.0f);

	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_set_1_N(v1, 10);  // Set to 1..10
	ytestf(aussie_vector_max(v1, 10), 10.0f);
	aussie_vector_top_k_shuffle(v1, 10, 5, vout); // Top-K shuffle
	ytestf(aussie_vector_max(v1, 10), 10.0f);  // v1 unchanged...
	ytestf(aussie_vector_max(v1, 10), vout[0]);
	ytest(vout[0] >= vout[1]);
	ytest(vout[1] >= vout[2]);
	ytest(vout[2] >= vout[3]);
	ytest(vout[3] >= vout[4]);
	ytest(vout[4] != -99.0f);
	ytest(vout[5] == -99.0f);
	ytestf(aussie_vector_max(vout, 5), 10.0f);
	ytestf(aussie_vector_min(vout, 5), 6.0f);


	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_set_1_N_reverse(v1, 10);  // Set to 10..1 (already sorted)
	ytestf(aussie_vector_max(v1, 10), 10.0f);
	aussie_vector_top_k_shuffle(v1, 10, 5, vout); // Top-K shuffle
	ytestf(aussie_vector_max(v1, 10), 10.0f);  // v1 unchanged...
	ytestf(aussie_vector_max(v1, 10), vout[0]);
	ytest(vout[0] >= vout[1]);
	ytest(vout[1] >= vout[2]);
	ytest(vout[2] >= vout[3]);
	ytest(vout[3] >= vout[4]);
	ytest(vout[4] != -99.0f);
	ytest(vout[5] == -99.0f);
	ytestf(aussie_vector_max(vout, 5), 10.0f);
	ytestf(aussie_vector_min(vout, 5), 6.0f);


	// Test QSORT version (TOP-K)
	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_set_1_N(v1, 10);  // Set to 1..10
	ytestf(aussie_vector_max(v1, 10), 10.0f);
	aussie_vector_top_k_qsort(v1, 10, 5, vout); // Top-K shuffle
	ytestf(aussie_vector_max(v1, 10), 10.0f);  // v1 unchanged...
	ytestf(aussie_vector_max(v1, 10), vout[0]);
	ytest(vout[0] >= vout[1]);
	ytest(vout[1] >= vout[2]);
	ytest(vout[2] >= vout[3]);
	ytest(vout[3] >= vout[4]);
	ytest(vout[4] != -99.0f);
	ytest(vout[5] == -99.0f);
	ytestf(aussie_vector_max(vout, 5), 10.0f);
	ytestf(aussie_vector_min(vout, 5), 6.0f);


	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_set_1_N_reverse(v1, 10);  // Set to 10..1 (already sorted)
	ytestf(aussie_vector_max(v1, 10), 10.0f);
	aussie_vector_top_k_qsort(v1, 10, 5, vout); // Top-K shuffle
	ytestf(aussie_vector_max(v1, 10), 10.0f);  // v1 unchanged...
	ytestf(aussie_vector_max(v1, 10), vout[0]);
	ytest(vout[0] >= vout[1]);
	ytest(vout[1] >= vout[2]);
	ytest(vout[2] >= vout[3]);
	ytest(vout[3] >= vout[4]);
	ytest(vout[4] != -99.0f);
	ytest(vout[5] == -99.0f);
	ytestf(aussie_vector_max(vout, 5), 10.0f);
	ytestf(aussie_vector_min(vout, 5), 6.0f);

	// Test PERMUTATION QSORT version (TOP-K)
	int pout[1000];   // Permutation array
	aussie_vector_setall_intarr(pout, 100, -88);
	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_set_1_N(v1, 10);  // Set to 1..10
	ytestf(aussie_vector_max(v1, 10), 10.0f);
	aussie_vector_top_k_qsort_permut(v1, 10, 5, vout, pout); // Top-K shuffle
	ytestf(aussie_vector_max(v1, 10), 10.0f);  // v1 unchanged...
	ytestf(aussie_vector_max(v1, 10), vout[0]);
	ytest(vout[0] >= vout[1]);
	ytest(vout[1] >= vout[2]);
	ytest(vout[2] >= vout[3]);
	ytest(vout[3] >= vout[4]);
	ytest(vout[4] != -99.0f);
	ytest(vout[5] == -99.0f);
	ytestf(aussie_vector_max(vout, 5), 10.0f);
	ytestf(aussie_vector_min(vout, 5), 6.0f);


	aussie_vector_setall_intarr(pout, 100, -88);
	aussie_vector_setall(vout, 100, -99.0f);
	aussie_vector_set_1_N_reverse(v1, 10);  // Set to 10..1 (already sorted)
	ytestf(aussie_vector_max(v1, 10), 10.0f);
	aussie_vector_top_k_qsort_permut(v1, 10, 5, vout, pout); // Top-K shuffle
	ytestf(aussie_vector_max(v1, 10), 10.0f);  // v1 unchanged...
	ytestf(aussie_vector_max(v1, 10), vout[0]);
	ytest(vout[0] >= vout[1]);
	ytest(vout[1] >= vout[2]);
	ytest(vout[2] >= vout[3]);
	ytest(vout[3] >= vout[4]);
	ytest(vout[4] != -99.0f);
	ytest(vout[5] == -99.0f);
	ytestf(aussie_vector_max(vout, 5), 10.0f);
	ytestf(aussie_vector_min(vout, 5), 6.0f);


}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

