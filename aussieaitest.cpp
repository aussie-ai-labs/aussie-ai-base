// aussieaitest.cpp -- Aussie AI Base Library main tests
// Created 27th July 2023
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
#include "atest.h"
#include "avector.h"
#include "aassert.h"
#include "abitwise.h"
#include "afloat.h"
#include "aactivation.h"
#include "amatmul.h"
#include "aprecompute.h"
#include "aportabtest.h"
#include "adebug.h"
#include "anorms.h"
#include "asoftmax.h"
#include "anormalize.h"
#include "aavx.h"
#include "abenchmark.h"
#include "abook1.h"  // Book examples
#include "adynarray.h"

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

// AUSSIE AI internal stats...
long int g_aussie_multiplications_count = 0;   // Count multiplications



void aussie_print_environment()
{
	fprintf(stdout, "AUSSIE AI Execution Environment:\n");
	fprintf(stdout, "sizeof float = %d bytes (%d bits)\n", (int)sizeof(float), 8* (int)sizeof(float));
	fprintf(stdout, "sizeof double = %d bytes (%d bits)\n", (int)sizeof(double), 8 * (int)sizeof(double));
	fprintf(stdout, "sizeof long double = %d bytes (%d bits)\n", (int)sizeof(long double), 8 * (int)sizeof(long double));
	fprintf(stdout, "sizeof int = %d bytes (%d bits)\n", (int)sizeof(int), 8 * (int)sizeof(int));
	fprintf(stdout, "sizeof uint = %d bytes (%d bits)\n", (int)sizeof(unsigned int), 8 * (int)sizeof(unsigned int));
	fprintf(stdout, "sizeof short = %d bytes (%d bits)\n", (int)sizeof(short), 8 * (int)sizeof(short));
	fprintf(stdout, "sizeof ushort = %d bytes (%d bits)\n", (int)sizeof(unsigned short), 8 * (int)sizeof(unsigned short));
	fprintf(stdout, "sizeof long = %d bytes (%d bits)\n", (int)sizeof(long), 8 * (int)sizeof(long));
	fprintf(stdout, "sizeof ulong = %d bytes (%d bits)\n", (int)sizeof(unsigned long), 8 * (int)sizeof(unsigned long));
	fprintf(stdout, "sizeof long long = %d bytes (%d bits)\n", (int)sizeof(long long), 8 * (int)sizeof(long long));

}

void aussie_test_one_reluize(float v[], int n)
{
	float vcopy[5000] = { 0 };
	yassert(n < 5000);
	aussie_vector_copy_basic(vcopy, v, n);

	float sumbefore = aussie_vector_sum(v, n);
	aussie_vector_reluize(v, n);
	float sumafter = aussie_vector_sum(v, n);


	aussie_vector_copy_basic(v, vcopy, n);  // Again...
	float sumbefore2 = aussie_vector_sum(v, n);
	ytestf(sumbefore, sumbefore2);
	aussie_vector_reluize(v, n);
	float sumafter2 = aussie_vector_sum(v, n);
	ytestf(sumafter2, sumafter);

#if !LINUX
	aussie_vector_copy_basic(v, vcopy, n);  // AVX1...
	sumbefore2 = aussie_vector_sum(v, n);
	ytestf(sumbefore, sumbefore2);
	aussie_vector_reluize_AVX1(v, n);
	sumafter2 = aussie_vector_sum(v, n);
	ytestf(sumafter2, sumafter);
#endif //LINUX

#if !LINUX
	aussie_vector_copy_basic(v, vcopy, n);  // AVX2...
	sumbefore2 = aussie_vector_sum(v, n);
	ytestf(sumbefore, sumbefore2);
	aussie_vector_reluize_AVX2(v, n);
	sumafter2 = aussie_vector_sum(v, n);
	ytestf(sumafter2, sumafter);
#endif //LINUX

}

void aussie_reluize_unit_tests()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	float v[1000] = { 0 };
	int n = 128; 

	aussie_vector_set_1_N(v, n);
	aussie_test_one_reluize(v, n);  // Multi-test
	float sum = aussie_vector_sum(v, n);
	aussie_vector_reluize(v, n);
	float sum2 = aussie_vector_sum(v, n);
	ytestf(sum2, sum);  // unchanged...

	aussie_vector_set_range(v, n, -100, -50);
	aussie_test_one_reluize(v, n);
	
	aussie_vector_set_range(v, n, -100, -50);
	sum = aussie_vector_sum(v, n);
	ytest(sum < -9000);
	aussie_vector_reluize(v, n);  // Multi-test
	sum = aussie_vector_sum(v, n);
	ytestf(sum, 0);

}

void aussie_unit_tests()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	aussie_test_dynarray();

	aussie_book_examples_unit_tests();

	aussie_reluize_unit_tests();

#if !LINUX
	aussie_unit_test_avx();
#endif //LINUX

	aussie_book_examples();

	aussie_unit_test_precompute();

	aussie_activation_unit_tests();

	aussie_portability_check(false/*printout*/);

	aussie_unit_test_bitwise();
	aussie_debugging_test_setup();

	// Test vector dot products...
	aussie_yvector_unit_tests();


	aussie_float_tests();

	aussie_matrix_tests_basic();  // Test basic matrix algebra

	aussie_precompute_tests();

	// Report at end of unit testing....
	aussie_unit_tests_report();


}

//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

int main(int argc, char* argv[], char *envp[])
{
	printf("AUSSIE AI BASE: starting test execution.\n");
	int action = 0;  // 0=unit tests, 1=printenv, 2=accuracy, 3=benchmark basic ops, 4=model!, 5=benchmarks

	switch (action) {
	case 0: 
		aussie_unit_tests();
		break;

	case 1:
		aussie_print_environment();
		break;

	case 2:
		yap_test_operator_accuracy();  // Test approximate operations accuracy...
		break;

	case 3:
		yap_benchmark_operations();  // Test benchmark performance of basic arithmetic operations
		break;

	case 4: // run model...
		fprintf(stderr, "ERROR: %s: Not supported\n", __func__);
		break;

	case 5:   // Benchmarks
		aussie_test_benchmarks();
		break;


	} // End switch
	exit(0);
}
