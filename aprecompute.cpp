//---------------------------------------------------
// yprecompute.cpp -- Precomputed table lookup APIs -- Aussie AI Base Library  
// Created Oct 29th 2023
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

#include "aussieai.h" // Overall API

#include "aassert.h"
#include "afloat.h"
#include "atest.h"

#include "aprecompute.h"  // self-include

//---------------------------------------------------
//---------------------------------------------------

void aussie_generic_precompute_int(float arr[], unsigned int maxn, float (*fnptr)(int))
{
	for (unsigned int i = 0; i < maxn; i++) {
		arr[i] = fnptr(i);
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_generic_precompute_24bit_float(float farr[], unsigned int maxn, float (*fnptr)(float))
{
	for (unsigned int u = 0; u < maxn; u++) {
		unsigned int unum = (u << 8u);  // 32-24=8 bits!
		float f = *(float*)&unum;
		farr[u] = fnptr(f);
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_generic_precompute_float(float farr[], unsigned int maxn, float (*fnptr)(float))
{
	for (unsigned int u = 0; u < maxn; u++) {
		unsigned int unum = (u << 16u);
		float f = *(float*)&unum;
		farr[u] = fnptr(f);
	}
}

float g_sqrt_float_precomp_table[1u << 16];
float g_sqrt_float_24bit_precomp_table[1u << 24];

 
float aussie_table_lookup_sqrt_float(float f)
{
	unsigned u = *(unsigned int*)&f;
	u >>= 16u;
	return g_sqrt_float_precomp_table[u];
}

float aussie_table_lookup_sqrt_24bit_float(float f)
{
	unsigned u = *(unsigned int*)&f;
	u >>= 8;  // 32-24 bits
	return g_sqrt_float_24bit_precomp_table[u];
}

//---------------------------------------------------
//---------------------------------------------------

#define AUSSIE_SQRT_PRECOMP_MAX (1u<<16)
float g_sqrt_precomp_table[AUSSIE_SQRT_PRECOMP_MAX];

float aussie_sqrtf_basic_float(float f)
{
	return sqrtf(f);
}

float aussie_sqrtf_basic_int(int x)
{
	return sqrtf((float)x);
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_precompute_sqrt()
{
	printf("INFO: Size of sqrt INT 16-bit table of %d = %d\n", AUSSIE_SQRT_PRECOMP_MAX,(int)sizeof(g_sqrt_precomp_table));
	aussie_generic_precompute_int(g_sqrt_precomp_table, AUSSIE_SQRT_PRECOMP_MAX, aussie_sqrtf_basic_int);

#define AUSSIE_SQRT_16bit_MAX (1u << 16) 
	printf("INFO: Size of sqrt FLOAT 16-bit table of %d = %d\n", (int)AUSSIE_SQRT_16bit_MAX, (int)sizeof(g_sqrt_float_precomp_table));
	aussie_generic_precompute_float(
		g_sqrt_float_precomp_table, 
		(int)AUSSIE_SQRT_16bit_MAX, 
		aussie_sqrtf_basic_float
	);

#define AUSSIE_SQRT_24bit_MAX (1u << 24) 
	printf("INFO: Size of sqrt FLOAT 16-bit table of %d = %d\n", (int)AUSSIE_SQRT_24bit_MAX, (int)sizeof(g_sqrt_float_24bit_precomp_table));
	aussie_generic_precompute_24bit_float(
		g_sqrt_float_24bit_precomp_table,
		(int)AUSSIE_SQRT_24bit_MAX, 
		aussie_sqrtf_basic_float
	);

}

//---------------------------------------------------
//---------------------------------------------------

float aussie_table_lookup_sqrt(int i)
{
	return g_sqrt_precomp_table[i];
}

#define AUSSIE_TABLE_LOOKUP_SQRT_BASIC(i) \
	 ( g_sqrt_precomp_table[(i)] )

float aussie_table_lookup_sqrt2(int i)
{
	return AUSSIE_TABLE_LOOKUP_BASIC(g_sqrt_precomp_table, i);
}



//---------------------------------------------------
//---------------------------------------------------

void aussie_test_sqrt_int(int i)
{
	float f = sqrtf((float)i);

	// Test int-to-float precomputed sqrt version
	float f2 = aussie_table_lookup_sqrt(i);
	ytestf(f, f2);
	f2 = aussie_table_lookup_sqrt2(i);
	ytestf(f, f2);
	f2 = AUSSIE_TABLE_LOOKUP_SQRT_BASIC(i);
	ytestf(f, f2);

	// Test 16-bit float-to-float precomputed version
	f2 = aussie_table_lookup_sqrt_float((float)i);
	ytestfapprox(f, f2, 0.1);  // Error 0.1?

	// Test 24-bit float-to-float precomputed version
	f2 = aussie_table_lookup_sqrt_24bit_float((float)i);
	ytestfapprox(f, f2, 0.001);  // Error 0.1?


}

void aussie_unit_test_precompute()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	aussie_precompute_sqrt();

	for (int i = 0; i < 1000; i++) {
		aussie_test_sqrt_int(i);
	}
}
//---------------------------------------------------
//---------------------------------------------------

void aussie_precompute_table_FP32_generic_24bits(   // Initialize precomputed table
	float arr_table[],
	unsigned int maxn,
	float (*fnptr)(float))
{
	unsigned long int u = 0;
	for (; u < maxn /*1u<<24*/; u++) {  // For all 2^24=~16.7M...
		unsigned int uval = u << 8;  // put zeros in the least significant 8 mantissa bits
		float f = AUSSIE_UINT_TO_FLOAT(uval);
		float g = (*fnptr)(f);
		arr_table[u] = g;  // Store precomputed result...
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_generic_setup_table_FP32_24bits_PRINT_SOURCE( // Print C++ of 24-bits GELU precomputed table 
	char* nickname,
	char* outfname,
	float (*fnptr)(float),  // e.g. GELU
	unsigned int maxn,  // eg. 1<<24
	float arrout[]  // array to store (optional, can be NULL)
)
{
	if (!fnptr) {
		yassert(fnptr);
		return;
	}
	// Generate C++ source code so we can pre-compile the precomputed GELU table (24-bits)
	// There are 2^24 = 16.7 million numbers...
	FILE* fp = stdout;
	bool writingfile = false;
	bool add_commented_number = true;
	if (outfname && *outfname) {
		fp = fopen(outfname, "w");
		if (!fp) {
			yassert(fp);  // file write failed
			return;  // fail
		}
		writingfile = true;
		add_commented_number = false;  // No extra comments for file output version
	}
	unsigned int u = 0;
	fprintf(fp, "// Precomputed table source code: %s, \"%s\"\n", nickname, outfname);
	fprintf(fp, "float g_gelu_table_precompute_24bits[] = { \n");
	char numbuf[5000] = "";
	for (; u < maxn /*1<<24*/ ; u++) {  // For all 2^24=~16.7M...
		unsigned int uval = u << 8;  // put zeros in the least significant 8 mantissa bits
		float f = AUSSIE_UINT_TO_FLOAT(uval);
		float g = fnptr(f);  // Call GELU or whatever
		if (arrout) arrout[u] = g;  // Store precomputed data (e.g. GELU)...

		// Format: %g means the smaller of %e or %f
		// ... %e is the exponent format (scientific-like format)
		char* buf = numbuf;
		sprintf(buf, "%40.40gf", g);  // Format %g (Number) and suffix "f" (float constant type)
		if (strchr(buf, 'n')) {
			// Nan or "-nan" ... 
			strcpy(buf, "0.0 /*nan*/");  // Dummy value for NaN
		}
		// Remove prefix padding spaces...
		while (buf[0] == ' ') buf++;

		// Remove suffix zeros ...
		int len = (int)strlen(buf);
		if (buf[len - 1] == 'f') len--;  // skip suffix f
		if (buf[len - 1] == '0') {
			while (len > 5) {
				if (buf[len - 1] == '0' && isdigit(buf[len - 2])) {
					if (buf[len] == 'f') {
						buf[len - 1] = 'f';  // remove it, but leave 'f'...
						buf[len] = 0;
					}
					else {
						buf[len - 1] = 0;  // remove it...
						buf[len] = 0;
					}
					len--;
				}
				else break;
			}
		}

		if (add_commented_number) {
			fprintf(fp, "%s // (%40.40f) [%u] \n", buf, f, u);
		}
		else {  // No comments...
			fprintf(fp, "%s,\n", buf);
		}

		// Progress update
		if (u % 100000 == 0 && u != 0) {
			if (writingfile) fprintf(stdout, "%u -- %s\n", u, buf);  // Progress to stdout...
			fprintf(fp, "// U= [%u]\n", u);  // Comment occasionally
		}
	}
	fprintf(fp, "}; \n");  // Close initializer...
	if (fp && fp != stdout) fclose(fp);

}
//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

