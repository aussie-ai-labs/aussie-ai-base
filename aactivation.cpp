// yactivation.cpp -- Activation Functions -- Aussie AI Base Library  
// Created Oct 12th 2023
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
//#include "aussieai.h"
#include "aassert.h"
#include "afloat.h"
#include "atest.h"
#include "aprecompute.h"

#include "aactivation.h"  // self-include

//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

float aussie_RELU_basic(float f)   // Basic RELU (inefficient)
{
	if (f <= 0.0) return 0.0;
	return f;
}

float aussie_RELU_if_test_slow(float f)
{
	if (f <= 0.0) return 0.0;
	else return f;
}

//---------------------------------------------------
//---------------------------------------------------

//#define AUSSIE_RELU_MACRO(f)  ( (f) <= 0.0 ? 0.0 : (f) )



float aussie_ELU_basic(float x, float alpha_hyperparam)   // Basic ELU activation (inefficient)
{
	// ELU = x  if x > 0 .0
	//     = alpha * ( exp(x) - 1) if x <= 0.0
	if (x <= 0.0) return alpha_hyperparam * (expf(x) - 1.0f);
	return x;  // x if x > 0.0
}

// ... alpha is a hyper-parameter controlling the version of ELU

//---------------------------------------------------
//---------------------------------------------------

void aussie_vector_reluize(float v[], int n)   // Apply RELU to each element
{
	for (int i = 0; i < n; i++) {
		v[i] = AUSSIE_RELU_MACRO(v[i]);
	}
}

//---------------------------------------------------
// GELU -- https://browse.arxiv.org/pdf/1606.08415v5.pdf
// GELU = Gaussian Error Linear Unit (GELU)
// = x * PHI(x)
// PHI is the Bernoulli probability function
// erf is the error function
// PHI(x) = 1/2 *  ( 1 + erf ( x / sqrt(2) ) ) 
// https://en.wikipedia.org/wiki/Logistic_distribution (sigma(x) -- Logistic function cumulative distribution function/CDF?
//---------------------------------------------------

float aussie_GELU_basic(float x)   // Basic Gaussian GELU (inefficient)
{
	float phival = 0.5f * (1.0f + erff(x / sqrtf(2.0f)));   // NOTE: erff() is float version of erf() "error function"
	return x * phival;
}

//static float s_sqrt_2_0 = sqrtf(2.0f);  // Once-only initializations

float aussie_GELU_basic2(float x)   // Basic Gaussian GELU (still inefficient)
{
	static float s_reciprocal_sqrt_2_0 = 1.0f / sqrtf(2.0f);  // Once-only initializations
	return x * (0.5f * (1.0f + erff(x * s_reciprocal_sqrt_2_0)));
}


//---------------------------------------------------
// GELU Approximation #1 in https://browse.arxiv.org/pdf/1606.08415v5.pdf
// ... 0.5 * x *  (1 + tanh ( (sqrt(2.0 / PI ) * ( x + 0.044715 * x^3) )  ) 
//---------------------------------------------------

float aussie_GELU_approx1(float f)   // Approximated Gaussian GELU
{
	// GELU paper approx #1 = 0.5 * x * ( 1 + tanh ( sqrt(2/PI) * (x + 0.44715 * x^3)  ) ) 

	return 0.5f * f * (1.0f + tanhf(sqrtf(2.0f / AUSSIE_PI) * (f + (0.44715f * (f * f * f)))));
}

float aussie_GELU_approx1_optimized(float f)   // Approximated Gaussian GELU (with minor optimizations)
{
	// GELU paper approx #1 = 0.5 * x * ( 1 + tanh ( sqrt(2/PI) * (x + 0.44715 * x^3)  ) ) 
	static float s_sqrt_2_div_pi = sqrtf(2.0f / AUSSIE_PI);
	return 0.5f * f *
		(1.0f + tanhf(s_sqrt_2_div_pi *
			(f + (0.44715f * (f * f * f))))
			);
}


//---------------------------------------------------
// GELU Approximation #2 in https://browse.arxiv.org/pdf/1606.08415v5.pdf
// ... x * sigmoid(1.702 * x)
// ... sigmoid(x) = 1.0 / ( 1.0 + expf(-x) )
// ... (sigma/sigmoid(x) -- Logistic function cumulative distribution function/CDF? (Same as sigmoid function??)
//---------------------------------------------------

float aussie_GELU_approx1_optimized2(float f)   // Approximated Gaussian GELU (with 2nd minor optimizations)
{
	// GELU paper approx #1 = 0.5 * x * ( 1 + tanh ( sqrt(2/PI) * (x + 0.44715 * x^3)  ) ) 
	// Optimize by factoring out one multiplication by f (reducing x*x*x to x*x)
	static float s_sqrt_2_div_pi = sqrtf(2.0f / AUSSIE_PI);
	return 0.5f * f *
		(1.0f
			+ tanhf(s_sqrt_2_div_pi *
				f *
				(1.0f + (0.44715f * (f * f)))
			)
			);
}

float aussie_sigmoid(float x)
{
	// SIGMOID = 1 / ( 1 + e^-x)
	return 1.0f / (1.0f + expf(-x));
}

float aussie_swish(float x, float beta)
{
	// SWISH = x * sigmoid(beta * x)
	return x * aussie_sigmoid(beta * x);
}

float aussie_swish2(float x, float beta)
{
	// SWISH = x * sigmoid(beta * x)
	// SIGMOID = 1 / ( 1 + e^-x)
	return x * ( 1.0f / (1.0f + expf(-(beta * x))));
}


float aussie_GELU_approx2(float x)   // Approximated Gaussian GELU
{
	// Another GELU approximation in the original 2016 paper...
	// GELU paper approx #2 = x * sigmoid (1.702 * x) 
	// SIGMOID = 1 / ( 1 + e^-x)
	// This is not very accurate, e.g. >0.01 errors in tests!
	return x * aussie_sigmoid(1.702f * x);

}

float aussie_GELU_approx2b(float x)   // Approximated Gaussian GELU
{
	// Another GELU approximation in the original 2016 paper...
	// GELU paper approx #2 = x * sigmoid (1.702 * x) 
	// SIGMOID = 1 / ( 1 + e^-x)
	// This is not super accurate, e.g. >0.01 errors in tests!
	// return x * 1.0 / (1.0 + expf(-(1.702 * x)));
	return x / (1.0f + expf(-1.702f * x));

}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
// GELU precomputation to 32 bits (doesn't work too well...)
//---------------------------------------------------

#if 0 // Doesn't compile!
float g_global_GELU_table_FP32[1 << 32 /*~4 billion*/];  // Is this really a good idea?
#endif

float g_global_GELU_table_FP32[1 << 16 /*~64k*/];  // Table for 16-bit precompute


float gelu_fast_FP32(float f)    // Table lookup GELU
{
	int i32 = *(int*)&f;
	return g_global_GELU_table_FP32[i32];   // FP32 version
}

float g_global_GELU_table_FP32_24bits[1UL << 24 /*~16Mx4 bytes = 64Meg*/];



void aussie_GELU_setup_table_FP32_24bits() // Initialize 24-bits GELU precomputed table
{
	static bool s_once = false;
	if (s_once) {
		yassert(!s_once);  // Should be once only
		return;  // Avoid double intialization!
	}
	s_once = true;
	aussie_precompute_table_FP32_generic_24bits(g_global_GELU_table_FP32_24bits, 1u << 24, aussie_GELU_basic);

#if 0
	unsigned long int u = 0;
	for (; u < (1 << 24); u++) {  // For all 2^24=~16.7M...
		unsigned int uval = u << 8;  // put zeros in the least significant 8 mantissa bits
		float f = AUSSIE_UINT_TO_FLOAT(uval);
		float g = aussie_GELU_basic(f);
		g_global_GELU_table_FP32_24bits[u] = g;  // Store precomputed GELU...
	}
#endif
}

void aussie_GELU_setup_table_FP32_24bits_PRINT_SOURCE( // Initialize 24-bits GELU precomputed table 
	char* nickname,
	char* outfname
)
{
	aussie_generic_setup_table_FP32_24bits_PRINT_SOURCE(
		nickname, 
		outfname, 
		aussie_GELU_basic,
		1u<<24,
		g_global_GELU_table_FP32_24bits
	);
	return;

#if 0 // Old code, new code in yprecompute.cpp
	// Generate C++ source code so we can pre-compile the precomputed GELU table (24-bits)
	// There are 2^24 = 16.7 million numbers...
	FILE* fp = stdout;
	bool writingfile = false;
	bool add_commented_number = true;
	if (outfname && *outfname) {
		fp = fopen(outfname, "w");
		if (!fp) {
			yassert(fp);
			return;  // fail
		}
		writingfile = true;
		add_commented_number = false;  // No extra comments for file output version
	}
	unsigned int u = 0;
	fprintf(fp, "// Precomputed table source code: %s, \"%s\"\n", nickname, outfname);
	fprintf(fp, "float g_gelu_table_precompute_24bits[] = { \n");
	char numbuf[5000] = "";
	for (; u < (1 << 24); u++) {  // For all 2^24=~16.7M...
		unsigned int uval = u << 8;  // put zeros in the least significant 8 mantissa bits
		float f = AUSSIE_UINT_TO_FLOAT(uval);
		float g = aussie_GELU_basic(f);
		g_global_GELU_table_FP32_24bits[u] = g;  // Store precomputed GELU...

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
	fprintf(fp, "} \n");  // Close initializer...
	if (fp && fp != stdout) fclose(fp);
#endif

}

float gelu_fast_FP32_24bits(float f)    // Table lookup GELU (using 24 bits)
{
	unsigned int u = AUSSIE_FLOAT_TO_UINT(f);
	u >>= 8;  // Cut least-significant 8 mantissa bits off
	return g_global_GELU_table_FP32_24bits[u];   // Look up in 24-bit FP32 version
}

void aussie_GELU_setup_table_FP32() // Initialize GELU precomputed table (huge 32-bit)
{
	yassert_once();
	yassert_N_times(1);

	static int s_calls = 0;
	++s_calls;
	if (s_calls > 1) {
		yassert(s_calls <= 1);
		return;  // Avoid double intialization!
	}
	unsigned long int i64 = 0;  // Has to be "long"!
	yassert(sizeof(i64) > 4);
	for (; i64 < (1UL << 32); i64++) {
		unsigned int ui32 = (int)i64;  // Switch down from 64-bit to 32-bit
		float f32 = *(float*)&ui32;  // Type-cast bit trick to get the float
		g_global_GELU_table_FP32[ui32] = aussie_GELU_basic2(f32);  // FP32
	}
}

float g_global_GELU_table_FP16[1UL << 16 /*~64k*/];  // FP16/INT16 (16-bit inputs)

void aussie_GELU_setup_table_FP16() // Initialize GELU precomputed table for FP16
{
	yassert_once();
	yassert_N_times(1);

	static int s_calls = 0;
	++s_calls;
	if (s_calls > 1) {
		yassert(s_calls <= 1);
		return;  // Avoid double intialization!
	}
	// NOTE: This isn't working yet!!!
	unsigned int i32 = 0;
	yassert(sizeof(i32) > 2);
	for (; i32 < (1UL << 16); i32++) {
		float f32 = *(float*)&i32;  // Type-cast bit trick to get the float
		g_global_GELU_table_FP32[i32] = aussie_GELU_basic2(f32);  // FP32
	}


}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------

//---------------------------------------------------

// Sigmoid Linear Unit (SiLU) 
// SiLU ... x * sigmoid(x) ...
// ... https://arxiv.org/abs/1702.03118v3 
// ... sigmoid(x) = 1.0 / ( 1.0 + expf(-x) )
// ... SiLU computation: x * 1.0 / (1.0 + expf(-x));
// 
// dSiLU -- see  https://arxiv.org/abs/1702.03118v3 
// ... derivative of sigmoid ?

float aussie_SiLU_basic(float x)   // Basic SiLU (inefficient)
{
	// Sigmoid = 1 + e^(-x)
	// SiLU = x * (1 + e^(-x) )
	//      = x * 1.0 / (1.0 + expf(-x));
	return x / (1.0f + expf(-x));
}

//---------------------------------------------------
//---------------------------------------------------

// Step function (basic)

float aussie_step_basic(float f)
{
	if (f < 0.0f) return 0.0f;
	return 1.0f;
}


//---------------------------------------------------
//---------------------------------------------------


void aussie_test_GELU(float f)
{
	float exp = aussie_GELU_basic(f);

	ytestf(exp, aussie_GELU_basic2(f));
	ytestf(exp, aussie_GELU_basic2(f));
	ytestfapprox(exp, aussie_GELU_approx1(f), 0.01);
	ytestfapprox(exp, aussie_GELU_approx1_optimized(f), 0.01);
	ytestfapprox(exp, aussie_GELU_approx1_optimized2(f), 0.01);
	ytestfapprox(exp, aussie_GELU_approx2(f), 0.02);
	ytestfapprox(exp, aussie_GELU_approx2b(f), 0.02);


}

void aussie_test_step_functions(float f, float expected)
{
	float f2 = aussie_step_basic(f);
	ytestf(f2, expected);

	float f3 = AUSSIE_STEP_FUNCTION(f);
	ytestf(f3, expected);

	float f4 = AUSSIE_STEP_FUNCTION2(f);
	ytestf(f4, expected);

	float f5 = AUSSIE_STEP_FUNCTION3(f);
	ytestf(f5, expected);

}
//---------------------------------------------------
//---------------------------------------------------

void aussie_activation_unit_tests()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	ytestf(aussie_RELU_basic(1.0), 1.0);

	aussie_test_GELU(3.0);
	aussie_test_GELU(1.0);

	float f = 1.0;  // Macro requires l-value
	ytesti(AUSSIE_ISNEGATIVE_FAST_FP32(f), 0);
	f = 0.0;
	ytesti(AUSSIE_ISNEGATIVE_FAST_FP32(f), 0);
	f = -1.0;
	ytesti(AUSSIE_ISNEGATIVE_FAST_FP32(f), 1);
	
	aussie_test_step_functions(1.0f, 1.0f);
	aussie_test_step_functions(-1.0f, 0.0f);
	aussie_test_step_functions(0.0f, 1.0f);  // Step of 0 is 1

}




//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

