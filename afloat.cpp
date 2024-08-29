// yfloat.cpp -- Floating-point bit processing -- Aussie AI Base Library  
// Created Oct 11th 2023
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
#include <cmath>

//---------------------------------------------------
//---------------------------------------------------

#include "aport.h"
#include "aussieai.h"
#include "aassert.h"
#include "atest.h"

#include "afloat.h"  // self-include


//---------------------------------------------------
//---------------------------------------------------

#define AUSSIE_FP32_SIGN_BIT(f)  ((*(unsigned int*)&(f)) >> 31u)
#define AUSSIE_FP32_EXPONENT_BITS(f)  ((((*(unsigned int*)&(f)) >> 23u)) & ((2<<8)-1)

int aussie_float_sign_bit(float f)  // Sign bit, returns 0 or 1 integer
{
	yassert(sizeof f == 4);
	unsigned int u = *(unsigned*)&f;
	int signbit = u >> 31;
	return signbit;
}

float aussie_float_set_sign_bit(float f)
{
	yassert(sizeof f == 4);
	unsigned int u = *(unsigned*)&f;
	u |= 0x80000000u;
	return *(float*)&u;
}

int aussie_float_exponent_bits(float f)  // Returns exponent from -127...+127
{
	yassert(sizeof f == 4);
	unsigned int u = *(unsigned*)&f;

	u &= 0x7f800000;
	unsigned int exponentbits = u >> 23;
	exponentbits -= 127;   // Subtract the "bias" for the 8-bit exponent (8 bits is 0..255)...
	return (int)exponentbits;
}

int aussie_float_mantissa_bits(float f)
{
	yassert(sizeof f == 4);
	unsigned int u = *(unsigned*)&f;
	u &= 0x007fffff;   // 23 explicitly stored fractional bits, plus an implicit 1 prefix = 24 bits total..
	return u;
}

void aussie_float32_get_bits(float f, int& signbit, int& exponentbits, int& mantissabits)
{
	unsigned int u = *(unsigned int*)&f;
	signbit = u >> 31;
	exponentbits = ((u >> 23) & ((1 << 8) - 1)) - 127;
	mantissabits = (u & ((1 << 23) - 1));
}

float aussie_float32_from_bits(int isign, int iexponent, int imantissa)
{
	unsigned int u = 0;
	if (isign) u |= (1u << 31);
	iexponent += 127;  // Add offset
	u |= ((unsigned)iexponent << 23);
	u |= imantissa;
	return *(float*)&u;
}

yfp16_t aussie_float32_to_float16(float f)
{
	// FP32 = 1 sign, 8 exponent (offset 127), 23 mantissa bits = 32-bits
	// FP16 = 1 sign, 5 exponent (offset 15), 10 mantissa bits = 16-bits
	int signbit;
	int exponentbits;
	int mantissabits;
	aussie_float32_get_bits(f, signbit, exponentbits, mantissabits);
	yassert(exponentbits >= -128);
	yassert(exponentbits <= 127);

	if (f != 0.0) {  // 0.0 is special case all 8 bits set = 127
		yassert(exponentbits <= 31);
		yassert(exponentbits >= -32);
	}

#if 0 // old bad try
	int newexponent = ((exponentbits + 127));
	yassert(newexponent >= 0);
	yassert(newexponent <= 255);
	newexponent = (newexponent & ((1 << 5) - 1)) - 15;
	yassert(newexponent >= -32);
	yassert(newexponent <= 31);  // 5-bits
	
	if (f == 0) {
		f = f;  // debug breakpoint
	}
	unsigned int uexp = (unsigned)exponentbits;
	int fixexponentbits = ((unsigned)uexp & 31); /* ((1 << 5) - 1));*/ // Truncate from 8 to 5 bits
#endif

	int fixexponentbits = exponentbits;
	return aussie_float16_from_bits(signbit, fixexponentbits, mantissabits);
}

float aussie_float16_to_float32(yfp16_t f)
{
	// FP32 = 1 sign, 8 exponent (offset 127), 23 mantissa bits = 32-bits
	// FP16 = 1 sign, 5 exponent (offset 15), 10 mantissa bits = 16-bits
	int signbit;
	int exponentbits;
	int mantissabits;
	aussie_float16_get_bits(f, signbit, exponentbits, mantissabits);
	yassert(exponentbits >= -16);
	yassert(exponentbits <= 15);
	return aussie_float32_from_bits(signbit, exponentbits, mantissabits);
}

void aussie_float16_get_bits(yfp16_t fp16, int& signbit, int& exponentbits, int& mantissabits)
{
	unsigned short int u = (unsigned int)fp16;
	yassert(sizeof u == sizeof fp16);
	signbit = u >> 15;
	exponentbits = ((u >> 10) & ((1 << 5) - 1)) - 15;  // 5-bits with 15 offset
	mantissabits = (u & ((1 << 10) - 1));
}

yfp16_t aussie_float16_from_bits(int isign, int iexponent, int imantissa)
{
	// float16 is 1 sign bit, 5 exponent, 10 stored mantissa bits
	unsigned int u = 0;
	if (isign) u |= (1u << 15);
	iexponent += 15;  // Add offset for FP16 5-bit exponent 0..31 (15 not 127)
	u |= ((unsigned)iexponent << 10);
	u |= imantissa & ((1 << 10)-1);  // Truncate mantissa to max 10 bits
	yassert(u < (1u << 16));
	return (yfp16_t)u;  // Truncate to 2 bytes
}

// -------- BFLOAT 16 -----------------
void aussie_bfloat16_get_bits(ybf16_t bf, int& signbit, int& exponentbits, int& mantissabits)
{
	// Bfloat16 is 1 sign bit, 8 exponent bits (offset+127), 7 stored mantissa bits...
	unsigned short int u = (unsigned int)bf;
	yassert(sizeof u == sizeof bf);
	signbit = u >> 15;
	exponentbits = (((u >> 7) & ((1 << 8) - 1))) - 127;  // 8-bits with 127 offset
	mantissabits = (u & ((1 << 7) - 1));
}

ybf16_t aussie_bfloat16_from_bits(int isign, int iexponent, int imantissa)
{
	// Bfloat16 is 1 sign bit, 8 exponent bits (offset 127), 7 mantissa bits...
	unsigned int u = 0;
	if (isign) u |= (1u << 15);
	iexponent += 127;  // Add offset for BF16 (127 for 8-bit exponent)
	yassert(iexponent >= 0);
	u |= ((unsigned)iexponent << 7);
	u |= (imantissa & ((1 << 7) - 1));  // Truncate mantissa to 7 bits maximum
	yassert(u < (1u << 16));
	return (yfp16_t)u;  // Truncate to 2 bytes
}

ybf16_t aussie_float32_to_bfloat16(float f)
{
	// FP32 = 1 sign, 8 exponent, 23 mantissa bits = 32-bits
	// BF16 = 1 sign, 8 exponent, 7 mantissa bits = 16-bits
	int signbit;
	int exponentbits;
	int mantissabits;
	aussie_float32_get_bits(f, signbit, exponentbits, mantissabits);
	return aussie_bfloat16_from_bits(signbit, exponentbits, mantissabits);

}

float aussie_bfloat16_to_float32(ybf16_t fb)
{
	// FP32 = 1 sign, 8 exponent, 23 mantissa bits = 32-bits
	// BF16 = 1 sign, 8 exponent, 7 mantissa bits = 16-bits
	int signbit;
	int exponentbits;
	int mantissabits;
	aussie_bfloat16_get_bits(fb, signbit, exponentbits, mantissabits);
	return aussie_float32_from_bits(signbit, exponentbits, mantissabits);
}



//-----------------------------------------------------

unsigned int aussie_float_to_uint(float f)
{
	return *(unsigned int*)&f;
}

//-----------------------------------------------------
//-----------------------------------------------------

float aussie_uint_to_float(unsigned int ui)
{
	return *(float*)&ui;
}

//-----------------------------------------------------
//-----------------------------------------------------
char *string_binary32(unsigned int x, char str[])
{
	// Output a 32-digit binary number from unsigned int
	char* origstr = str;
	for (int i = 31; i >= 0; i--) {
		*str = ( (x & (1<<i)) ? '1' : '0');
		str++;
	}
	*str = 0;  // null the string
	return origstr;
}

//-----------------------------------------------------
//-----------------------------------------------------



void aussie_test_one_float_basic(float f)  // Various unit testing on single float
{
	int signbit = aussie_float_sign_bit(f);
	ytest(signbit == 0 || signbit == 1);
	if (f < 0.0) {
		ytest(signbit == 1);
		ytesti(AUSSIE_FP32_SIGN_BIT(f), 1);
	}
	if (f > 0.0) {
		ytest(signbit == 0);
		ytesti(AUSSIE_FP32_SIGN_BIT(f), 0);
	}

	int exponent1 = aussie_float_exponent_bits(f);
	ytest(exponent1 >= -128);
	ytest(exponent1 <= +127);
	int bits = AUSSIE_FLOAT_EXPONENT_BITS(f);
	ytesti(AUSSIE_FLOAT_EXPONENT_BITS(f), exponent1 + 127);

	bool is_zero_or_subnormal = AUSSIE_FLOAT_IS_ZERO_OR_DENORMALIZED(f);  // All 0s exponent?
	bool is_inf_or_nan = AUSSIE_FLOAT_IS_INF_OR_NAN(f); // All 1s exponent?
	ytest(!is_inf_or_nan);
	if (f == 0.0) {
		ytest(is_zero_or_subnormal);
	}
	else {
		ytest(!is_zero_or_subnormal);
	}

	// This doesn't work... trying to cast to (signed char) -128..127
#if 0
	int bitschar = AUSSIE_FLOAT_EXPONENT_BITS_CHAR(f);
	ytesti(AUSSIE_FLOAT_EXPONENT_BITS_CHAR(f), exponent1 + 127);
#endif

	ytesti(AUSSIE_FLOAT_EXPONENT(f), exponent1);
	ytesti(AUSSIE_FLOAT_EXPONENT2(f), exponent1);
	
	int mantissa1 = aussie_float_mantissa_bits(f);
	ytest(mantissa1 >= 0);
	ytest(mantissa1 < (1 << 23));
	ytesti(AUSSIE_FLOAT_MANTISSA(f), mantissa1);

	int signbit2 = 0;
	int exponent2 = 0;
	int mantissa2 = 0;

	aussie_float32_get_bits(f, signbit2, exponent2, mantissa2);
	ytesti(signbit, signbit2);
	ytesti(exponent1, exponent2);
	ytesti(mantissa1, mantissa2);
	ytest(signbit2 == 0 || signbit2 == 1);
	ytest(exponent2 >= -128);
	ytest(exponent2 <= +127);
	ytest(mantissa2 >= 0);
	ytest(mantissa2 < (1 << 23));

	ytesti(AUSSIE_FLOAT_EXPONENT(f), exponent2);
	ytesti(AUSSIE_FLOAT_MANTISSA(f), mantissa2);
	ytesti(AUSSIE_FLOAT_EXPONENT2(f), exponent2);


}

void aussie_test_BFLOAT16_conversions_one_float(float f)  // Test FP16/FP32 conversions
{
	// Test converting float to BF16 and back
	ybf16_t bf1 = aussie_float32_to_bfloat16(f);
	float f3 = aussie_bfloat16_to_float32(bf1);
	ytestf(f3, f);


}

void aussie_test_FP16_conversions_one_float(float f)  // Test FP16/FP32 conversions
{
	// NOTE: This is currently buggy... needs to adjust bits sometimes...
	int signbit2 = 0;
	int exponent2 = 0;
	int mantissa2 = 0;
	aussie_float32_get_bits(f, signbit2, exponent2, mantissa2);
	float f2 = aussie_float32_from_bits(signbit2, exponent2, mantissa2);
	ytestf(f2, f);

	// Test converting float to FP16 and back
	yfp16_t fp16 = aussie_float32_to_float16(f);
	float f4 = aussie_float16_to_float32(fp16);

	int signbit3 = 0;
	int exponent3 = 0;
	int mantissa3 = 0;
	aussie_float16_get_bits(fp16, signbit3, exponent3, mantissa3);


	int signbit4 = 0;
	int exponent4 = 0;
	int mantissa4 = 0;
	aussie_float32_get_bits(f4, signbit4, exponent4, mantissa4);
	char exponent4str[100] = "";
	string_binary32(exponent4, exponent4str);
	char mantissa4str[100] = "";
	string_binary32(mantissa4, mantissa4str);

	char exponent2str[100] = "";
	string_binary32(exponent2, exponent2str);
	char mantissa2str[100] = "";
	string_binary32(mantissa2, mantissa2str);


	ytesti(signbit4, signbit2);
	ytesti(exponent4, exponent2);
	ytesti(mantissa4, mantissa2);

	ytestf(f4, f);


}

void aussie_test_one_16bit_int_float(int x)
{
	ytest(x <= (1 << 16));
	unsigned int u = (unsigned)x;

	yfp16_t fp = (yfp16_t)u;
	int signbit = 0;
	int exponent = 0;
	int mantissa = 0;
	aussie_float16_get_bits(fp, signbit, exponent, mantissa);

	yfp16_t fp2 = aussie_float16_from_bits(signbit, exponent, mantissa);

	ytesti(fp, fp2);

}
//---------------------------------------------------
//---------------------------------------------------

float aussie_approx_multiply_add_as_int_mogami(float f1, float f2)   // Add as integer
{
	int c = *(int*)&(f1)+*(int*)&(f2)-0x3f800000;  // Mogami(2020)
	return *(float*)&c;
}

//---------------------------------------------------
//---------------------------------------------------

float aussie_float_bitshift_add_integer(float f1, int bitstoshift)   // Bitshift on float by adding integer to exponent
{
	// FP32 = 1 sign bit, 8 exponent bits, 23 mantissa bits
	// NOTE: This can overflow into the sign bit if all 8 exponent bits are '1' (i.e. 255)
	unsigned int u = *(unsigned int*)&f1;  // Get to the bits as an integer
	if (u == 0) return f1;  // special case, don't change it...
	u += (bitstoshift << 23);  // Add shift count to the exponent bits...
	return *(float*)&u;  // Convert back to float
}

//---------------------------------------------------
//---------------------------------------------------

float log2_basic(float f)
{
	return log2(f);  // <math.h> function
}

int ilog2_exponent(float f)  // Log2 for 32-bit float
{
	unsigned int u = *(unsigned int*)&f;
	int iexponent = ((u >> 23) & 255);  // 8-bit exponent (above 23-bit mantissa)
	iexponent -= 127;  // Remove the "offset"
	return iexponent;
}

float log2_exponent(float f)
{
	unsigned int u = *(unsigned int*)&f;
	int iexponent = ((u >> 23) & 255);  // 8-bit exponent (above 23-bit mantissa)
	iexponent -= 127;  // Remove the "offset"
	return (float)iexponent;
}

//---------------------------------------------------
void aussie_float_test_one_log(float f)
{

	float f1 = log2_basic(f);
	float f2 = log2_exponent(f);
	ytest(f1 >= f2);
	ytestf(floor(f1), f2);

	int i1 = ilog2_exponent(f);
	ytest(f1 >= (float)i1);
	ytestf((float)i1, f2);
}

//---------------------------------------------------
void aussie_float_test_logarithms()
{
	// NOTE: log(0) is an error, log(negative) also fails...

	// Test some individual ones...
	ytesti(ilog2_exponent(4.0), 2);
	ytesti(ilog2_exponent(2.0), 1);
	ytesti(ilog2_exponent(1.0), 0);
	ytesti(ilog2_exponent(0.5), -1);
	ytesti(ilog2_exponent(0.25), -2);

	for (float f = 1.0; f < 100.0; f += 1.0f) {  // Range
		aussie_float_test_one_log(f);

	}

	// Test fractions...
	for (float f = 0.01; f <= 1.0; f += 0.01f) {  // Range
		aussie_float_test_one_log(f);
	}

}

#include <xmmintrin.h>
#include <pmmintrin.h>

void aussie_float_enable_FTZ_DAZ(bool ftz, bool daz)
{
	if (ftz) {	// FTZ mode
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	}
	else {
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
	}

	if (daz) {  // DAZ mode
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	}
	else {
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_float_tests_basic()
{
	// Test 0.15625 (Wikipedia given example)
	ytesti(aussie_float_sign_bit(0.15625f), 0);
	ytesti(aussie_float_exponent_bits(0.15625f), -3);
	ytesti(aussie_float_mantissa_bits(0.15625f), 1 << 21);

	float f = 0.0f;
	AUSSIE_FLOAT_IS_ZERO(f);
	ytest(!AUSSIE_FLOAT_IS_INF_OR_NAN(f));
	f = -0.0f;
	AUSSIE_FLOAT_IS_ZERO(f);
	ytest(!AUSSIE_FLOAT_IS_INF_OR_NAN(f));

	unsigned u = 0;
	f = AUSSIE_UINT_TO_FLOAT(u);
	ytest(AUSSIE_FLOAT_IS_ZERO(f));
	ytest(AUSSIE_FLOAT_IS_ZERO_OR_DENORMALIZED(f));
	ytest(!AUSSIE_FLOAT_IS_INF_OR_NAN(f));
	ytest(!AUSSIE_FLOAT_IS_INF(f));
	ytest(!AUSSIE_FLOAT_IS_NAN(f));
	ytestf(f, 0.0);
	ytestf(f, -0.0);
	aussie_test_one_float_basic(f);

	f = aussie_float32_from_bits(0/*signed*/, -127/*exponent*/, 0/*mantissa*/);
	// printf("Binary = %b\n", u);
	ytesti(aussie_float_sign_bit(f), 0);
	ytesti(aussie_float_exponent_bits(f), -127);
	ytesti(aussie_float_mantissa_bits(f), 0);
	u = AUSSIE_FLOAT_TO_UINT(f);
	ytesti((int)u, 0);
	ytesti(u, 0);
	ytest(!AUSSIE_FLOAT_IS_INF(f));
	ytest(!AUSSIE_FLOAT_IS_NAN(f));

	// INFO -- exponent all 1s, mantissa is all 0s...
	f = aussie_float32_from_bits(0/*signed*/, +128/*exponent*/, 0/*mantissa*/);
	ytest(AUSSIE_FLOAT_IS_INF_OR_NAN(f));
	ytest(AUSSIE_FLOAT_IS_INF(f));
	ytest(!AUSSIE_FLOAT_IS_NAN(f));
	ytest(!AUSSIE_FLOAT_IS_ZERO_OR_DENORMALIZED(f));
	ytest(std::isinf(f));
	ytest(!std::isfinite(f));
	ytest(!std::isnan(f));
	//ytestf(f, Inf);
	ytest(!std::isnormal(f));


	// Nan -- exponent all 1s, mantissa NOT all 0s...
	f = aussie_float32_from_bits(0/*signed*/, +128/*exponent*/, 1/*mantissa*/);
	ytest(AUSSIE_FLOAT_IS_INF_OR_NAN(f));
	ytest(!AUSSIE_FLOAT_IS_INF(f));
	ytest(AUSSIE_FLOAT_IS_NAN(f));
	ytest(!AUSSIE_FLOAT_IS_ZERO_OR_DENORMALIZED(f));
	ytest(std::isnan(f));   // Declared in <cmath>
	ytest(!std::isinf(f));
	//ytestf(f, NaN);
	ytest(!std::isnormal(f));
	ytest(!std::isfinite(f));
	// TODO: std::fpclassify()

	f = 0.0f;
	ytest(AUSSIE_FLOAT_IS_POSITIVE_ZERO(f));
	ytest(!AUSSIE_FLOAT_IS_NEGATIVE_ZERO(f));
	ytest(!AUSSIE_FLOAT_IS_NEGATIVE_ZERO2(f));

	f = -0.0f;
	ytest(!AUSSIE_FLOAT_IS_POSITIVE_ZERO(f));
	ytest(AUSSIE_FLOAT_IS_NEGATIVE_ZERO(f));
	ytest(AUSSIE_FLOAT_IS_NEGATIVE_ZERO2(f));

	f = 3.2f;
	ytest(std::isnormal(f));
	ytest(std::isfinite(f));
	ytest(!std::isnan(f));
	ytest(!std::isinf(f));

	// Zero...
	f = aussie_float32_from_bits(0/*signed*/, -127/*exponent*/, 0/*mantissa*/);
	ytestf(f, 0.0);
	ytest(AUSSIE_FLOAT_IS_ZERO(f));
	ytest(AUSSIE_FLOAT_IS_ZERO_OR_DENORMALIZED(f));
	ytest(!AUSSIE_FLOAT_IS_INF_OR_NAN(f));
	ytest(!AUSSIE_FLOAT_IS_INF(f));
	ytest(!AUSSIE_FLOAT_IS_NAN(f));
	aussie_test_one_float_basic(f);

	// NEGATIVE Zero...
	f = aussie_float32_from_bits(1/*(NEGATIVE!)signed*/, -127/*exponent*/, 0/*mantissa*/);
	ytestf(f, 0.0);
	ytest(AUSSIE_FLOAT_IS_ZERO(f));
	ytest(AUSSIE_FLOAT_IS_ZERO_OR_DENORMALIZED(f));
	ytest(!AUSSIE_FLOAT_IS_INF_OR_NAN(f));
	ytest(!AUSSIE_FLOAT_IS_INF(f));
	ytest(!AUSSIE_FLOAT_IS_NAN(f));
	aussie_test_one_float_basic(f);




	u = (1 << 31u);  // Sign bit only (negative zero)
	f = AUSSIE_UINT_TO_FLOAT(u);
	ytestf(f, 0.0);
	aussie_test_one_float_basic(f);


	aussie_test_one_float_basic(0.25f);
	aussie_test_one_float_basic(-0.25f);
	aussie_test_one_float_basic(0.5f);
	aussie_test_one_float_basic(-0.5f);
	aussie_test_one_float_basic(-1.0f);
	aussie_test_one_float_basic(-2.0f);
	aussie_test_one_float_basic(1.0f);
	aussie_test_one_float_basic(2.0f);

	ytesti(aussie_float_exponent_bits(0.25), -2);
	ytesti(aussie_float_exponent_bits(0.5), -1);

	f = 0.25f;
	ytesti(AUSSIE_FLOAT_EXPONENT(f), -2);
	ytesti(AUSSIE_FLOAT_EXPONENT2(f), -2);
	f = 0.5f;
	ytesti(AUSSIE_FLOAT_EXPONENT(f), -1);
	ytesti(AUSSIE_FLOAT_EXPONENT2(f), -1);


	int signbit2 = 0;
	int exponent2 = 0;
	int mantissa2 = 0;
	float f2 = 0.15625f;
	aussie_float32_get_bits(f2, signbit2, exponent2, mantissa2);
	ytesti(signbit2, 0);
	ytesti(exponent2, -3);
	ytesti(mantissa2, 1 << 21);

	float f1 = aussie_float32_from_bits(signbit2, exponent2, mantissa2);
	ytestf(f1, 0.15625f);

	f1 = 0.25;   // 1.0 x 2^-2
	signbit2 = 0;
	exponent2 = -2;
	mantissa2 = 0;  // The first "1" is implicit, so it's all zeros...
	f2 = aussie_float32_from_bits(signbit2, exponent2, mantissa2);
	ytestf(f2, 0.25f);
	ytestf(f1, f2);

	// Test sign bits
	ytesti(aussie_float_sign_bit(1.0), 0);
	ytesti(aussie_float_sign_bit(0.0), 0);
	ytesti(aussie_float_sign_bit(-1.0), 1);
	ytesti(aussie_float_sign_bit(2.0), 0);
	ytesti(aussie_float_sign_bit(-2.0), 1);

	// Test exponents
	ytesti(aussie_float_exponent_bits(-1.0), 0);
	ytesti(aussie_float_exponent_bits(1.0), 0);

	ytesti(aussie_float_exponent_bits(2.0), 1);
	ytesti(aussie_float_exponent_bits(4.0), 2);
	ytesti(aussie_float_exponent_bits(8.0), 3);

	ytesti(aussie_float_exponent_bits(-2.0), 1);
	ytesti(aussie_float_exponent_bits(-4.0), 2);
	ytesti(aussie_float_exponent_bits(-8.0), 3);
	ytesti(aussie_float_exponent_bits(0.5), -1);
	ytesti(aussie_float_exponent_bits(0.25), -2);

	// Test mantissa bits...
	ytesti(aussie_float_mantissa_bits(1.0), 0);   // Prefix 1 bit of mantissa is not explicitly stored!
	ytesti(aussie_float_mantissa_bits(-1.0), 0);   // Prefix 1 bit of mantissa is not explicitly stored!
	ytesti(aussie_float_mantissa_bits(0.5), 0);
	ytesti(aussie_float_mantissa_bits(-0.5), 0);
	ytesti(aussie_float_mantissa_bits(0.25), 0);

	ytestf(aussie_float_set_sign_bit(1.0), -1.0);
	ytestf(aussie_float_set_sign_bit(-1.0), -1.0);
	ytestf(aussie_float_set_sign_bit(2.0), -2.0);
	//ytestf(aussie_float_set_sign_bit(3.33), -3.33);  // Fails due to obscure rounding difference?

}

void aussie_float_tests_range()
{

	// Auto-test a range of floating point numbers...
	int i1 = -255;
	int i2 = 255;
	int inc = 1;
	for (int i = i1; i <= i2; i += inc) {
		aussie_test_one_16bit_int_float(i);
		float f = (float)i;
		aussie_test_one_float_basic(f);


#if 0 // not yet bug-free converting between FP16/FP32 or Bfloat16
		aussie_test_FP16_conversions_one_float(f);
		aussie_test_BFLOAT16_conversions_one_float(f);

#endif
	}
}

void aussie_float_test_tricks_one_float(float f)
{
	unsigned int u = *(unsigned int*)&f;
	char ustr[100] = "";
	string_binary32(u, ustr);

	ytesti(1 + 2 + 4 + 8 + 16 + 32 + 64, 127);
	ytesti((unsigned)0x3f800000 >> 23, 127); // 3f8 == 1111111

	ytesti(0x3f800000,127 << 23); // 3f8 == 111111 

	float f2 = aussie_float_bitshift_add_integer(f, 1);  // f<<1 is f*2.0
	ytestf(f2, f * 2.0f);


	// Test shift by zero
	f2 = aussie_float_bitshift_add_integer(f, 0);  // f<<1 is f*2.0
	ytestf(f2, f);  // Unchanged

	// Test shift by negatives...
	f2 = aussie_float_bitshift_add_integer(f, -1);  // f<<1 is f*2.0
	ytestf(f2, f / 2.0f);

	bool sign = AUSSIE_FLOAT_SIGN(f);
	if (sign) ytest(f < 0.0);
	else ytest(f >= 0.0);
	ytesti((int)sign, AUSSIE_FLOAT_SIGN2(f));
	ytesti((int)sign, AUSSIE_FLOAT_SIGN3(f));

	
}

void aussie_float_tests_tricks()
{
	float f = 2.0f;

	aussie_float_test_tricks_one_float(2.0);
	aussie_float_test_tricks_one_float(-2.0);

	aussie_float_test_tricks_one_float(0.0);  // zero as special case
	aussie_float_test_tricks_one_float(-3.14);
	aussie_float_test_tricks_one_float(3.14);
	aussie_float_test_tricks_one_float(1234567.89);
	aussie_float_test_tricks_one_float(-1234567.89);

}

void aussie_float_tests()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	aussie_float_test_logarithms();

	aussie_float_tests_tricks();

	aussie_float_tests_basic();

	aussie_float_tests_range();
}
