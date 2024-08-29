// abitwise.cpp -- Low-level bitwise operations -- Aussie AI Base Library  
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
#include "aussieai.h"
#include "aassert.h"
#include "atest.h"

#include "abitwise.h"  // self-include


#if !LINUX
#include <intrin.h>   // MSVC intrinsics
#endif

//---------------------------------------------------

void aussie_unit_test_bitflags() 
{
	ytest(AUSSIE_ONE_BIT_SET(3, 1));
	ytest(!AUSSIE_ONE_BIT_SET(3, 4));

	ytest(AUSSIE_ANY_BITS_SET(3, 1));
	ytest(AUSSIE_ANY_BITS_SET(3, 3));

	ytest(AUSSIE_ALL_BITS_SET(3, 1));
	ytest(AUSSIE_ALL_BITS_SET(3, 3));
	ytest(!AUSSIE_ALL_BITS_SET(2, 1));
	ytest(!AUSSIE_ALL_BITS_SET(1, 3));

	ytest(!AUSSIE_NO_BITS_SET(3, 3));
	ytest(AUSSIE_NO_BITS_SET(3, 4));

	ytesti(AUSSIE_SET_BITS(3, 4), 7);
	ytesti(AUSSIE_SET_BITS(3, 1), 3);

	ytesti(AUSSIE_CLEAR_BITS(3, 4), 3);
	ytesti(AUSSIE_CLEAR_BITS(3, 1), 2);
	ytesti(AUSSIE_CLEAR_BITS(3, 3), 0);


}

void aussie_unit_test_one_popcount(unsigned int x, int expected)
{
	ytesti(aussie_popcount_basic(x), expected);
	ytesti(aussie_popcount_kernighan_algorithm(x), expected);
#if !LINUX
	ytesti(aussie_popcount_intrinsics1(x), expected);
	ytesti(aussie_popcount_intrinsics2(x), expected);
#endif //LINUX

#if !LINUX
	ytesti(AUSSIE_POPCOUNT_MACRO(x), expected);
#endif //LINUX
	


	
}

void aussie_unit_test_popcount()
{
	aussie_unit_test_one_popcount(3, 2);
	aussie_unit_test_one_popcount(2, 1);
	aussie_unit_test_one_popcount(0, 0);
	aussie_unit_test_one_popcount(1 << 2, 1);
	aussie_unit_test_one_popcount(1u << 16, 1);
	aussie_unit_test_one_popcount(1u << 31, 1);
	aussie_unit_test_one_popcount(15, 4);
	aussie_unit_test_one_popcount(255, 8);

}

void aussie_unit_test_bitwise()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
	aussie_unit_test_clz();
	aussie_unit_test_log2();
	aussie_unit_test_bitflags();
	aussie_unit_test_popcount();



	ytestui(AUSSIE_BITWISE_NAND(1, 1), -2);
	ytestui(AUSSIE_BITWISE_NAND(0, 0), AUSSIE_BITVALUE_ALL_BITS_SET);
	ytestui(AUSSIE_BITWISE_NAND(1, 0), AUSSIE_BITVALUE_ALL_BITS_SET);
	ytestui(AUSSIE_BITWISE_NAND(0, 1), AUSSIE_BITVALUE_ALL_BITS_SET);

	ytestui(AUSSIE_BITWISE_NOR(0, 0), AUSSIE_BITVALUE_ALL_BITS_SET);
	ytestui(AUSSIE_BITWISE_NOR(1, 0), -2);
	ytestui(AUSSIE_BITWISE_NOR(0, 1), -2);
	ytestui(AUSSIE_BITWISE_NOR(1, 1), -2);

	ytestui(AUSSIE_BITWISE_XNOR(0, 0), AUSSIE_BITVALUE_ALL_BITS_SET);
	ytestui(AUSSIE_BITWISE_XNOR(1, 0), -2);
	ytestui(AUSSIE_BITWISE_XNOR(0, 1), -2);
	ytestui(AUSSIE_BITWISE_XNOR(1, 1), AUSSIE_BITVALUE_ALL_BITS_SET);

}

//---------------------------------------------------
//---------------------------------------------------

int aussie_popcount_basic(unsigned int x) // Count number of 1's
{
	const int bitcount = 8 * sizeof(x);
	int ct = 0;
	for (int i = 0; i < bitcount; i++) {
		if (AUSSIE_ONE_BIT_SET(x, 1u << i)) ct++;
	}
	return ct;
}

//---------------------------------------------------

int aussie_popcount_kernighan_algorithm(unsigned int x) // Brian Kernighan algorithm
{
	// Count number of 1's with Kernighan bit trick
	const int bitcount = 8 * sizeof(x);
	int ct = 0;
	while (x != 0) {
		x = x & (x - 1);  // Remove rightmost 1 bit: n & (n-1)
		ct++;
	}
	return ct;
}

int aussie_popcount_intrinsics1(unsigned int x) // MSVC version &lt;intrin.h>
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// TODO: __builtin_popcount for GCC on Linux
	return _mm_popcnt_u32(x);  // Microsoft intrinsics MSVS
#endif //LINUX
}

int aussie_popcount_intrinsics2(unsigned int x) // MSVC version &lt;intrin.h>
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// TODO: __builtin_popcount for GCC on Linux
	return __popcnt(x);  // Microsoft intrinsics MSVS
#endif //LINUX
}

//---------------------------------------------------
// LOG2 tests
//---------------------------------------------------

void aussie_test_one_log2_integer(unsigned int u, int expected)
{
	ytesti(aussie_log2_integer_slow(u), expected);
	ytesti(aussie_log2_integer_clz(u), expected);
#if !LINUX
	ytesti(aussie_log2_integer_clz_intrinsic(u), expected);
#endif //LINUX
#if !LINUX
	ytesti(AUSSIE_LOG2_LZCNT(u), expected);
#endif


}

//---------------------------------------------------
//---------------------------------------------------

void aussie_unit_test_log2() // Log2 of integer unit tests
{
	aussie_test_one_log2_integer(1, 0);  // log2(1)==0
	aussie_test_one_log2_integer(2, 1);  // log2(1)==0
	aussie_test_one_log2_integer(3, 1);  // log2(1)==0
	aussie_test_one_log2_integer(4, 2);  // log2(1)==0
	aussie_test_one_log2_integer(256, 8);  // log2(1)==0
	aussie_test_one_log2_integer(255, 7);  // log2(1)==0

}

void aussie_unit_test_one_clz(unsigned int u, int expected)
{
	ytesti(aussie_clz_slow(u), expected);
#if !LINUX
	ytesti(aussie_clz_intrinsics1(u), expected); // __lzcnt
	ytesti(aussie_clz_intrinsics2(u), expected); // _BitScanReverse
#endif //LINUX
	

}

void aussie_unit_test_clz()
{
	aussie_unit_test_one_clz(1, 31);
	aussie_unit_test_one_clz(2, 30);
	aussie_unit_test_one_clz(3, 30);
	aussie_unit_test_one_clz(4, 29);
	aussie_unit_test_one_clz(8, 28);
	aussie_unit_test_one_clz((unsigned)-1, 0);
}

int aussie_clz_slow(unsigned int u)
{

	const int bitcount = 8 * sizeof(u);
	int ct = 0;
	for (int i = bitcount - 1; i >= 0; i--) {  // Left-to-right
		if (AUSSIE_ONE_BIT_SET(u, 1u << i)) {
			// Found a 1 bit, so we're done
			return ct;
		}
		ct++;  // Count this zero bit
	}
	return ct;
}

#if !LINUX
#include <intrin.h>  // Windows intrinsics (e.g. __lzcnt)
#endif

int aussie_clz_intrinsics1(unsigned int u)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// TODO: __builtin_clz on GCC for Linux
	return __lzcnt(u);  // Windows <intrin.h>
#endif //LINUX
}

#if !LINUX
#pragma intrinsic(_BitScanReverse)
#endif

int aussie_clz_intrinsics2(unsigned int u)
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	// TODO: __builtin_clz on GCC for Linux
	// _BitScanReverse or _BitScanForward
	unsigned long ulongret = 0;
	unsigned char foundbits = _BitScanReverse(&ulongret, u);  // Windows <intrin.h>
	if (foundbits) return 31 - (int)ulongret;
	return 32;  // Didn't find any bits set...
#endif //LINUX
}



//--------------------------------------------------------------
// LOG2 integer
//--------------------------------------------------------------

int aussie_log2_integer_slow(unsigned int u)  // Slow float-to-int version
{
	return (int)log2f((float)u);
}

int aussie_log2_integer_clz(unsigned int u)  // LOG2 using count-leading-zeros
{
	int clz = aussie_clz_slow(u);  // Count leading zeros
	const int bits = 8 * sizeof(u);
	return bits - clz - 1;
}

#if !LINUX
#include <intrin.h>  // Windows intrinsics (e.g. __lzcnt)
#endif

int aussie_log2_integer_clz_intrinsic(unsigned int u)  // LOG2 using CLZ
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
	int clz = __lzcnt(u);  // Count leading zeros
	const int bits = 8 * sizeof(u);
	return bits - clz - 1;
#endif //LINUX
}



//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
// IS-POWER-OF-TWO integer
//--------------------------------------------------------------

bool aussie_is_power_of_two_popcount(unsigned int u)
{
	return aussie_popcount_basic(u) == 1;
}

bool aussie_is_power_of_two_kernighan(unsigned int u)
{
	return (u & ( u - 1)) == 0; // True if only 1 bit found
}

//---------------------------------------------------
//---------------------------------------------------

