// abitwise.h -- Low-level bitwise operations -- Aussie AI Base Library  
// Created Oct 12th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YBITWISE_INCLUDE_HEADER_H
#define AUSSIE_YBITWISE_INCLUDE_HEADER_H

//--------------------------------------------------------------
// NAND, NOR, XNOR
//--------------------------------------------------------------

#define AUSSIE_BITWISE_NAND(x,y)  ( ~ ( ((unsigned int)x) & ((unsigned int)y) ) )
#define AUSSIE_BITWISE_NOR(x,y)  ( ~ ( ((unsigned int)x) | ((unsigned int)y) ) )
#define AUSSIE_BITWISE_XNOR(x,y)  ( ~ ( ((unsigned int)x) ^ ((unsigned int)y) ) )

//--------------------------------------------------------------
//--------------------------------------------------------------

// Bit Flags in Integers
#define AUSSIE_ONE_BIT_SET(x, bit)    (( ((unsigned)(x)) & ((unsigned)(bit))) != 0 )
#define AUSSIE_ANY_BITS_SET(x, bits)  (( ((unsigned)(x)) & ((unsigned)(bits))) != 0 )
#define AUSSIE_ALL_BITS_SET(x, bits)  (( ((unsigned)(x)) & ((unsigned)(bits))) == ((unsigned)(bits)) )
#define AUSSIE_NO_BITS_SET(x, bits)   (( ((unsigned)(x)) & ((unsigned)(bits))) == 0 )
#define AUSSIE_SET_BITS(x, bits)      (( ((unsigned)(x)) | ((unsigned)(bits))))
#define AUSSIE_CLEAR_BITS(x, bits)    (( ((unsigned)(x)) & (~((unsigned)(bits)))))

//--------------------------------------------------------------
// Unit Testing
//--------------------------------------------------------------

void aussie_unit_test_bitwise();  // Unit tests
void aussie_unit_test_popcount(); // Popcount unit tests
void aussie_unit_test_bitflags(); // Basic bit flag unit testing
void aussie_unit_test_log2(); // Log2 of integer unit tests
void aussie_test_one_log2_integer(unsigned int u, int expected);

//--------------------------------------------------------------
// POPCOUNT -- count the number of 1's in unsigned integer...
//--------------------------------------------------------------
int aussie_popcount_basic(unsigned int x);  // Count number of 1's
int aussie_popcount_kernighan_algorithm(unsigned int x); // Count number of 1's
int aussie_popcount_intrinsics1(unsigned int x); // MSVC version <intrin.h>
int aussie_popcount_intrinsics2(unsigned int x); // MSVC version <intrin.h>
#define AUSSIE_POPCOUNT_MACRO(x) ( __popcnt((unsigned int)(x)) )

//--------------------------------------------------------------
// LOG2 integer
//--------------------------------------------------------------
int aussie_log2_integer_slow(unsigned int u);
int aussie_log2_integer_clz(unsigned int u);  // LOG2 using count-leading-zeros;
int aussie_log2_integer_clz_intrinsic(unsigned int u);  // LOG2 using CLZ
#define AUSSIE_LOG2_LZCNT(u)  ( (8 * sizeof(unsigned)) - (__lzcnt((unsigned)(u))) - 1 )


//--------------------------------------------------------------
// CLZ (Count Leading Zero Bits)
//--------------------------------------------------------------

int aussie_clz_slow(unsigned int u);
int aussie_clz_intrinsics1(unsigned int u);
int aussie_clz_intrinsics2(unsigned int u);

// Test CLZ (Count Leading Zero Bits)
void aussie_unit_test_one_clz(unsigned int u, int expected);
void aussie_unit_test_clz();



//--------------------------------------------------------------
// IS-POWER-OF-TWO integer
//--------------------------------------------------------------
bool aussie_is_power_of_two_popcount(unsigned int u);

//--------------------------------------------------------------
// MISC
//--------------------------------------------------------------

#define AUSSIE_BITVALUE_ALL_BITS_SET (-1)

//--------------------------------------------------------------
//--------------------------------------------------------------

#endif //AUSSIE_YBITWISE_INCLUDE_HEADER_H

