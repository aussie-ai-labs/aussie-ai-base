// afloat.h -- Floating-point bit processing -- Aussie AI Base Library  
// Created Oct 11th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YFLOAT_INCLUDE_HEADER_H
#define AUSSIE_YFLOAT_INCLUDE_HEADER_H

//------------------------------------------------------------
// Logarithm calculations
//------------------------------------------------------------
float log2_basic(float f);
float log2_exponent(float f);
int ilog2_exponent(float f);

//------------------------------------------------------------
// Unit tests -- Logarithm calculations
//------------------------------------------------------------
void aussie_float_test_one_log(float f);
void aussie_float_test_logarithms();

//------------------------------------------------------------
// Floating point bitwise computations (single float number)
int aussie_float_sign_bit(float f);  // Sign bit, returns 0 or 1 integer
float aussie_float_set_sign_bit(float f);
int aussie_float_exponent_bits(float f);  // Returns exponent from -127...+127
int aussie_float_mantissa_bits(float f);

//------------------------------------------------------------
//------------------------------------------------------------

unsigned int aussie_float_to_uint(float f);
float aussie_uint_to_float(unsigned int);
#define AUSSIE_FLOAT_TO_UINT(f)  (*(unsigned int*)&f)
#define AUSSIE_UINT_TO_FLOAT(ui)  (*(float*)&ui)

//------------------------------------------------------------
//------------------------------------------------------------

void aussie_float32_get_bits(float f, int& signbit, int& exponentbits, int& mantissabits);
float aussie_float32_from_bits(int isign, int iexponent, int imantissa);

//------------------------------------------------------------
// FP16 float type
//------------------------------------------------------------

typedef unsigned short int yfp16_t;  // 16-bit type
void aussie_float16_get_bits(yfp16_t f, int& signbit, int& exponentbits, int& mantissabits);
yfp16_t aussie_float16_from_bits(int isign, int iexponent, int imantissa);
yfp16_t aussie_float32_to_float16(float f);
float aussie_float16_to_float32(yfp16_t f);

//------------------------------------------------------------
// Bfloat16 type (brain float)
//------------------------------------------------------------

typedef unsigned short int ybf16_t;  // 16-bit type
void aussie_bfloat16_get_bits(ybf16_t f, int& signbit, int& exponentbits, int& mantissabits);
ybf16_t aussie_bfloat16_from_bits(int isign, int iexponent, int imantissa);
ybf16_t aussie_float32_to_bfloat16(float f);
float aussie_bfloat16_to_float32(ybf16_t f);

//------------------------------------------------------------
// Bit twicks on floats...
//------------------------------------------------------------

float aussie_approx_multiply_add_as_int_mogami(float f1, float f2);   // Add as integer
float aussie_float_bitshift_add_integer(float f1, int bitstoshift);   // Bitshift on float by adding integer to exponent

//------------------------------------------------------------
// Unit tests for floating point
//------------------------------------------------------------

void aussie_float_tests();
void aussie_float_tests_basic();
void aussie_test_one_float(float f);
void aussie_float_tests_range();
void aussie_float_test_tricks_one_float(float f);

//------------------------------------------------------------
// Floating point bit manipulations
//------------------------------------------------------------
#define AUSSIE_FLOAT_SIGN(f)   (( (*(unsigned *)&(f)) & (unsigned)(1u<<31)) != 0)
#define AUSSIE_FLOAT_SIGN2(f)  ( (*(unsigned *)&(f)) >> 31u)   // Leftmost bit
#define AUSSIE_FLOAT_SIGN3(f)  ( (f) < 0.0f)   // Negatives

#define AUSSIE_FLOAT_EXPONENT_BITS(f)  (int)( (( (*(unsigned*)&(f)) & 0x7f800000u )>> 23u) /*-127*/ ) 

#if 0 // This idea doesn't work  cast to signed char...
#define AUSSIE_FLOAT_EXPONENT_BITS_CHAR(f)  (signed char)( (( (*(unsigned*)&(f)) & 0x7f800000u )>> 23u) /*-127*/ ) 
#endif

#define AUSSIE_FLOAT_EXPONENT(f)  ((int)( ((int)( (*(unsigned*)&(f)) & 0x7f800000u )>> 23u) - 127 )) 
#define AUSSIE_FLOAT_EXPONENT2(f)  (int)( (int)((( (*(unsigned*)&(f)) )>> 23u) & 255u ) - 127 ) 

#define AUSSIE_FLOAT_MANTISSA(f)  ((*(unsigned*)&(f)) & 0x007fffffu)  // Rightmost 23 bits

#define AUSSIE_FLOAT_IS_ZERO_OR_DENORMALIZED(f) (AUSSIE_FLOAT_EXPONENT_BITS(f) == 0)  // All 0s
#define AUSSIE_FLOAT_IS_INF_OR_NAN(f) (AUSSIE_FLOAT_EXPONENT_BITS(f) == 255)  // All 1s
#define AUSSIE_FLOAT_IS_ZERO(f)  ((( AUSSIE_FLOAT_TO_UINT(f) &(~(1<<31))) == 0))  // All 0s

#define AUSSIE_FLOAT_IS_POSITIVE_ZERO(f)  ((( AUSSIE_FLOAT_TO_UINT(f) )) == 0)  // All 0s

#define AUSSIE_FLOAT_TO_UINT(f)  (*(unsigned int*)&f)

#define AUSSIE_FLOAT_IS_NEGATIVE_ZERO(f)  ((( AUSSIE_FLOAT_TO_UINT(f) )) == (1u<<31))  // Sign bit only
#define AUSSIE_FLOAT_IS_NEGATIVE_ZERO2(f) ( ( (f) == 0.0f ) && ! AUSSIE_FLOAT_IS_POSITIVE_ZERO(f))

// INF: Exponent all 1s, Mantissa all 0s
#define AUSSIE_FLOAT_IS_INF(f) (AUSSIE_FLOAT_EXPONENT_BITS(f) == 255 && (AUSSIE_FLOAT_MANTISSA(f)==0)) 

// NaN: Exponent all 1s, Mantissa NOT all 0s
#define AUSSIE_FLOAT_IS_NAN(f) (AUSSIE_FLOAT_EXPONENT_BITS(f) == 255&& (AUSSIE_FLOAT_MANTISSA(f)!=0) )


#endif //AUSSIE_YFLOAT_INCLUDE_HEADER_H

