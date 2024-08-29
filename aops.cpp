//---------------------------------------------------
// yops.cpp -- Operations -- Aussie AI Base Library  
// Created Nov 17th 2023
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

#include "aops.h"  // self-include

//---------------------------------------------------


float basic_float_multiply(float a, float b)
{
	return a * b;
}
float basic_float_divide(float a, float b)
{
	return a / b;
}
float basic_float_add(float a, float b)
{
	return a + b;
}
float basic_float_equals(float a, float b)
{
	return a == b;
}
float basic_float_leq(float a, float b)
{
	return a <= b;
}
float basic_float_geq(float a, float b)
{
	return a >= b;
}

float float_convert_to_int_multiply(float a, float b)
{
	return (float)((int)a * (int)b);   // Do integer multiplication on truncated float's
}

float float_fake_add(float a, float b)
{
	// Trick to use IEEE754 format to pretend a 4-byte 32-bit float is a 4-byte 32-bit int...
	// Does an "approximate" multiplication of 2 floats...
	float c = 0.0;
	// WRONG!! ... return (float)(*(int*)&a + *(int*)&b);

	// This is correct: store the additive result into a float...
	(*(int*)&c) = (*(int*)&a + *(int*)&b);
	return c;
}

float float_approx_mogami(float a, float b)   // Mogami (2020)
{
	// Algorithm code from Mogami[2020]
	// integer addition instruction actually results in Mitchell’s approximate multiplication
	// Reportedly "~12.5% approximation of the FP32 multiplication"
	// Mogami[2020]
	int c = *(int*)&a + *(int*)&b - 0x3f800000;
	return *(float*)&c;

}

int basic_int_multiply(int a, int b)
{
	return a * b;
}

int basic_int_add(int a, int b)
{
	return a + b;
}
int basic_int_divide(int a, int b)
{
	return a / b;
}
int basic_int_mod(int a, int b)
{
	return a % b;
}
int basic_int_bitor(int a, int b)
{
	return a | b;
}

int basic_int_bitand(int a, int b)
{
	return a & b;
}
int basic_int_bitxor(int a, int b)
{
	return a ^ b;
}
int basic_int_bitshift_left(int a, int b)
{
	return a << b;
}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

