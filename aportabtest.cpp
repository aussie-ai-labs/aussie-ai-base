// yportabtest.cpp -- Portability testing examples (not for porting the Aussie AI library itself) -- Aussie AI Base Library  
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

#include "aportabtest.h"  // self-include

//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

char aussie_number_to_letter_nonportable(int x)
{
	// Convert 1..26 to A..Z
	yassert(x >= 1);
	yassert(x <= 26);
	char c = x - 1 + 'A';  // Fails on EBCDIC
	return c;
}

char aussie_number_to_letter_portable(int x)
{
	yassert(x >= 1);
	yassert(x <= 26);
	char c = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x - 1];  // Works on ASCII and EBCDIC
	return c;

}

char aussie_number_to_digit_nonportable(int x)
{
	// Convert 0..9 to '0'..'9'
	yassert(x >= 0);
	yassert(x <= 9);
	char c = x + '0';  // Fails on EBCDIC
	return c;
}

char aussie_number_to_digit_portable(int x)
{
	yassert(x >= 0);
	yassert(x <= 9);
	char c = "0123456789"[x];  // Works on ASCII and EBCDIC
	return c;
}

int aussie_count_and_process_letters_non_portable()
{
	int ct = 0;
	for (int c = 'a'; c <= 'z'; c++) {
		ct++;
	}
	return ct;
}



int aussie_count_and_process_letters_portable()
{
	int ct = 0;
	for (int c = 0; c <= 25; c++) {
		char letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[c];
		ct++;
	}
	return ct;
}

void aussie_test_ebcdic_ascii_portability()
{
	ytesti(aussie_number_to_letter_nonportable(1), 'A');
	ytesti(aussie_number_to_letter_nonportable(26), 'Z');

	ytesti(aussie_number_to_letter_portable(1), 'A');
	ytesti(aussie_number_to_letter_portable(26), 'Z');

	ytesti(aussie_number_to_digit_nonportable(1), '1');
	ytesti(aussie_number_to_digit_portable(0), '0');
	ytesti(aussie_number_to_digit_portable(9), '9');

	ytesti(aussie_count_and_process_letters_non_portable(), 26);
	ytesti(aussie_count_and_process_letters_portable(), 26);

	ytest(!AUSSIE_IS_EBCDIC);
	yassert(!AUSSIE_IS_EBCDIC);
}

//---------------------------------------------------
//---------------------------------------------------
int aussie_sizeof_array(int arr[])
{
	return sizeof(arr);
}
int aussie_sizeof_array2(int arr[512])
{
	return sizeof(arr);
}
void aussie_test_pointer_array_sizes()
{
	int arr[50] = { 0 };
	int arr2[512] = { 0 };
	ytesti(aussie_sizeof_array(arr), sizeof(int*));
	ytesti(aussie_sizeof_array2(arr2), sizeof(int*));



}

//---------------------------------------------------
//---------------------------------------------------

void aussie_portability_check(bool printout)
{
	aussie_test_pointer_array_sizes();

	aussie_test_ebcdic_ascii_portability();

	// Test zero integer portability
	int i1 = 0;
	unsigned char* cptr = (unsigned char*)&i1;
	yassert(cptr[0] == 0);
	yassert(cptr[1] == 0);
	yassert(cptr[2] == 0);
	yassert(cptr[3] == 0);

	// Actually that code assumes 32-bits, here's the general code
	int i2 = 0;
	unsigned char* cptr2 = (unsigned char*)&i2;
	for (int i = 0; i < sizeof(int); i++) {
		yassert(cptr2[i] == 0);
	}


	// Test pointer NULL portability
	char* ptr1 = NULL;
	unsigned char* cptr3 = (unsigned char*)&ptr1;
	for (int i = 0; i < sizeof(char*); i++) {
		yassert(cptr3[i] == 0);
	}

	// Test float zero portability
	float f1 = 0.0f;
	unsigned char* cptr4 = (unsigned char*)&f1;
	for (int i = 0; i < sizeof(float); i++) {
		yassert(cptr4[i] == 0);
	}

	// Test basic numeric sizes
	yassert(sizeof(int) == 4);
	yassert(sizeof(float) == 4);
	yassert(sizeof(short) == 2);

	// Test pointers can be stored in LONGs
#if 0 // NOT!!!
	yassert(sizeof(char*) <= sizeof(long));
	yassert(sizeof(void*) <= sizeof(long));
	yassert(sizeof(int*) <= sizeof(long));
#endif

#if 0 // sizeof doesn't work in #if!
#if(sizeof(char*) >= sizeof(long))
#endif

#if(sizeof(void*) >= sizeof(long))
#endif

#if(sizeof(int*) >= sizeof(long))
#endif
#endif

	// Test LONGs can be stored in pointers
	yassert(sizeof(char*) >= sizeof(long));
	yassert(sizeof(void*) >= sizeof(long));
	yassert(sizeof(int*) >= sizeof(long));

	// Macro trick using macro stringize "#" operator and also the standard adjacent string concatenation features of C++
#define PRINT_TYPE_SIZE(type) \
	printf("Config: sizeof " #type " = %d bytes (%d bits)\n", \
	(int)sizeof(type), 8*(int)sizeof(type));

	if (printout) {
		PRINT_TYPE_SIZE(int);
		PRINT_TYPE_SIZE(float);
		PRINT_TYPE_SIZE(short);
	}

	if (printout) {
		printf("Config: sizeof int = %d bytes (%d bits)\n", (int)sizeof(int), 8 * (int)sizeof(int));
		printf("Config: sizeof short = %d bytes (%d bits)\n", (int)sizeof(short), 8 * (int)sizeof(short));
	}

	if (printout) {

		PRINT_TYPE_SIZE(long);
		PRINT_TYPE_SIZE(long long);
		PRINT_TYPE_SIZE(float);
		PRINT_TYPE_SIZE(double);
		PRINT_TYPE_SIZE(long double);
		PRINT_TYPE_SIZE(unsigned int);
		PRINT_TYPE_SIZE(unsigned long);
		PRINT_TYPE_SIZE(unsigned long long);
	}

	//printf("Config: sizeof long = %d bytes (%d bits)\n", (int)sizeof(long));
	//printf("Config: sizeof float = %d bytes (%d bits)\n", (int)sizeof(float));
	//printf("Config: sizeof double = %d bytes (%d bits)\n", (int)sizeof(double));
	//printf("Config: sizeof char pointer = %d bytes (%d bits)\n", (int)sizeof(char *));
	//printf("Config: sizeof void pointer = %d bytes (%d bits)\n", (int)sizeof(void*));
	//printf("Config: sizeof int pointer = %d bytes (%d bits)\n", (int)sizeof(int*));
	//printf("Config: sizeof float pointer = %d bytes (%d bits)\n", (int)sizeof(float*));

	if (printout) {

		PRINT_TYPE_SIZE(char*);
		PRINT_TYPE_SIZE(void*);
		PRINT_TYPE_SIZE(int*);
	}

	// Test zero bytes of SHORT
	short s = 0;
	memset((char*)&s, 0, sizeof(short));
	if (s != 0) {
		yassert(s == 0);  // Trigger an assert fail
		printf("Config: Warn: int zero is NOT zero bytes: %d\n", (int)s);
	}

	// Test zero bytes of INT
	int i = 0;
	memset((char*)&i, 0, sizeof(int));
	if (i != 0) {
		yassert(i == 0);  // Trigger an assert fail
		printf("Config: Warn: int zero is NOT zero bytes: %d\n", (int)i);
	}

	// Test zero bytes of INT
	long int L = 0;
	memset((char*)&L, 0, sizeof(int));
	if (L != 0) {
		yassert(L == 0);  // Trigger an assert fail
		printf("Config: Warn: int zero is NOT zero bytes: %ld\n", (long)L);
	}


	// Test zero bytes of FLOAT
	float f = 0.0f;
	memset((char*)&f, 0, sizeof(int));
	if (f != 0.0) {
		yassert(f == 0.0);  // Trigger an assert fail
		printf("Config: Warn: float zero is NOT zero bytes: %f\n", f);
	}

	// Test zero bytes of DOUBLE
	float d = 0.0;
	memset((char*)&d, 0, sizeof(int));
	if (d != 0.0) {
		yassert(d == 0.0);  // Trigger an assert fail
		printf("Config: Warn: float zero is NOT zero bytes: %f\n", d);
	}

	// Test it's ASCII not EBCDIC
	yassert('a' == 97);
	yassert('z' == 122);
	yassert('A' == 65);
	yassert('Z' == 90);

	if (printout) {
		PRINT_TYPE_SIZE(size_t);
		PRINT_TYPE_SIZE(time_t);
	}

}
//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

