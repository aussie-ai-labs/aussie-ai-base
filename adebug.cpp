// ydebug.cpp -- Debug output functions -- Aussie AI Base Library  
// Created Oct 28th 2023
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

#include "adebug.h"  // self-include

//---------------------------------------------------

bool g_aussie_debug_enabled = false;
int g_aussie_debug_level = 0;

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------



long int g_aussie_debug_srand_seed = 0;

void aussie_debugging_test_setup()
{
	fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);

	if (g_aussie_debug_srand_seed != 0) {
		srand(g_aussie_debug_srand_seed);   // Non-random randomness!
	}
	else {  // Normal run
		srand(time((time_t)NULL));
	}

	if (g_aussie_debug_srand_seed != 0) {
		srand(g_aussie_debug_srand_seed);   // Non-random randomness!
	}
	else {  // Normal run
		long int iseed = (long)time((time_t)NULL);
		fprintf(stderr, "INFO: Random number seed: %ld 0x%lx\n", iseed, iseed);
		srand(iseed);
	}

	if (0) {
		srand(time(NULL));
	}

}
//---------------------------------------------------
//---------------------------------------------------

