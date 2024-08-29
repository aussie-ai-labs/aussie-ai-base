// Created Oct 6th 2023
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

#include "aussieai.h"  // Overall API

#include "aassert.h"  // self-include

int g_aussie_assert_failure_count = 0;

//---------------------------------------------------
//---------------------------------------------------
bool aussie_yassert_fail(char* str, char* fname, int ln)  // Assert failure has occurred...
{
	g_aussie_assert_failure_count++;
	fprintf(stderr, "AUSSIE AI ASSERTION FAILURE: %s, %s:%d\n", str, fname, ln);
	return false;  // Always fail (need return type to be bool for use in conditions)
}

bool aussie_yassert_fail_extra(char* condstr, char *extra, char* extrastr, char* fname, int ln)  // STRING version.
{
	g_aussie_assert_failure_count++;
	fprintf(stderr, "AUSSIE AI ASSERTION FAILURE: %s [%s], %s:%d\n", condstr, extra, fname, ln);
	return false;  // Always fail
}

//aussie_yassert_fail_varargs()

bool aussie_yassert_fail_extra(char* condstr, int extra, char* extrastr, char* fname, int ln)  // INT version.
{
	g_aussie_assert_failure_count++;
	fprintf(stderr, "AUSSIE AI ASSERTION FAILURE: %s [%s=%d], %s:%d\n", condstr, extrastr, extra, fname, ln);
	return false;  // Always fail
}


bool aussie_yassert_fail_int(char* str, int x, char *opstr, int y, char* fname, int ln)  // Assert failure has occurred...
{
	g_aussie_assert_failure_count++;
	fprintf(stderr, "AUSSIE AI INT ASSERT FAILURE: %s, %d %s %d, %s:%d\n", str, x, opstr, y, fname, ln);
	return false;  // Always fail (need return type to be bool for use in conditions)
}




//---------------------------------------------------
//---------------------------------------------------
