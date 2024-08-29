// ywrap.cpp -- Debug wrapper functions -- Aussie AI Base Library  
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
#include "aussieai.h"
#include "aassert.h"

#include "awrap.h"  // self-include

//---------------------------------------------------

void *ymemset(void *dest, int val, int sz)  // Wrap memset
{
	if (dest == NULL) {
		yassert2(dest != NULL, "memset null dest");
		return NULL;
	}
	if (sz < 0) {
		// Why we have "int sz" not "size_t sz" above
		yassert2(sz >= 0, "memset size negative");
		return dest;  // fail
	}
	if (sz == 0) {
		yassert2(sz != 0, "memset zero size (reorder params?)");
		return dest;
	}
	if (sz <= sizeof(void*)) {
		// Suspiciously small size
		yassert2(sz > sizeof(void*), "memset with sizeof array parameter?");
		// Allow it, keep going
	}
	if (val >= 256) {
		yassert2(val < 256, "memset value not char");
		return dest; // fail
	}
	void* sret = ::memset(dest, val, sz);  // Call real one!
	return sret;
}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

