// awrap.h -- Debug wrapper functions -- Aussie AI Base Library  
// Created Oct 29th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YWRAP_INCLUDE_HEADER_H
#define AUSSIE_YWRAP_INCLUDE_HEADER_H

void* ymemset(void* dest, int val, int sz);  // Wrap memset

#if YDEBUG
	// Debug mode, leave wrappers..
#else // Production (remove them all)
#define ymemset memset
#endif

#endif //AUSSIE_YWRAP_INCLUDE_HEADER_H

