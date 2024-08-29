// aussieai.h -- Aussie AI Base Library header 
// Created 27th July 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_INCLUDE_HEADER_H
#define AUSSIE_INCLUDE_HEADER_H

#ifndef AUSSIEAI
#define AUSSIEAI 1
#endif

extern long int g_aussie_multiplications_count;   // Count multiplications


#if __GNUC__ || __GCC__
#define LINUX 1
#else
#define LINUX 0
#endif

#endif //AUSSIE_INCLUDE_HEADER_H

