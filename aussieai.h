// aport.h -- Aussie AI Base Library -- PORTING HEADER FILE
// Created 27th July 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef YPORT_INCLUDE_HEADER_H
#define YPORT_INCLUDE_HEADER_H

#ifndef AUSSIEAI
#define AUSSIEAI 1
#endif

// NOTE: It doesn't yet work if set either of these to 1 (Dave 11/Aug/2023)
#define AUSSIE_TRY_LOG_ADDITION 0  // Logarithmic addition of weights (fails with error)
#define AUSSIE_TRY_MOGAMI_APPROX_ADDITION 0  // Multiplication via approximate addition like Mitchell in Mogami(2020) // (fails with error)
#define AUSSIE_EARLY_EXIT_N_LAYERS  9  // 0 doesn't early exit, e.g. 5 exits after 5 layers

#if _MSC_VER  // fix problems with "restrict" keyword on MSVC++
#define restrict /*nothing*/
#endif

extern long int g_aussie_multiplications_count;   // Count multiplications


#endif //YPORT_INCLUDE_HEADER_H

