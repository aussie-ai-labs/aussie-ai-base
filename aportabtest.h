// aportabtest.h -- Portability testing examples (not for porting the Aussie AI library itself) -- Aussie AI Base library
// Created Nov 12th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YPORTABTEST_INCLUDE_HEADER_H
#define AUSSIE_YPORTABTEST_INCLUDE_HEADER_H

void aussie_portability_check(bool printout);
void aussie_test_ebcdic_ascii_portability();

char aussie_number_to_letter_nonportable(int x);
char aussie_number_to_letter_portable(int x);
char aussie_number_to_digit_nonportable(int x);
char aussie_number_to_digit_portable(int x);
int aussie_count_and_process_letters_non_portable();
int aussie_count_and_process_letters_portable();
int aussie_sizeof_array(int arr[]);
int aussie_sizeof_array2(int arr[512]);
void aussie_test_pointer_array_sizes();



#if ('A' != 65)
#define AUSSIE_IS_EBCDIC 1
#else
#define AUSSIE_IS_EBCDIC 0
#endif

#endif //AUSSIE_YPORTABTEST_INCLUDE_HEADER_H

