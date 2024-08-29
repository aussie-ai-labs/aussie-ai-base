// atest.h -- Unit testing -- Aussie AI Base Library  
// Created Oct 6th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YTEST_INCLUDE_HEADER_H
#define AUSSIE_YTEST_INCLUDE_HEADER_H

bool aussie_ytest_fail(char* str, char* fname, int ln);  // Failure has occurred...
bool aussie_ytest_faili(char* condstr, int ival, int iexpect, char* fname, int ln);  // Failure -- int version
bool aussie_ytest_failf(char* condstr, float fval, float fexpect, char* fname, int ln);  // Failure -- float version
bool aussie_ytest_failui(char* condstr, unsigned int ival, unsigned int iexpect, char* fname, int ln);  // Failure -- UINT version

#define ytest(cond) ( (cond) || aussie_ytest_fail(#cond, __FILE__, __LINE__) )
#define ytesti(ival, iexpect) ( (((int)ival) == ((int)iexpect)) || aussie_ytest_faili(#ival "==" #iexpect, ival, iexpect, __FILE__, __LINE__) )
#define ytestf(fval, fexpect) ( ((fval) == (fexpect)) || aussie_ytest_failf(#fval "==" #fexpect, fval, fexpect, __FILE__, __LINE__) )
#define ytestfapprox(fval, fexpect, err) ( fabs(( (fval) - (fexpect)) < err) || aussie_ytest_failf(#fval "==" #fexpect, fval, fexpect, __FILE__, __LINE__) )
#define ytestui(uival, uiexpect) ( ((unsigned)(uival) == (unsigned)(uiexpect)) || aussie_ytest_failui(#uival "==" #uiexpect, (unsigned)uival, (unsigned)uiexpect, __FILE__, __LINE__) )

void aussie_unit_tests_report();

extern int g_aussie_unit_test_failure_count;

extern bool g_aussie_testcov_enabled; 
#define AUSSIE_TESTCOV_ENABLE()  (g_aussie_testcov_enabled = true)
#define AUSSIE_TESTCOV_DISABLE()  (g_aussie_testcov_enabled = false)

#define YTESTCOV(namestr) \
	(g_aussie_testcov_enabled && \
	aussie_testcov( (namestr), __FILE__, __LINE__))

bool aussie_testcov(char* namestr, char *fname, int lnum);

#if YTESTCOV_COMPILE_OUT 
#undef YTESTCOV
#define YTESTCOV(namestr)  ((void)0)  // Remove it
#endif


void aussie_book_examples();

void aussie_test_benchmarks();

#endif //AUSSIE_YTEST_INCLUDE_HEADER_H

