// aassert.h -- Assertions -- Aussie AI Base Library  
// Created Oct 6th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YASSERT_INCLUDE_HEADER_H
#define AUSSIE_YASSERT_INCLUDE_HEADER_H

bool aussie_yassert_fail(char* str, char* fname, int ln);  // Assertion failure has occurred...
bool aussie_yassert_fail_extra(char* condstr, char* extra, char *extrastr, char* fname, int ln);  // STRING extra...
bool aussie_yassert_fail_extra(char* condstr, int extra, char* extrastr, char* fname, int ln);  // INT extra.

#define yassert(cond) ( (cond) || aussie_yassert_fail(#cond, __FILE__, __LINE__) )

#define yassert2(cond, extra) ( (cond) || aussie_yassert_fail_extra(#cond,extra, #extra, __FILE__, __LINE__) )

#define yassertvarargs(cond,...)    ( (cond) || aussie_yassert_fail_varargs(#cond,__FILE__, __LINE__, __VA_ARGS__) )

#define yassert_not_reached()   ( yassert(false) )
#define yassert_not_reached2() \
  ( aussie_yassert_fail("Unreachable code was reached", __FILE__, __LINE__) )

#define yassert_nonnull(var) \
	yassert( (var) != NULL) 

#define yassert_nonzero(var) \
	yassert( (var) != 0) 

#define yassert_and_return(cond,retval) \
	if (cond) {} else { \
	    aussie_yassert_fail(#cond " == NULL", __FILE__, __LINE__); \
	    return (retval); \
	}

#define yassert_param_tolerate_null(var,retval) \
	if ((var) != NULL) {} else { \
	    aussie_yassert_fail(#var " == NULL", __FILE__, __LINE__); \
	    return (retval); \
	}

#define yassert_param_tolerate_null2(var,retval) \
	do { if ((var) != NULL) {} else { \
	    aussie_yassert_fail(#var " == NULL", __FILE__, __LINE__); \
	    return (retval); \
	}} while(0)

bool aussie_yassert_fail_int(char* str, int x, char* opstr, int y, char* fname, int ln); // Assert failure has occurred...

#define yassertieq(x,y) \
	(( (x) == (y)) || \
	 aussie_yassert_fail_int(#x "==" #y, \
		 (x), "==", (y), \
		__FILE__, __LINE__))


#define yassertiop(x, op, y) \
	(( (x) op (y)) || \
	 aussie_yassert_fail_int(#x #op #y, \
		 (x), #op, (y), \
		__FILE__, __LINE__))


extern int g_aussie_assert_failure_count; // = 0;

#define yassert_once()  do { \
        static int s_count = 0; \
		++s_count; \
		if (s_count > 1) { \
	        aussie_yassert_fail("Code executed twice", \
				__FILE__, __LINE__); \
		} \
   } while(0)

#define yassert_N_times(ntimes)  do { \
        static int s_count = 0; \
		++s_count; \
		if (s_count > (ntimes)) { \
	        aussie_yassert_fail( \
               "Code executed more than " #ntimes " times", \
				__FILE__, __LINE__); \
		} \
   } while(0)

#endif //AUSSIE_YASSERT_INCLUDE_HEADER_H

