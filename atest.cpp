// ytest.cpp -- Unit testing wrapper APIs -- Aussie AI Base Library  
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
#include <math.h>

#include <iostream>

//---------------------------------------------------
//---------------------------------------------------

#include "aport.h"
#include "aussieai.h"  // Overall API

#include "aassert.h"
#include "adebug.h"
#include "aactivation.h"
#include "avector.h"
#include "anorms.h"
#include "asoftmax.h"
#include "anormalize.h"
#include "abenchmark.h"

#include "atest.h"  // self-include

//---------------------------------------------------

int g_aussie_unit_test_failure_count = 0;

#if 0 

//#include <iostream>
////#define _HAS_CXX23 1 
//#include <stacktrace> 

void aussie_print_stacktrace()
{
	std::cout << std::stacktrace::curent() << std::endl;
}
#endif 

//#define YDEBUG 1 // nothing
//#define YDEBUG // nothing
//#define YDEBUG 0
//#undef YDEBUG

#ifdef YDEBUG 
#if YDEBUG == 0
#error YDEBUG wrongly set to 0
#undef YDEBUG
#endif
#endif

//extern int __isa_available;
//bool x = __isa_available; 

//#include <intrin.h>
//__cpuid 
//__cpuidex

//---------------------------------------------
// AI C++ BOOK EXAMPLE CODE
//---------------------------------------------
#define errprintf(fmt,...)  fprintf(stderr, (fmt), __VA_ARGS__ )
#define errprintf2(fmt,...) \
	fprintf(stderr, "ERROR [%s:%d:%s]: ", __FILE__, __LINE__, __func__), \
	fprintf(stderr, (fmt), __VA_ARGS__ )

class valgrind_buffer {
private: char* m_buf;
public:
	valgrind_buffer(int n) { m_buf = ::new char[n]; }
	~valgrind_buffer() { ::delete m_buf; }
};
#if AUSSIE_COMPILE_FOR_VALGRIND
#define VALGRIND_BUFFER(buf, n) valgrind_buffer buf(n)
#else
#define VALGRIND_BUFFER(n) char buf[n];
#endif


#if !Linux
#define RUNNING_ON_VALGRIND 0
#endif

bool g_aussie_testcov_enabled = false;


bool aussie_testcov(char* namestr, char* fname, int lnum)
{
	// Test coverage: has been visited...

	return true;
}

#include <bitset>

bool aussie_test_bitset()
{
#if 0
#if sizeof(float) != sizeof(unsigned int)
#error Big blue bug
#endif
#endif
	static_assert(sizeof(float) == sizeof(unsigned int), "Big blue bug");

	std::bitset<1024> b;

	ytesti((int)b.count(), 0);

	int bit = b[0];
	ytesti(bit, 0);
	bit = b[1];
	ytesti(bit, 0);
	bit = b[1023];  // 1024 raises exception..
	ytesti(bit, 0);
	bit = b.test(0);
	ytesti(bit, 0);
	bit = b.test(1023);
	ytesti(bit, 0);

	ytesti((int)b.count(), 0);

	b.flip(0);
	bit = b.test(0);
	ytesti(bit, 1);
	b.flip(0);
	bit = b.test(0);
	ytesti(bit, 0);

	ytest(b.none());
	ytest(!b.any());
	ytest(!b.all());
	ytesti((int)b.size(), 1024);

	return true;
}

#define YDEBUG 1
#if YDEBUG
#define YDBG(token_list) do { token_list } while(0)   // Safer
#else
#define YDBG(token_list)   ((void)0)
#endif
#if YDEBUG
#define YDBG2(token_list) { token_list }    // Safer
#else
#define YDBG2(token_list)   ((void)0)
#endif

#define count_elements(table) 1

void aussie_internal_error(char* mesg)
{

}

enum selftest_areas {
	SELFTEST_NORMALIZATION,
	SELFTEST_MATMUL,
	SELFTEST_SOFTMAX,
	// ...
	SELFTEST_EOL_DUMMY
};

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

int g_aussie_sigint_arr[10]; // Array is global var

void aussie_sigint_debug_handler(int sig)
{
	// Print out debugging info
	for (int i = 0; i < 10; i++)
		printf("%d ", g_aussie_sigint_arr[i]);
	printf("\n");
	signal(SIGINT, aussie_sigint_debug_handler); // reinstall (needed for some systems)
}

void aussie_setup_signals_example()
{
	signal(SIGINT, aussie_sigint_debug_handler);
	printf("AUSSIE AI Signal Handler Setup ....\n");
	for (int i = 0; ; i = (i + 1) % 10)
		g_aussie_sigint_arr[i] = rand() % 100;
}

void aussie_test_sizeof(int arr[3])
{
#if 0 // manual enable
	printf("TEST SIZEOF ARRAY PARAM: Size of array is %d\n", (int)sizeof(arr));
#endif
}

int aussie_factorial(int n)
{
	int result = 1;
	for (int i = 1; i <= n; i++)
		result *= i;
	return result;
}

class MyString {
private:
	char* m_str;
public:
	MyString() { m_str = new char[1]; m_str[0] = '\0'; }
	MyString(char* s)
	{
		m_str = new char[strlen(s) + 1]; strcpy(m_str, s);
	}
	void operator =(const MyString& s);
	~MyString() { delete[] m_str; }
	void print() { printf("STRING: %s\n", m_str); }
};
#if 0 // Breaks
void MyString::operator = (const MyString& s)
{
	delete[] m_str; // delete old string
	m_str = new char[strlen(s.m_str) + 1]; // allocate memory
	strcpy(m_str, s.m_str); // copy new string
}
#endif

void MyString::operator = (const MyString& s)
{
	if (this != &s) {
		delete[] m_str; // delete old string
		m_str = new char[strlen(s.m_str) + 1]; // allocate memory
		strcpy(m_str, s.m_str); // copy new string
	}
}

#include <string.h>

class MyString2 {
private:
	char* m_str;
public:
	MyString2() { m_str = new char[1]; m_str[0] = '\0'; }
	MyString2(const MyString2& s); // Copy constructor
	void operator = (const MyString2& s);
	MyString2(char* s)
	{
		m_str = new char[strlen(s) + 1]; strcpy(m_str, s);
	}
	~MyString2() { delete[] m_str; }
};
void testMystring2()
{
	MyString2 s1("abc"), s2("xyz");
	s1 = s2; // Dangerous bitwise copy
}
void MyString2::operator = (const MyString2& s) // SAME AS ABOVE
{
	if (this != &s) { // Avoid aliasing problem
		delete[] m_str; // delete old string
		m_str = new char[strlen(s.m_str) + 1]; // allocate memory
		strcpy(m_str, s.m_str); // copy new string
	}
}


MyString2::MyString2(const MyString2& s) // Copy constructor
{
	m_str = new char[strlen(s.m_str) + 1]; // allocate memory
	strcpy(m_str, s.m_str); // copy string
}


void test_MyString()
{
	MyString s("abc");
	s = s;
	//s.print();
}

class Base {
	int base_data;
public:
	Base() { // Base default constructor
		base_data = 0;
	}
	Base(const Base& b) { // Base copy constructor
		base_data = b.base_data;
	}
};
class Derived : public Base {
	int derived_data;
public:
	Derived() {
		derived_data = 0;
	}
#if 0 // bad version (test manually)
	Derived(const Derived& d) {
		derived_data = d.derived_data;
	}
#endif
#if 1
	Derived(const Derived& d) : Base(d) {
		derived_data = d.derived_data;
	}
#endif
};




void testBase()
{
	Derived d1;
	Derived d2(d1);
}


void aussie_switch_example()
{


#if 0 // enable for a compilation error
	int x = 1;
	switch (x) {
		int y = 2;   // Error
	case 1: break;
	case 2: int z = 3;  // Error
	default: break;
	}
#endif
}

const float PI = 3.14f;
constexpr float PI2 = 3.14f;  // Same same

constexpr inline int twice(int x)
{
	return x + x;
}

const float SQRTPI = sqrtf(3.14f);   // Fails?
//constexpr float SQRTPI2 = sqrtf(3.14f); // Works

void aussie_book_examples()  // AI C++ book example code tests
{
	testBase();
	test_MyString();
	testMystring2();



	aussie_switch_example();

	int arr[3];
	aussie_test_sizeof(arr);

	ytesti(aussie_factorial(10), 3628800);

	//ytesti(sizeof("hello"), sizeof(char*));
	ytesti(sizeof("hello"), 6);

	// sizeof side-effect tests...
	char* ptr = NULL;
	int x = sizeof(*ptr);
	ytesti(x , 1);

	x = sizeof(*ptr++);
	ytesti(x, 1);
	yassert(ptr == NULL);

#if 0 // manual enable (loops infinitely awaiting SIGINT)
	aussie_setup_signals_example();   // SIGINT debugging test...
#endif

	// Test YDGB macros...
#if 0 // Manual enable
	YDBG(printf("DEBUG: Entered function print_list\n"); );
	YDBG(std::cerr << "DEBUG: Entered function print_list\n"; );
#endif

	int count = 1;
	YDBG2( count++; )
	YDBG2( if (count != count_elements(table)) {	)
	YDBG2( aussie_internal_error("DEBUG: Element count wrong"); )
	YDBG2( } )

	YDBG2(
		count++; 
		if (count != count_elements(table)) { // self-test
		aussie_internal_error("DEBUG: Element count wrong"); // error
	}
	)

#if !PRODUCTION // if using self-tests
#define SELFTEST // nothing
#else
#define SELFTEST if(1) {} else
#endif
	SELFTEST {
		yassert(1 == 1);
	}
	bool YDEBUG_FLAGS[100];

	extern bool g_aussie_debug_enabled;
	//extern bool YDEBUG_FLAGS[100];

#define YSELFTEST2(flagarea) \
        if(g_aussie_debug_enabled == 0 || YDEBUG_FLAGS[flagarea] == 0) \
        { /* do nothing */ } else

	g_aussie_debug_enabled = 1;
	YSELFTEST2(SELFTEST_NORMALIZATION) {
		yassert(1 == 1);
	}

	// Test assert2 examples...
	yassert2(true, "Extra");
	// yassert2(false, "Extra");
	x = 3;
	//yassert2(x == 2, x);

	aussie_test_bitset();

	// Test coverage macros....
	AUSSIE_TESTCOV_ENABLE();
	YTESTCOV("A");
	YTESTCOV("B");
	AUSSIE_TESTCOV_DISABLE();

	// Generalized assertions 
	yassertieq(10, 10);
	int a = 10;
	int b = 10;
	yassertieq(a, b);
	a = 110;
	b = 10;
	yassertiop(a, != , b);

	// BUG: sizeof a pointer type (using allocated arrays, smart pointers, smart buffers, Valgrind workaround)
#if 0
	char buf3[100] = "";
	std::cout << "DEBUG: Sizeof Buf3 = " << sizeof(buf3) << std::endl;
	char *buf4 = ::new char[100];
	std::cout << "DEBUG: Sizeof Buf4 = " << sizeof(buf4) << std::endl;
#endif

	// Valgrind limitation workaround
#define BUFSIZE 100
#if AUSSIE_COMPILE_FOR_VALGRIND
	char* buf = ::new char[BUFSIZE];
#else
	char buf[BUFSIZE];
#endif

	char hiddenbuf[BUFSIZE] = "";
	char* buf2 = hiddenbuf;
	if (RUNNING_ON_VALGRIND) {
		buf2 = ::new char[BUFSIZE];
		buf2[0] = 0; // Init!
	}

	// Debug tracing printouts
	ydebug("Hello world 1\n");
	ydebug2("Hello world 2\n");
	ydebug3("Hello world 3");

	aussie_debug_on();  // Turn on debug
	ydbg("YDBG: Hello world 1\n");
	ydbg2("YDBG: Hello world 2\n");
	ydbg3("YDBG: Hello world 3");
	aussie_debug_off();  // Turn off debug
	ydbg("YDBG: SHOULD NOT Hello world 1\n");
	ydbg2("YDBG: SHOULD NOT Hello world 2\n");
	ydbg3("YDBG: SHOULD NOT Hello world 3");

	//aussie_debug_on();  // Turn on debug
	aussie_debug_set_level(1);
#if 0 // manual enable
	ydbglevelA(1, "DBGLEVEL Hello world 1\n");
	ydbglevelB(1, "DBGLEVEL Hello world 2\n");
	ydbglevelC(1, "DBGLEVEL Hello world 3");
#endif
#if 0 // manual enable // TODO: Compile problem on g++ with __VA_ARGS__
	int dontwant = 55;
	ydbglevelA(2, "DBGLEVEL Hello world 1\n");
	ydbglevelB(3, "DBGLEVEL Hello world 2\n");
	ydbglevelC(2, "DBGLEVEL Hello world 3");

	aussie_debug_set_level(0);
	ydbglevelA(1, "DBGLEVEL SHOULD NOT Hello world 1\n");
	ydbglevelB(1, "DBGLEVEL SHOULD NOT Hello world 2\n");
	ydbglevelC(1, "DBGLEVEL SHOULD NOT Hello world 3");

	aussie_debug_off();  // Turn off debug
	ydbglevelA(1, "DBGLEVEL SHOULD NOT Hello world 1\n");
	ydbglevelB(1, "DBGLEVEL SHOULD NOT Hello world 2\n");
	ydbglevelC(1, "DBGLEVEL SHOULD NOT Hello world 3");
#endif

	int shouldnot = 4;
	x = 3;

#if 0 // manual enable
	ydbgvar(x);
#endif

	aussie_debug_set_level(1);  // Turn on...
	int y = 5;
#if 0 // manual enable
	ydbgvarlevel(1, y);
#endif

	shouldnot = 7;
	ydbgvarlevel(2, shouldnot);
	shouldnot = 77;
	ydbgvarlevel(3, shouldnot);

	aussie_debug_off();  // Turn off debug
	aussie_debug_set_level(0);  // Turn off...
	shouldnot = 8;
	ydbgvarlevel(1, shouldnot);
	shouldnot = 88;
	ydbgvarlevel(0, shouldnot);
	shouldnot = 888;
	ydbgvarlevel(2, shouldnot);



#if 0  // manually enable with "#if 1" to test them
	errprintf("Hello x = %d\n", 3);
	errprintf2("Hello x = %d\n", 3);
#endif


}

//---------------------------------------------------
//---------------------------------------------------
bool aussie_ytest_fail(char* str, char* fname, int ln)  // Failure has occurred...
{
	g_aussie_unit_test_failure_count++;
	fprintf(stderr, "AUSSIE AI UNIT TEST FAILURE(#%d): %s, %s:%d\n",
		g_aussie_unit_test_failure_count,
		str, fname, ln);
	return false;  // Always fail (need return type to be bool for use in conditions)
}

bool aussie_ytest_faili(char *condstr, int ival, int iexpect, char* fname, int ln)  // Failure -- INT version
{
	g_aussie_unit_test_failure_count++;
	fprintf(stderr, "AUSSIE AI UNIT TEST FAILURE (INT)(#%d): %s, val=%d, expect=%d, %s:%d\n",
		g_aussie_unit_test_failure_count,
		condstr, ival, iexpect, fname, ln);

	return false;  // Always fail (need return type to be bool for use in conditions)
}

bool aussie_ytest_failui(char* condstr, unsigned int ival, unsigned int iexpect, char* fname, int ln)  // Failure -- UINT version
{
	g_aussie_unit_test_failure_count++;
	fprintf(stderr, "AUSSIE AI UNIT TEST FAILURE (UINT)(#%d): %s, val=%u (%d), expect=%u (%d), %s:%d\n",
		g_aussie_unit_test_failure_count,
		condstr, ival, (signed int)ival,
		iexpect, (signed int)iexpect,
		fname, ln);

	return false;  // Always fail (need return type to be bool for use in conditions)
}

bool aussie_ytest_failf(char* condstr, float fval, float fexpect, char* fname, int ln)  // Failure -- float version
{
	g_aussie_unit_test_failure_count++;
	fprintf(stderr, "AUSSIE AI UNIT TEST FAILURE (INT)(#%d): %s, val=%f, expect=%f, %s:%d\n", 
		g_aussie_unit_test_failure_count, 
		condstr, fval, fexpect, fname, ln);

	return false;  // Always fail (need return type to be bool for use in conditions)
}


//---------------------------------------------------
//---------------------------------------------------

void aussie_unit_tests_report()
{
	if (g_aussie_unit_test_failure_count == 0
		&& g_aussie_assert_failure_count == 0) {
		fprintf(stderr, "AUSSIE AI: Unit Test Success! (zero unit test failures or assertions)\n");
	}
	else {
		fprintf(stderr, "AUSSIE AI: Unit Test failures: %d\n", g_aussie_unit_test_failure_count);
		fprintf(stderr, "AUSSIE AI: Assertion failures: %d\n", g_aussie_assert_failure_count);

	}
}

//---------------------------------------------------
//---------------------------------------------------

typedef float (*aussie_float_function_pointer_type)(float param);
typedef float (*aussie_float_vector_function_pointer_type)(float arr[], int n);

class aussie_benchmark_function {
private:
	char* m_name;
	aussie_float_function_pointer_type m_float_fnptr;
	aussie_float_vector_function_pointer_type m_float_vector_fnptr;
public:
	aussie_benchmark_function(char* name, aussie_float_function_pointer_type fnptr) {
		if (!name) { m_name = NULL; m_float_fnptr = NULL; return; }
		m_name = strdup(name);
		m_float_fnptr = fnptr;
		m_float_vector_fnptr = NULL;
	}
	aussie_benchmark_function(char* name, aussie_float_vector_function_pointer_type fnptr) {
		if (!name) { m_name = NULL; m_float_fnptr = NULL; return; }
		m_name = strdup(name);
		m_float_fnptr = NULL;
		m_float_vector_fnptr = fnptr;
	}
	aussie_float_vector_function_pointer_type float_vector_func() { return m_float_vector_fnptr;  }

	char* name() { return m_name; }
	aussie_float_function_pointer_type float_func() { return m_float_fnptr; }
};

void aussie_benchmark_float_functions(char *test_suite_name, aussie_benchmark_function funcobjarr[], unsigned long int numiter)
{
	printf("AUSSIE AI Benchmark test suite: %s\n", test_suite_name);
	float (*float_fptr)(float) = NULL;
	aussie_float_vector_function_pointer_type float_vector_fptr = NULL;

	for (int fn = 0; funcobjarr[fn].name() != NULL; fn++) {
		
		// Get the function
		float_fptr = funcobjarr[fn].float_func();  // Pointer to function
		float_vector_fptr = funcobjarr[fn].float_vector_func();


		// Test N calls to this function...
		clock_t before = clock();
		float fval = 3.5;

		// Loop N times...
		if (float_fptr) {
			for (unsigned long int i = 0; i < numiter; i++) {
				float f2 = float_fptr(fval);
			}
		}
		else if (float_vector_fptr) {
			float v[1000] = { 0.0f, 3.2f };
			for (unsigned long int i = 0; i < numiter; i++) {
				float f2 = float_vector_fptr(v,  1000);
			}
		}
		else {
			yassert_not_reached();
			continue;
		}

		// After the call ... track the time
		clock_t after = clock();
		long int clockdiff = (after - before);
		float clocksecs = (double)clockdiff / (double)CLOCKS_PER_SEC;
		char* name = funcobjarr[fn].name();
		printf("%d. %s -- %3.4f seconds (%ld ticks)\n", fn + 1, name, clocksecs, (long)clockdiff);
	}
}

//---------------------------------------------------
//---------------------------------------------------

void aussie_test_benchmarks()
{
	aussie_benchmark_vecdot();  // Vector Dot Product


	aussie_benchmark_matrix_vector_multiply();  // Matrix-Vector
	aussie_benchmark_matrix_matrix_multiplication();  // MatMUL matrix-matrix
	aussie_benchmark_normalization();  // Norm
	aussie_benchmark_softmax();  // Softmax
	aussie_benchmark_vector_exponentiation_operations();  // Expf
	aussie_benchmark_vector_scalar_operations();  // Ops

	const int MILLION = 1000000;

	// Basic vector reductions (Vector -> Scalar Float)
	aussie_benchmark_function vector_scalar_funcarr[] = {
	aussie_benchmark_function("aussie_vector_max", aussie_vector_max),
	aussie_benchmark_function("aussie_vector_min", aussie_vector_min),
	aussie_benchmark_function("aussie_vector_sum", aussie_vector_sum),
	aussie_benchmark_function("aussie_vector_sum_squares", aussie_vector_sum_squares),
	aussie_benchmark_function("aussie_vector_product", aussie_vector_product),

	
	

	aussie_benchmark_function("aussie_vector_L1_norm", aussie_vector_L1_norm),
	aussie_benchmark_function("aussie_vector_L1_norm_if_test", aussie_vector_L1_norm_if_test),
	aussie_benchmark_function("aussie_vector_L1_norm_bitwise_fabs", aussie_vector_L1_norm_bitwise_fabs),

	

	aussie_benchmark_function("aussie_vector_L2_norm", aussie_vector_L2_norm),
	aussie_benchmark_function("aussie_vector_L3_norm", aussie_vector_L3_norm),
	aussie_benchmark_function("aussie_vector_L2_squared_norm", aussie_vector_L2_squared_norm),
	

	//aussie_benchmark_function("xxx", xxx),
	//aussie_benchmark_function("xxx", xxx),
	aussie_benchmark_function(NULL, (aussie_float_function_pointer_type)NULL)
	};

	aussie_benchmark_float_functions(
		"Vector-to-Scalar Functions",  // Test suite name
		vector_scalar_funcarr,  // Function pointers
		1 * MILLION      // Number of iterations to test
	);

	// Activation GELU Function
	aussie_benchmark_function activation_funcarr[] = {
	aussie_benchmark_function("aussie_GELU_basic", aussie_GELU_basic),
	aussie_benchmark_function("aussie_GELU_basic2", aussie_GELU_basic2),
	aussie_benchmark_function("aussie_GELU_approx1", aussie_GELU_approx1),
	aussie_benchmark_function("aussie_GELU_approx1_optimized", aussie_GELU_approx1_optimized),
	aussie_benchmark_function("aussie_GELU_approx1_optimized2", aussie_GELU_approx1_optimized2),
	aussie_benchmark_function("aussie_GELU_approx2", aussie_GELU_approx2),
	aussie_benchmark_function("aussie_GELU_approx2b", aussie_GELU_approx2b),
	aussie_benchmark_function("aussie_sigmoid", aussie_sigmoid),
	aussie_benchmark_function(NULL, (aussie_float_function_pointer_type)NULL)
	};

	aussie_benchmark_float_functions(
		"Activation Functions",  // Test suite name
		activation_funcarr,  // Function pointers
		100 * MILLION      // Number of iterations to test
	);
	

	aussie_benchmark_function funcarr[] = {
		aussie_benchmark_function("sqrtf", sqrtf),
		aussie_benchmark_function("logf", logf),
		aussie_benchmark_function("expf", expf),
	aussie_benchmark_function(NULL, (aussie_float_function_pointer_type)NULL)
	};


	aussie_benchmark_float_functions(
		"Basic Math Functions",  // Test suite name
		funcarr,  // Function pointers
		100 * MILLION      // Number of iterations to test
	);
}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------


