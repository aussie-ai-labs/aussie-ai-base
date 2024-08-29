//---------------------------------------------------
// abook1.h -- Book example 1 -- Aussie AI Base Library  
//   ... The purpose is mainly to check that they compile...
// Created 15th Dec 2023
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
#include <limits.h>

//---------------------------------------------------
//---------------------------------------------------

#include "aport.h"
#include "aussieai.h"
#include "aassert.h"
#include "atest.h"
#include "abitwise.h"

#include "abook1.h"  // self-include

//---------------------------------------------------

//---------------------------------------------------
// PREFACE
//---------------------------------------------------

#if 0 // Doesn't quite compile yet :-)
#include <numbers>
class AICPPBook {  // Try to get it to work...
private:
    char* pages[800];
    char* brainmem[1000];
    int dst;
    bool reader(float iq = 200.0f, bool ontoilet = false);
    void pick_it_up(void*, int x) { }
    int __mm256_read_words(char* pgs[]) { return 0; }
    void __mm256_store_ideas(int x, char* brainmem[]) {}
    bool posix_asleep() {  return false;  }
    void wake_up_again(int tmseconds) { }
    const int SECS_PER_HOUR = 60 * 60;
    bool write_review() { return true; }
    bool good_book() { return true; }
#define hands numbers // std::numbers
#define left pi_v  // std::numbers::pi
#define grumpy nan
#define Exception std  // std::nan
};


bool AICPPBook::reader(float iq = 200.0f, bool ontoilet = false)
{
    pick_it_up(*this, std::hands::left);
    do {
        for (int i = 0; i < n; i += 8) {
            dst = __mm256_read_words(&pages[i]);
            __mm256_store_ideas(dst, &brainmem[i]);
        }
    } while (!posix_asleep() && !std::brainfull());
    wake_up_again(time(NULL) + 8 * SECS_PER_HOUR);
    if (this->good_book()) [[likely]] return write_review();
    else [[unlikely]] throw Exception::grumpy();
}
#endif

//---------------------------------------------------
// CHAPTER 6 -- BITWISE
//---------------------------------------------------


#define aussie_popcount_basic aussie_popcount_basic2
int aussie_popcount_basic(unsigned int x) // Count number of 1's
{
    const int bitcount = 8 * sizeof(x);
    int ct = 0;
    for (int i = 0; i < bitcount; i++) {
        if (AUSSIE_ONE_BIT_SET(x, 1u << i)) ct++;
    }
    return ct;
}

#define aussie_popcount_kernighan_algorithm aussie_popcount_kernighan_algorithm2
int aussie_popcount_kernighan_algorithm(unsigned int x) // Brian Kernighan algorithm
{
    // Count number of 1's with Kernighan bit trick
    const int bitcount = 8 * sizeof(x);
    int ct = 0;
    while (x != 0) {
        x = x & (x - 1);  // Remove rightmost 1 bit: n & (n-1)
        ct++;
    }
    return ct;
}

#if !LINUX
#include <intrin.h>
#include <immintrin.h>
#endif //LINUX

#define aussie_popcount_intrinsics2 aussie_popcount_intrinsics2xxx
int aussie_popcount_intrinsics2(unsigned int x) // MSVC version
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
    return __popcnt(x);  // Microsoft intrinsics
#endif //LINUX
}

#define AUSSIE_POPCOUNT_MACRO(x) ( __popcnt((unsigned int)(x)) )

#define aussie_log2_integer_slow aussie_log2_integer_slowxx
int aussie_log2_integer_slow(unsigned int u)  // Slow float-to-int version
{
    return (int)log2f(u);
}

#define aussie_log2_integer_clz_intrinsic aussie_log2_integer_clz_intrinsicxxx
int aussie_log2_integer_clz_intrinsic(unsigned int u)  // LOG2 using CLZ
{
#if LINUX
	fprintf(stderr, "ERROR: %s: Not supported on Linux\n", __func__);
	return 0;
#else
    int clz = __lzcnt(u);  // Count leading zeros
    const int bits = 8 * sizeof(u);
    return bits - clz - 1;
#endif //LINUX
}

#define log2_integer_fast aussie_log2_integer_slow

int aussie_highest_power_of_two(int i)
{
    int bitoffset = log2_integer_fast(i);
    int highestpowerof2 = 1u << bitoffset;
    return highestpowerof2;
}

void chapter6_bitwise_tests()
{
    // Mostly just test it compiles...
    const int ten = 10;     // decimal
    const int ten1 = 0xA;    // hexadecimal
    const int ten2 = 012;    // octal
    const int ten3 = 0b1010; // binary

    // Bit Flags in Integers
#define AUSSIE_ONE_BIT_SET(x, bit)    (( ((unsigned)(x)) & ((unsigned)(bit))) != 0 )
#define AUSSIE_ANY_BITS_SET(x, bits)  (( ((unsigned)(x)) & ((unsigned)(bits))) != 0 )
#define AUSSIE_ALL_BITS_SET(x, bits)  (( ((unsigned)(x)) & ((unsigned)(bits))) == ((unsigned)(bits)) )
#define AUSSIE_NO_BITS_SET(x, bits)   (( ((unsigned)(x)) & ((unsigned)(bits))) == 0 )

#define AUSSIE_SET_BITS(x, bits)      (( ((unsigned)(x)) | ((unsigned)(bits))))
#define AUSSIE_CLEAR_BITS(x, bits)    (( ((unsigned)(x)) & (~((unsigned)(bits)))))
#define AUSSIE_TOGGLE_BITS(x, bits)   (( ((unsigned)(x)) ^ ((unsigned)(bits))))

    unsigned int u = 0, u1 = 0, u2 = 0, u3 = 0;

    if (u != 0) { /* At least one bit flag set */ }
    u3 = u2 & u1;  // Intersection of the 32-bit sets (Bitwise-AND)
    u3 = u2 | u1;  // Union of the 32-bit sets (Bitwise-OR)
    u3 = u2 ^ u1;  // Toggle bits of the 32-bit sets (Bitwise-XOR)
    u3 = ~u1;      // Set complement

#if !LINUX
    ytesti(AUSSIE_POPCOUNT_MACRO(0), 0);
#endif //LINUX
    ytesti(aussie_popcount_basic(0), 0);
#if !LINUX
    ytesti(aussie_popcount_intrinsics2(0), 0);
#endif //LINUX

    ytesti(aussie_log2_integer_slow(1), 0);
#if !LINUX
    ytesti(aussie_log2_integer_clz_intrinsic(1), 0);
#endif //LINUX
    
    ytesti(aussie_highest_power_of_two(3), 2);
    ytesti(aussie_highest_power_of_two(2), 2);
    ytesti(aussie_highest_power_of_two(4), 4);
    ytesti(aussie_highest_power_of_two(5), 4);
    ytesti(aussie_highest_power_of_two(7), 4);

#define assert yassert
    int x = INT_MAX;
    assert(x >= 0);
    ++x;  // Overflow!
    assert(x < 0);

#undef x
#define x x111
    int x = INT_MIN;
    assert(x < 0);
    --x;  // Underflow!
    assert(x > 0);

#undef x
#define x x222
    float f = (float)INT_MAX * (float)INT_MAX;  // Fine!
    int x = (float)f;  // overflow
    x = x;

#if GCC
    if (__builtin_add_overflow(x, y, &z)) {
        // Overflow!
    }
#endif

    if (INT_MAX == x) {
        // overflow!
    }
    else {
        x++;  // Safe increment
    }

    int y = 0;

    if (x > INT_MAX - y) {  // x + y > INT_MAX
        // overflow!
    }
    else {
        x += y;  // Add safely
    }

    y = 1;  // avoid divide-by-zero
    if (x > INT_MAX / y) {  // x * y > INT_MAX
        // overflow!
    }
    else {
        x *= y;  // Multiply safely
    }

#if 0
    NAND(x, y) = ~(x & y)
    NOR(x, y) = ~(x | y)
    XNOR(x, y) = ~(x ^ y)
#endif

#define NAND(x,y) ~(x & y)  // Bug alert!
#define NOR(x,y)  ~(x | y)
#define XNOR(x,y) ~(x ^ y)

#define AUSSIE_BITWISE_NAND(x,y)  ( ~ ( ((unsigned int)(x)) & ((unsigned int)(y)) ) )
#define AUSSIE_BITWISE_NOR(x,y)   ( ~ ( ((unsigned int)(x)) | ((unsigned int)(y)) ) )
#define AUSSIE_BITWISE_XNOR(x,y)  ( ~ ( ((unsigned int)(x)) ^ ((unsigned int)(y)) ) )

}

//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
// CHAPTER 7 -- FLOATING POINT
//---------------------------------------------------

#include <xmmintrin.h>
#include <pmmintrin.h>

#define aussie_float_enable_FTZ_DAZ aussie_float_enable_FTZ_DAZxxx
void aussie_float_enable_FTZ_DAZ(bool ftz, bool daz)
{
    if (ftz) {	// FTZ mode
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    }
    else {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    }

    if (daz) {  // DAZ mode
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
    else {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    }
}

#define aussie_approx_multiply_add_as_int_mogami aussie_approx_multiply_add_as_int_mogamiXXX
float aussie_approx_multiply_add_as_int_mogami(float f1, float f2)   // Add as integer
{
    int c = *(int*)&(f1)+*(int*)&(f2)-0x3f800000;  // Mogami(2020)
    return *(float*)&c;
}


#define aussie_float_bitshift_add_integer aussie_float_bitshift_add_integerXXX
float aussie_float_bitshift_add_integer(float f1, int bitstoshift)
{
    // Bitshift on 32-bit float by adding integer to exponent bits
    // FP32 = 1 sign bit, 8 exponent bits, 23 mantissa bits
    // NOTE: This can overflow into the sign bit if all 8 exponent bits are '1' (i.e. 255)
    unsigned int u = *(unsigned int*)&f1;  // Get to the bits as an integer
    if (u == 0) return f1;  // special case, don't change it...
    u += (bitstoshift << 23);  // Add shift count to the exponent bits...
    return *(float*)&u;  // Convert back to float
}

#define ilog2_exponent ilog2_exponentXXX
int ilog2_exponent(float f)  // Log2 for 32-bit float
{
    unsigned int u = *(unsigned int*)&f;
    int iexponent = ((u >> 23) & 255);  // 8-bit exponent (above 23-bit mantissa)
    iexponent -= 127;  // Remove the "offset"
    return iexponent;
}


void chapter7_floating_point_tests()
{
#define AUSSIE_FLOAT_TO_UINT(f)  (*(unsigned int*)&f)
#define AUSSIE_FLOAT_IS_POSITIVE_ZERO(f) \
        ((( AUSSIE_FLOAT_TO_UINT(f) )) == 0)  // All 0s
#define AUSSIE_FLOAT_IS_NEGATIVE_ZERO(f)  \
    (((AUSSIE_FLOAT_TO_UINT(f))) == (1u << 31))  // Sign bit only

    float f = 3.14;
    unsigned int u = (unsigned)f;  // Fail!

#define u u111
    unsigned int u = *(unsigned int*)(&f);  // Tricky!

#define u u222
    unsigned int u = *reinterpret_cast<unsigned int*>(&f);  // Fancy!

    f = *(float*)(&u);   // Floating again...
    f = *reinterpret_cast<float*> (&u);  // Better version

#define FLOAT_IS_ZERO(f) \
     ((* reinterpret_cast<unsigned int*>(&f)) == 0u)  // Bug!

    yassert(sizeof(int) == 4);
    yassert(sizeof(short int) == 2);
    yassert(sizeof(float) == 4);
    yassert(sizeof(unsigned int) == 4);

#if 0 // This is a non-compiling example
#if sizeof(float) != sizeof(unsigned int)  // Fails!
#error Big blue bug
#endif
#endif 
    static_assert(sizeof(float) == sizeof(unsigned int), "Big blue bug");

    int signbit = (u >> 31);
    int exponent = ((u >> 23) & 255);  // Fail!
    int mantissa = (u & ((1 << 23) - 1));

#define exponent exponent111
    int exponent = ((u >> 23) & 255) - 127;  // Correct!

#define AUSSIE_FLOAT_SIGN(f)      ( (*(unsigned *)&(f)) >> 31u)   // Leftmost bit
#define AUSSIE_FLOAT_EXPONENT(f)  ((int)( ((( (*(unsigned*)&(f)) )>> 23u) & 255 ) - 127 )) 
#define AUSSIE_FLOAT_MANTISSA(f)  ((*(unsigned*)&(f)) & 0x007fffffu)  // Rightmost 23 bits

#define AUSSIE_FLOAT_SIGN(f)  ( (f) < 0.0f)   // Sign test

}

//---------------------------------------------------
// CHAPTER 8 -- ARITHMETIC
//---------------------------------------------------


void chapter8_arithmetic_tests()
{
    int x = 0;
    int y = 0;

    y = x * 4;

    y = x << 2;

    y = x / 4;
    y = x >> 2u;  // faster

    int v[10] = { 0, 0, 0 };
    int i = 0;

    v[i] /= sqrtf(3.14159f);

    v[i] *= 1.0f / sqrtf(3.14159f);

    static const float scalefactor = 1.0f / sqrtf(3.14159f);
    v[i] *= scalefactor;

    y = x % 512;    // Remainder (mod)
    y = x & 511u;   // Bitwise-AND

    if (x >= 512) {

    }
    if (x & ~511u)  // Bitwise-AND of the complement (unsigned)
    {
    }

    y = x * 2;
    y = x + x;   // Addition
    y = x << 1;  // Shift

    const int N = 10;

    x = (x + 1) % N;

    if (x == N - 1)
        x = 0;
    else
        x++;

    (x == N - 1) ? (x = 0) : (x++);

    y =
        x % 256
        ;
    y =
        x & 255
        ;
    y =
        (unsigned char)x
        ;

    float f = 1.0f;
    float g = 1.0f;
    const float DIVISOR = 1.0f;

    f = g / 100.0;

    f = g * 0.01;  // Reciprocal

    f = g / DIVISOR;


    f = g * (1.0 / DIVISOR);

    x = (i * i) + (i * i);

    int temp = 0;

    temp = i * i;
    x = temp + temp;

    x = (temp = i * i) + temp; // Bug

    if (x > y && x > 10) {
        // ...
    }
    if (x > y && y > 10) {
        // ...
    }

    temp = (x > y);
    if (temp && x > 10) {
        // ...
    }
    if (temp && y > 10) {
        // ...
    }

#define scalefactor scalefactor111
    float scalefactor = sqrt(2.0) * 3.14159;

#define scalefactor scalefactor222
    float scalefactor = sqrtf(2.0f) * 3.14159f;

    int value = 0;
    int cents = 0;
    int dollars = 0;

    cents = value % 100;
    dollars = value / 100;


}

//---------------------------------------------------
// CHAPTER 9 -- COMPILE-TIME
//---------------------------------------------------

#define max max111
inline int max(int a, int b)
{
    if (a > b)
        return a;
    else
        return b;
}

#undef max
#define max(a, b) ( (a) > (b) ? (a) : (b) )

#define Complex Complex_Ch9 // dummy class
class Complex {
public:
    Complex(float x, float y) { }
};

#if 0 // compiler not supporting inline variables yet
inline int g_x = 3;
#endif

constexpr int twice(int x)
{
    return x + x;
}

#define something 1
int some_random_number() { return 0; }

int somefunc()
{
    if (something) return 27;
    else return some_random_number();
}
#if 0 // constinit not working yet
constinit int s_myconst = somefunc();
#endif

#define SymbolTable SymbolTable_chapter9
#define TABLE_SIZE 1000
class Node;

class SymbolTable {
private:
    Node* table[TABLE_SIZE]; // Hash table - array of pointers
public:
    SymbolTable(); // constructor
};
//-----------------------------------------------------------
// Constructor - initialize the hash table to empty
//-----------------------------------------------------------
SymbolTable::SymbolTable()
{
    for (int i = 0; i < TABLE_SIZE; i++) // all pointers are NULL
        table[i] = NULL;
}

#define SymbolTable SymbolTable_ch9_B

class SymbolTable { // ONE INSTANCE ONLY
private:
    static Node* table[TABLE_SIZE]; // Compile-time initialization
public:
    SymbolTable() { } // constructor does nothing
};


void chapter9_compile_time_tests()
{
    int x = 0;
    int y = 0;

    x = 3 + 4;
    x = 7;
    const float scalefactor = sqrtf(2.0f * 3.14159f);

#define scalefactor scalefactor333
    static const float scalefactor = sqrtf(2.0f * 3.14159f);

#define scalefactor scalefactor444
    static const float scalefactor = 2.506627216f; // sqrtf(2.0f * 3.14159f);


    x = 3;
    y = x;

    x = 3;
    y = 3;  // Propagated

    float f = 0.0f;
    f = sqrtf(3.14f);

    const float pi = 3.14;

#define get_config(str) 1
    const int scale_factor = get_config("scale");
    const int primes[] = { 2, 3, 5, 7, 11, 13, 17 };

    const Complex cfactor(3.14, 1.0);

    const float PI = 3.14f;
#define PI PI_111
    constexpr float PI = 3.14f;  // Same same

#if 0 // Not works :-)
    const float SQRTPI = sqrtf(3.14f);   // Works?
    constexpr float SQRTPI = sqrtf(3.14f); // Works?
#endif

    const bool cond = true;

    if constexpr (cond)
    {

    }

    static_assert(sizeof(float) == 4, "float is not 32 bits");

}


class MyVector {
    int MyVector::count() const;
};

//---------------------------------------------------
// CHAPTER 10 -- POINTER ARITHMETIC
//---------------------------------------------------

#define aussie_vecdot_basic aussie_vecdot_basic_CH10
float aussie_vecdot_basic(float v1[], float v2[], int n)   // Basic vector dot product
{
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

int aussie_vecdot_integer_fixed_point(int v1[], int v2[], int n)   // Integer vector dot product
{
    int sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}


int aussie_vecdot_integer_bitshift(int v1[], int v2[], int n)   // Integer vector dot product
{
    // Dot product-like bitshift version (to evaluate whether bitshift is any faster way to do integer vector dot product)
    // e.g. this could be used in logarithmic quantization if weights are bitshift counts...
    int sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (unsigned)v1[i] << v2[i];
    }
    return sum;
}



#define aussie_vecdot_pointer_arithmetic aussie_vecdot_pointer_arithmetic_CH10
float aussie_vecdot_pointer_arithmetic(float v1[], float v2[], int n)   // Pointer arithmetic vector dot product
{
    float sum = 0.0;
    float* endv1 = v1 + n;  // v1 start plus n*4 bytes
    for (; v1 < endv1; v1++, v2++) {
        sum += (*v1) * (*v2);
    }
    return sum;
}

void processit(MyVector v)  // Slow
{
    // ....
}

#define processit processit222
void processit(const MyVector& v)  // Reference argument
{
    // ....
}

#define processit processit333
void processit(MyVector* v)  // Pointer argument
{
    // ....
}


void chapter10_pointer_arithmetic_tests()
{
    int x = 0;

    int arr[10];
    int* ptr;

    ptr = arr;
    x = ptr[3];

    x = *(ptr + 3);  // Same as ptr[3]

#define arr arr111
#define ptr ptr111
    int arr[10];
    int* ptr1 = &arr[1];
    int* ptr2 = &arr[2];
    int diff = ptr2 - ptr1;

    int diffbytes = (char*)ptr2 - (char*)ptr1;

#define arr arr222

    int arr[100];
    int stride = &arr[2] - &arr[1];  // Wrong

#define arr arr333
#define stride stride333
    int arr[100];
    int stride = (char*)&arr[2] - (char*)&arr[1];

    int* ptr = &arr[0];

    x = ptr[3];

#define arr arr444
    char* p = "hello";
    char arr[100] = "hello";

    MyVector mv;
    MyVector* myptr = &mv;  // Pointer to mv object
    MyVector& myref = mv;   // Reference to mv object

#if 0 // These are non-compiling examples of bad code
    MyVector& v;          // Cannot do this
    MyVector& v = NULL;   // Nor this
    MyVector& v = 0;      // Nor this
#endif

#if 0 // non-compiling example of bad code
    MyVector& v = myptr;  // Disallowed
#endif
    MyVector& v = *myptr; // Works if non-null


}

//---------------------------------------------------
// CHAPTER 11 -- ALGORITHMS
//---------------------------------------------------

const int NUM_PREC = 100; // Precalculate to 100

float square_root_lazy_eval(int n)
{
    static float sqrt_table[NUM_PREC + 1]; // values
    static bool precalc[NUM_PREC + 1];     // flags

    if (!precalc[n]) { // precalculated?
        sqrt_table[n] = sqrtf((float)n); // real sqrt
        precalc[n] = true; // Mark as computed
    }
    return sqrt_table[n];
}

void generate_sqrt_table()
{
    const int NUM = 100; // Precalculate to 100
    printf("static float sqrt_table[] = {\n");
    for (int i = 0; i < NUM; i++) {
        printf("%ff", sqrtf((float)i));
        if (i + 1 < NUM)
            printf(", "); // comma after all but last
        if (i % 4 == 3 && i + 1 < NUM)
            printf("\n"); // newline every 4 numbers
    }
    printf("\n};\n"); // finish off declaration
}

float square_root_precalc(int n)
{
    const int NUM_PRECALC = 100; // Precalculate to 100
    static float sqrt_table[] = {
    0.000000f, 1.000000f, 1.414214f, 1.732051f, 2.000000f,
    2.236068f, 2.449490f, 2.645751f, 2.828427f, 3.000000f,
    3.162278f, 3.316625f, 3.464102f, 3.605551f, 3.741657f,
    3.872983f, 4.000000f, 4.123106f, 4.242640f, 4.358899f,
    4.472136f, 4.582576f, 4.690416f, 4.795832f, 4.898980f,
    5.000000f, 5.099020f, 5.196152f, 5.291502f, 5.385165f,
    5.477226f, 5.567764f, 5.656854f, 5.744563f, 5.830952f,
    5.916080f, 6.000000f, 6.082763f, 6.164414f, 6.244998f,
    6.324555f, 6.403124f, 6.480741f, 6.557438f, 6.633250f,
    6.708204f, 6.782330f, 6.855655f, 6.928203f, 7.000000f,
    7.071068f, 7.141428f, 7.211102f, 7.280110f, 7.348469f,
    7.416198f, 7.483315f, 7.549834f, 7.615773f, 7.681146f,
    7.745967f, 7.810250f, 7.874008f, 7.937254f, 8.000000f,
    8.062258f, 8.124039f, 8.185352f, 8.246211f, 8.306623f,
    8.366600f, 8.426149f, 8.485281f, 8.544003f, 8.602325f,
    8.660254f, 8.717798f, 8.774964f, 8.831760f, 8.888194f,
    8.944272f, 9.000000f, 9.055386f, 9.110434f, 9.165152f,
    9.219544f, 9.273619f, 9.327379f, 9.380832f, 9.433981f,
    9.486833f, 9.539392f, 9.591663f, 9.643651f, 9.695360f,
    9.746795f, 9.797959f, 9.848858f, 9.899495f, 9.949874f
    };
    if (n >= NUM_PRECALC) return sqrtf((float)n);
    return sqrt_table[n];
}


int factorial_precalc(int n)
{
    const int NUM_PRECALC = 5; // How many precalculated
    static int s_precalc[NUM_PRECALC + 1] = 
        { 1, 1, 2, 6, 24, 120 };

    if (n <= NUM_PRECALC)
        return s_precalc[n];
    else
        return n * factorial_precalc(n - 1);
}

bool check_point_clicked(int xm, int ym, int xp, int yp)
{
    const float DISTANCE = 2.0f;
    int xd = xp >= xm ? xp - xm : xm - xp;
    if (xd > DISTANCE)
        return false;
    int yd = yp >= ym ? yp - ym : ym - yp;
    if (yd > DISTANCE)
        return false;
    return xd * xd + yd * yd <= DISTANCE * DISTANCE;
}


void chapter11_algorithms_tests()
{
#undef x
#undef y

#undef isupper
#define isupper(ch) precomputed_array[ch]

#if 0 // manually enabled (outputs to stdout)
    generate_sqrt_table();
#endif
    ytestf(square_root_precalc(0), 0.0f);
    ytestf(square_root_lazy_eval(0), 0.0f);
    
    int key = 0;
    int a[10];
    int i = 0;
    int x = 0;

    if (key > a[i]) {
        // ...
    }
    else if (key < a[i]) {
        // ...
    }
    else { // equality
        // ...
    }

#undef x
#define expensive_fn(x) sqrtf(x)
    if (x != 0) {
        if (expensive_fn(x) != 0) {
            // ...
        }
    }
    if (x != 0 && expensive_fn(x) != 0) {
        // ...
    }

    ytesti(factorial_precalc(1), 1);
    float xMouse = 0.0f;
    float xPoint = 0.0f;
    float yMouse = 0.0f;
    float yPoint = 0.0f;

    const float DISTANCE = 2.0f;
    float diffx = xMouse - xPoint;
    float diffy = yMouse - yPoint;
    float distance = sqrtf(diffx * diffx + diffy * diffy);
    if (distance <= DISTANCE) {
        // clicked! ...
    }

    float distance_squared = diffx * diffx + diffy * diffy;  // No sqrtf
    if (distance_squared <= DISTANCE * DISTANCE) {
        // clicked! ...
    }

    struct line_segment {
        int x1, y1; // Start point
        int x2, y2; // End point
    };

    float x1 = 0.0f, y1 = 0.0f;
    float x2 = 0.0f, y2 = 0.0f;

    float flen = sqrtf((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));

#define line_segment line_segment222
    struct line_segment {
        int x1, y1; // Start point
        int x2, y2; // End point
        float length; // Length of line segment
    };
}

//---------------------------------------------------
// CHAPTER 12 -- HASHING
//---------------------------------------------------
const int SIZE = 713;

int hash(char* key)
{
    unsigned int sum = 0;
    for (; *key != '\0'; key++)
        sum += *key;
    return sum % SIZE;
}

#undef TABLE_SIZE

//-----------------------------------------------------------
// Hash Table Implementation of the Symbol Table ADT
//-----------------------------------------------------------

#include <string.h> // declare strcpy, strcmp, etc


class Node { // Node on the chained lists
private:
    static const int STR_LEN = 30; // Maximum length of string
    char symbol[STR_LEN + 1]; // symbol being stored
    Node* next; // pointer to next node in list
public:
    Node() { next = NULL; }
    friend class Dictionary; // allow easy access to nodes
};

class Dictionary{
private:
    static const int TABLE_SIZE = 211; // Hash Table Size: a prime number
    Node* table[TABLE_SIZE]; // Hash table - array of pointers
public:
    Dictionary(); // constructor
    Node* search(char* symbol);
    Node* insert(char* symbol);
    void remove(char* symbol);
    static int hash(char* symbol);
};

//------------------------------------------------------------------
// Constructor - initialize the hash table to empty
//------------------------------------------------------------------
Dictionary::Dictionary()
{
    for (int i = 0; i < TABLE_SIZE; i++) // all pointers are NULL
        table[i] = NULL;
}

//-----------------------------------------------------------------
// HASH: Generate an integer hash value for a symbol
//-----------------------------------------------------------------
int Dictionary::hash(char* symbol)
{
    unsigned int sum = 0;
    while (*symbol != '\0')
        sum += *symbol++;
    return sum % TABLE_SIZE;
}

//------------------------------------------------------------------
// SEARCH: Find a symbol in the dictionary; return pointer to it
//------------------------------------------------------------------
Node* Dictionary::search(char* symbol)
{
    int posn = hash(symbol); // Find hash value
    // Search linked list for the symbol
    for (Node *temp = table[posn]; temp != NULL; temp = temp->next) {
        if (strcmp(symbol, temp->symbol) == 0)
            return temp; // found it
    }
    return NULL; // not found
}

//--------------------------------------------------------------------
// INSERT: Enter a symbol in the hash table and return a pointer to it
//--------------------------------------------------------------------
Node* Dictionary::insert(char* symbol)
{
    Node *temp = search(symbol);
    if (temp != NULL) {
        return temp; // duplicate found; return pointer to it
    }
    else { // No duplicate found. Insert it
        int pos = hash(symbol); // get hash value
        temp = table[pos]; // get front of list
        table[pos] = new Node;
        strcpy(table[pos]->symbol, symbol); // store symbol
        table[pos]->next = temp; // link up the node
    }
    return temp; // return pointer to newly created node
}

//------------------------------------------------------------
// DELETE: delete a symbol from the symbol table
//------------------------------------------------------------
void Dictionary::remove(char* symbol)
{
    int pos;
    pos = hash(symbol);
    Node* temp = table[pos];
    Node* prev = NULL;
    for (; temp != NULL; prev = temp, temp = temp->next) {
        if (strcmp(symbol, temp->symbol) == 0)
            break; // Found it; exit for loop
    }
    if (temp == NULL) { // Not found
        return; // Ignore it
    }
    else { // Found
        if (prev == NULL) // Delete at front of list
            table[pos] = temp->next;
        else // Delete at middle/end of list
            prev->next = temp->next;
        delete temp; // Return deleted node to heap
    }
}

void chapter12_hashing_tests()
{
    Dictionary d;

    d.insert("a");
    ytest(d.search("a") != NULL);
    ytest(d.search("b") == NULL);
    d.remove("a");
    ytest(d.search("a") == NULL);
    ytest(d.search("b") == NULL);
}

//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

#include <stdio.h>
#include <time.h>

void profile_shifts()
{
    const int MILLION = 1000000;
    const int ITERATIONS = 100 * MILLION;

    int x = 1, y = 2, z = 3;

    clock_t before = clock();
    for (int i = 0; i < ITERATIONS; i++)
        x = y << z;
    printf("Profiling: %d Shifts took %f seconds\n", ITERATIONS,
        (double)(clock() - before) / CLOCKS_PER_SEC);

    before = clock();
    for (int i = 0; i < ITERATIONS; i++)
        x = y * z;
    printf("Profiling: %d Multiplications took %f seconds\n", ITERATIONS,
        (double)(clock() - before) / CLOCKS_PER_SEC);
}


void profile_shifts2()
{
    const int MILLION = 1000000;
    const int ITERATIONS = 100 * MILLION;

    volatile int x = 0; /* volatile to prevent optimizations */
    clock_t before = clock();
    for (int i = 0; i < ITERATIONS; i++)
        x = x << 1;
    printf("Profiling: %d Shifts took %f seconds\n", ITERATIONS,
        (double)(clock() - before) / CLOCKS_PER_SEC);
    before = clock();
    for (int i = 0; i < ITERATIONS; i++)
        x = x * 2;
    printf("Profiling: %d Multiplications took %f seconds\n", ITERATIONS,
        (double)(clock() - before) / CLOCKS_PER_SEC);
}

void profile_shifts3()
{
    const int MILLION = 1000000;
    const int ITERATIONS = 100 * MILLION;

    volatile int x = 0; /* volatile to prevent optimizations */
    clock_t before = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        x = x << 1; x = x << 1; x = x << 1; x = x << 1;
        x = x << 1; x = x << 1; x = x << 1; x = x << 1;
        x = x << 1; x = x << 1; x = x << 1; x = x << 1;
        x = x << 1; x = x << 1; x = x << 1; x = x << 1;
        x = x << 1; x = x << 1; x = x << 1; x = x << 1;
    }
    printf("Profiling: %d Shifts took %f seconds\n", ITERATIONS * 20,
        (double)(clock() - before) / CLOCKS_PER_SEC);
    before = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        x = x * 2; x = x * 2; x = x * 2; x = x * 2;
        x = x * 2; x = x * 2; x = x * 2; x = x * 2;
        x = x * 2; x = x * 2; x = x * 2; x = x * 2;
        x = x * 2; x = x * 2; x = x * 2; x = x * 2;
        x = x * 2; x = x * 2; x = x * 2; x = x * 2;
    }
    printf("Profiling: %d Multiplications took %f seconds\n", ITERATIONS * 20,
        (double)(clock() - before) / CLOCKS_PER_SEC);
}

void profile_shifts4()
{
    const int MILLION = 1000000;
    const int ITERATIONS = 1000 * MILLION;
    volatile int x = 0; // volatile to prevent optimizations
    double time1, time2;

    // Time the loop overhead
    clock_t before = clock();
    for (int i = 0; i < ITERATIONS; i++)
        x = 1;
    clock_t loop_cost = clock() - before; // overhead
    double ovtime = (double)(loop_cost) / CLOCKS_PER_SEC;
    printf("Profiling: %d loop overhead: %f seconds\n", ITERATIONS, ovtime);

    // Shifts
    before = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        x = x << 1;
    }
    time1 = (double)(clock() - before - loop_cost) / CLOCKS_PER_SEC;
    printf("Profiling: %d Shifts took %f seconds\n", ITERATIONS, time1);

    // Multiplications
    before = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        x = x * 2;
    }
    time2 = (double)(clock() - before - loop_cost) / CLOCKS_PER_SEC;
    printf("Profiling: %d Multiplications took %f seconds\n", ITERATIONS, time2);

    // Compare both times, and print percentage difference
    const float ACCURACY = 0.00001f; // maximum error
    if (fabs(time1 - time2) < ACCURACY) // (almost) equal?
        printf("Profiling: Shift and multiplications: same time\n");
    else if (time1 < time2) {
            printf("Profiling: Shifts faster by %5.2f percent\n",
                (time2 - time1) / time2 * 100.0);
    } 
    else {
        printf("Profiling: Multiplications faster by %5.2f percent\n",
            (time1 - time2) / time1 * 100.0);
    }
}


void chapter38_tuning_tests() // Tuning, profiling & benchmarking
{
    profile_shifts4();
    profile_shifts3();  // Unrolled
    profile_shifts2();
    profile_shifts();

}

//---------------------------------------------------
//---------------------------------------------------
// APPENDIX: SLUG CATALOG
//---------------------------------------------------
//---------------------------------------------------

const int N = 10; // Number of elements in vector/matrix
class Vector {
    double data[N];
public:
    double get_element(int i) const { return data[i]; }
    void set_element(int i, double value) { data[i] = value; }
};

class Matrix {
    double data[N][N];
public:
    double get_element(int i, int j) const { return data[i][i]; }
};

Vector operator * (const Matrix& m, const Vector& v)
{
    Vector temp;
    // multiply matrix by vector
    for (int i = 0; i < N; i++) { // for each row
        double sum = 0.0; // sum of N multiplications
        for (int j = 0; j < N; j++) {
            sum += m.get_element(i, j) * v.get_element(j);
        }
        temp.set_element(i, sum); // store new vector element
    }
    return temp; // return new vector
}

//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------

// More efficient version with friend function:
#define Matrix Matrix111
#define Vector Vector111
#define N N111

const int N = 10; // Number of elements in vector/matrix
class Matrix;
class Vector {
    double data[N];
public:
    friend Vector operator * (const Matrix& m, const Vector& v);
};

class Matrix {
    double data[N][N];
public:
    friend Vector operator * (const Matrix& m, const Vector& v);
};

Vector operator * (const Matrix& m, const Vector& v)
{
    Vector temp;
    // multiply matrix by vector
    for (int i = 0; i < N; i++) { // for each row
        double sum = 0.0; // sum of N multiplications
        for (int j = 0; j < N; j++) {
            sum += m.data[i][j] * v.data[j]; // access data directly
        }
        temp.data[i] = sum; // store new vector element
    }
    return temp; // return new vector
}

void slug_vector_friends_test()
{
#undef Vector
#undef Matrix
    Vector v1;
    Matrix m1;
    Vector v2 = m1 * v1;
#define Matrix Matrix111
#define Vector Vector111
    Vector v1b;
    Matrix m1b;
    Vector v2b = m1b * v1b;


}

//---------------------------------------------------
//---------------------------------------------------

void appendix_slug_catalog_tests()
{
    slug_vector_friends_test();
}

//---------------------------------------------------
//---------------------------------------------------
//---------------------------------------------------


void aussie_book_examples_unit_tests()
{
    fprintf(stderr, "INFO: %s: Running unit tests\n", __func__);
    appendix_slug_catalog_tests();

    chapter38_tuning_tests(); // Tuning, profiling & benchmarking

    chapter6_bitwise_tests();
    chapter7_floating_point_tests();
    chapter8_arithmetic_tests();
    chapter9_compile_time_tests();
    chapter10_pointer_arithmetic_tests();
    chapter11_algorithms_tests();
    chapter12_hashing_tests();

}

//---------------------------------------------------
//---------------------------------------------------

//---------------------------------------------------
//---------------------------------------------------


//---------------------------------------------------
//---------------------------------------------------

