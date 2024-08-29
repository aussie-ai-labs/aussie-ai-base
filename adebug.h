// ydebug.h -- Debug output functions -- Aussie AI Base Library  
// Created Oct 28th 2023
// Copyright (c) 2023 Aussie AI Labs Pty Ltd

#ifndef AUSSIE_YDEBUG_INCLUDE_HEADER_H
#define AUSSIE_YDEBUG_INCLUDE_HEADER_H

#ifndef AUSSIE_DEBUG_REMOVE  // Not in cmd-line args
#define AUSSIE_DEBUG_REMOVE 1  // Removes them all if set to 1
#endif

#define ydebug0(str)  ( fprintf(stderr, "%s", (str)) )
#define ydebug0b(str)  ( std::cerr << str << std::endl )

#define ydebug(fmt,...)  fprintf(stderr, (fmt), __VA_ARGS__ )

// Version that adds file/line/function context
#define ydebug2(fmt,...)  \
	(fprintf(stderr, "DEBUG [%s:%d:%s]: ", \
		__FILE__, __LINE__, __func__ ), \
	fprintf(stderr, (fmt), __VA_ARGS__ ))

// Version that makes the newline optional
#define ydebug3(fmt,...)  \
	(fprintf(stderr, "DEBUG [%s:%d:%s]: ", \
		__FILE__, __LINE__, __func__ ), \
	fprintf(stderr, (fmt), __VA_ARGS__ ), \
	(strchr((fmt), '\n') != NULL \
		|| fprintf(stderr, "\n")))

// Versions with a dynamic flag to turn on/off...
extern bool g_aussie_debug_enabled;

#define aussie_debug_off()  ( g_aussie_debug_enabled = false )
#define aussie_debug_on()  ( g_aussie_debug_enabled = true )

#define ydbg(fmt,...)  ( g_aussie_debug_enabled && \
	fprintf(stderr, (fmt), __VA_ARGS__ ))

#define ydbg2(fmt,...)  \
	( g_aussie_debug_enabled && \
	( fprintf(stderr, "DEBUG [%s:%d:%s]: ", \
		__FILE__, __LINE__, __func__ ), \
	fprintf(stderr, (fmt), __VA_ARGS__ )))

// Version that makes the newline optional
#define ydbg3(fmt,...)  \
	( g_aussie_debug_enabled && \
	(fprintf(stderr, "DEBUG [%s:%d:%s]: ", \
		__FILE__, __LINE__, __func__ ), \
	fprintf(stderr, (fmt), __VA_ARGS__ ), \
	(strchr((fmt), '\n') != NULL \
		|| fprintf(stderr, "\n"))))

// Version with a debug level...

extern bool g_aussie_debug_enabled;
extern int g_aussie_debug_level;

#undef aussie_debug_off
#define aussie_debug_off()  ( \
	g_aussie_debug_enabled = false, \
	g_aussie_debug_level = 0)

#undef aussie_debug_on
#define aussie_debug_on()  ( \
	g_aussie_debug_enabled = true, \
	g_aussie_debug_level = 1 )

#define aussie_debug_set_level(lvl)  ( \
	g_aussie_debug_enabled = (((lvl) != 0)), \
	g_aussie_debug_level = (lvl) )

#define ydbglevel1(fmt) (ydebuglevel(1, (fmt)))
#define ydbglevel2(fmt) (ydebuglevel(2, (fmt)))

#define ydbglevelA(lvl,fmt,...)  ( \
	g_aussie_debug_enabled && \
	(lvl) <= g_aussie_debug_level && \
	fprintf(stderr, (fmt), __VA_ARGS__ ))

#define ydbglevelB(lvl,fmt,...)  \
	( g_aussie_debug_enabled && \
	(lvl) <= g_aussie_debug_level && \
	( fprintf(stderr, "DEBUG [%s:%d:%s]: ", \
		__FILE__, __LINE__, __func__ ), \
	fprintf(stderr, (fmt), __VA_ARGS__ )))

// Version that makes the newline optional
#define ydbglevelC(lvl,fmt,...)  \
	( g_aussie_debug_enabled && \
	(lvl) <= g_aussie_debug_level && \
	(fprintf(stderr, "DEBUG [%s:%d:%s]: ", \
		__FILE__, __LINE__, __func__ ), \
	fprintf(stderr, (fmt), __VA_ARGS__ ), \
	(strchr((fmt), '\n') != NULL \
		|| fprintf(stderr, "\n"))))

#define ydbgvar(v) \
  std::cerr << "DEBUG: " << #v \
	<< " = " \
	<< (v) \
	<< std::endl

#define ydbgvarlevel(lvl, v) \
	( g_aussie_debug_enabled && \
	(lvl) <= g_aussie_debug_level && \
  ( std::cerr << "DEBUG: " << #v \
	<< " = " \
	<< (v) \
	<< std::endl ))


// Full removal of all debug code at compile-time...
#if AUSSIE_DEBUG_REMOVE 
#undef ydebug
#undef ydebug2
#undef ydebug3

#define ydebug(...)  /*nothing*/
#define ydebug2(...)  /*nothing*/
#define ydebug3(...)  /*nothing*/

#undef ydbg
#undef ydbg2
#undef ydbg3

#define ydbg(...)  /*nothing*/
#define ydbg2(...)  /*nothing*/
#define ydbg3(...)  /*nothing*/

#endif


extern long int g_aussie_debug_srand_seed; // = 0;

void aussie_debugging_test_setup();

#endif //AUSSIE_YDEBUG_INCLUDE_HEADER_H

