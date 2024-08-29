#--------------------------------------------------
# MAKEFILE for AUSSIE AI BASE library
#--------------------------------------------------
# - Source code for book "Generative AI in C++", David Spuler, March 2024.
#
#--------------------------------------------------
# On some Linux platforms, run this command to enable g++:
# scl enable devtoolset-8 -- bash
##--------------------------------------------------
#Profiling flags for gprof
PFLAGS=-pg

##--------------------------------------------------
##CFLAGS=-I../../RMLib_Project/RMLib_Source/ -fpermissive -Wall -Wno-write-strings -Wno-address -Wno-parentheses $(PFLAGS)
CFLAGS=-fpermissive -Wall -Wno-write-strings -Wno-address -Wno-parentheses $(PFLAGS)
##LINKFLAGS=-L../../RMLib_Project/RMLib_Source/ -L/usr/lib64/ -g $(PFLAGS)
LINKFLAGS=-L/usr/lib64/ -g $(PFLAGS)

OBJS= aactivation.o aassert.o aprecompute.o atest.o adebug.o abenchmark.o \
aavx.o abitwise.o abook1.o adynarray.o afloat.o amatmul.o anormalize.o \
anorms.o aops.o aportabtest.o asoftmax.o atopk.o  \
avector.o awrap.o

# UNUSED:
# aussieaitest.o 

TESTER=aussieaitest.cpp

#--------------------------------------------------
#--------------------------------------------------

.cpp.o:
	-@echo Compiling Source File $<
	g++ $(CFLAGS) $(CCFLAGS) $(PFLAGS) -g -c $<


all: ALL
ALL: $(OBJS) LIBRARY EXE

EXE:
	-@echo Making BINARY TEST FILE
	g++ $(PFLAGS) $(CFLAGS) $(CCFLAGS) $(LINKFLAGS) $(OBJS) $(TESTER) -o aussieai

LIBRARY: aussieai.a

clean:
	/bin/rm $(OBJS) aussieai.a

aussieai.a: $(OBJS)
	-@echo Making LIBRARY
	ar ruvs aussieai.a $(OBJS) 
	ranlib aussieai.a


test:
	./aussieai

#--------------------------------------------------
val: valgrind
valgrind:
	valgrind ./aussieai


#--------------------------------------------------
# gprof requires "-pg" flag in compilation
# Executing the app then creates data file "gmon.out"
# gprof then analyzes "gmon.out" along with the exe file.
#--------------------------------------------------

prof:
	./aussieai
	gprof ./ycodecpp > prof1.prof
	more prof1.prof

