cc=g++

#cflags = -O0 -g3 -w -msse3 -funroll-loops -std=c++11\
	-I/home/fox/project/FoxUtil \
	-I/home/fox/Downloads/eigen-master \
	-I/home/fox/Downloads/LibN3L-2.0-master \
	-Ibasic -Imodel
	
cflags = -O3 -w -msse3 -funroll-loops  -std=c++11\
	-I/home/fox/project/FoxUtil \
	-I/home/fox/Downloads/eigen-master \
	-I/home/fox/Downloads/LibN3L-2.0-master \
	-static-libgcc -static-libstdc++ \
	-Ibasic -Imodel

libs = -lm -Wl,-rpath,./ \
 
all: clef genia

clef: clef2013.cpp NNclef.h clef2013.h NNclef1.h NNclef2.h NNclef3.h NNclef4.h globalsetting.h
	$(cc) -o clef clef2013.cpp $(cflags) $(libs)
	
genia: genia.cpp genia.h NNgenia.h NNgenia1.h NNgenia2.h NNgenia3.h globalsetting.h
	$(cc) -o genia genia.cpp $(cflags) $(libs)
	





clean:
	rm -rf *.o
	rm -rf clef
	rm -rf genia

