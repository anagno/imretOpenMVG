# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../../src" -I"../../ext/libconfig-1.4.5/lib" -I"/usr/lib"
Release_Include_Path=-I"../../src" -I"../../ext/libconfig-1.4.5/lib" -I"/usr/lib"

# Library paths...
Debug_Library_Path=-L"../../gccDebug" -L"../../ext/clapack-3.2.1" -L"/usr/lib/x86_64-linux-gnu/"
Release_Library_Path=-L"../../gccRelease" -L"../../ext/clapack-3.2.1" -L"/usr/lib/x86_64-linux-gnu" 

# Additional libraries...
Debug_Libraries=-Wl,--start-group -lUSAC  -lblas -llapack -lf2c  -Wl,--end-group
Release_Libraries=-Wl,--start-group -lUSAC  -lblas -llapack -lf2c  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D GCC_BUILD -D _DEBUG -D _CONSOLE 
Release_Preprocessor_Definitions=-D GCC_BUILD -D NDEBUG -D _CONSOLE -D _SECURE_SCL=0 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 -g 
Release_Compiler_Flags=-O2 

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders gccDebug/RunSingleTest.o 
	g++ gccDebug/RunSingleTest.o  $(Debug_Library_Path) $(Debug_Libraries) -Wl,-rpath,./ -o ../../gccDebug/RunSingleTest -l:libconfig++.so.9

# Compiles file RunSingleTest.cpp for the Debug configuration...
-include gccDebug/RunSingleTest.d
gccDebug/RunSingleTest.o: RunSingleTest.cpp

# Compiles file RunSingleTest.cpp for the Release configuration...
-include gccRelease/RunSingleTest.d
gccRelease/RunSingleTest.o: RunSingleTest.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c RunSingleTest.cpp $(Release_Include_Path) -o gccRelease/RunSingleTest.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM RunSingleTest.cpp $(Release_Include_Path) > gccRelease/RunSingleTest.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p gccDebug/source
	mkdir -p ../../gccDebug
	mkdir -p gccRelease/source
	mkdir -p ../../gccRelease

# Cleans intermediate and output files (objects, libraries, exe
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c RunSingleTest.cpp $(Debug_Include_Path) -o gccDebug/RunSingleTest.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM RunSingleTest.cpp $(Debug_Include_Path) > gccDebug/RunSingleTest.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders gccRelease/RunSingleTest.o 
	g++ gccRelease/RunSingleTest.o  $(Release_Library_Path) $(Release_Libraries) -Wl,-rpath,./ -o ../USAC/gccRelease/RunSingleTest -l:libconfig++.so.9

# Compiles file RunSingleTest.cpp for the Release configuration...
-include gccRelease/RunSingleTest.d
gccRelease/RunSingleTest.o: RunSingleTest.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c RunSingleTest.cpp $(Release_Include_Path) -o gccRelease/RunSingleTest.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM RunSingleTest.cpp $(Release_Include_Path) > gccRelease/RunSingleTest.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p gccDebug/source
	mkdir -p ../../gccDebug
	mkdir -p gccRelease/source
	mkdir -p ../../gccRelease

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f gccDebug/*.o
	rm -f gccDebug/*.d
	rm -f ../USAC/gccDebug/*.a
	rm -f ../USAC/gccDebug/*.so
	rm -f ../USAC/gccDebug/*.dll
	rm -f ../USAC/gccDebug/*.exe
	rm -f gccRelease/*.o
	rm -f gccRelease/*.d
	rm -f ../USAC/gccRelease/*.a
	rm -f ../USAC/gccRelease/*.so
	rm -f ../USAC/gccRelease/*.dll
	rm -f ../USAC/gccRelease/*.exe








