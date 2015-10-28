# Builds all the projects in the solution...
.PHONY: all_projects
all_projects: USAC RunSingleTest 

# Builds project 'USAC'...
.PHONY: USAC
USAC: 
	make --directory="." --file=USAC.makefile

# Builds project 'RunSingleTest'...
.PHONY: RunSingleTest
RunSingleTest: 
	make --directory="./msvc/RunSingleTest/" --file=RunSingleTest.makefile

# Cleans all projects...
.PHONY: clean
clean:
	make --directory="." --file=USAC.makefile clean
	make --directory="./msvc/RunSingleTest/" --file=RunSingleTest.makefile clean

