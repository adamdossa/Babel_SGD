TARGET = alglib_test

CFLAGS = -O3 -I ~/proj/alglib/src -Wall

LFLAGS =

CC = g++

SRC = test_i.cpp
SRC += ap.cpp
SRC += alglibinternal.cpp
SRC += alglibmisc.cpp
SRC += dataanalysis.cpp
SRC += diffequations.cpp
SRC += fasttransforms.cpp
SRC += integration.cpp
SRC += interpolation.cpp
SRC += linalg.cpp
SRC += optimization.cpp
SRC += solvers.cpp
SRC += specialfunctions.cpp
SRC += statistics.cpp

TMP = $(SRC:.cpp=.o)
OBJ = $(TMP:.c=.o)

gcc main_pca.out src/ap.cpp src/alglibinternal.cpp src/alglibmisc.cpp src/dataanalysis -o main_pca.out
g++ -O3 main_pca.cpp src/ap.cpp src/alglibinternal.cpp src/alglibmisc.cpp src/dataanalysis.cpp src/optimization.cpp src/linalg.cpp src/statistics.cpp src/specialfunctions.cpp src/solvers.cpp -o main_pca.out
$(TARGET): $(OBJ)

@$(CC) $(LFLAGS) $(OBJ) -o $@
@echo Build Complete

.cpp.o: $<
@echo Compiling $<
@$(CC) -c $(CFLAGS) $< -o $@

clean:
@rm -f $(OBJ) $(TARGET)
@echo All object files and binaries removed