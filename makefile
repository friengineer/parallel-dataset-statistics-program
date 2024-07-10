EXE = main
CC = mpicc
CCFLAGS = -Wall -lm -std=c99

all:
	$(CC) $(CCFLAGS) -o $(EXE) main.c
