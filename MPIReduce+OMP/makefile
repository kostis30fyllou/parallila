CC = mpicxx
CFLAGS = -O3 -lm -fopenmp
OBJS =  main.o ParallelProcess.o

all: $(OBJS)
	$(CC) -o mpi.out $(OBJS)

main.o: main.cpp
	$(CC) -c main.cpp $(CFLAGS)

ParallelProcess.o: ParallelProcess.cpp
	$(CC) -c ParallelProcess.cpp $(CFLAGS)


clean:
	rm -rf *.o *.out *.dat

	
