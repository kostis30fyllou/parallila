CC = mpicxx
CFLAGS = -O0 -lm
OBJS =  main.o

all: $(OBJS) ParallelProcess.o
	$(CC) -o mpi.out $(OBJS) ParallelProcess.o
		
main.o: main.cpp
	$(CC) -c main.cpp $(CFLAGS)
	
ParallelProcess.o: ParallelProcess.cpp
	$(CC) -c ParallelProcess.cpp $(CFLAGS)
	
	
clean:
	rm -rf *.o *.out *.dat
	
run:
	mpiexec -n $(n) ./mpi.out

	
