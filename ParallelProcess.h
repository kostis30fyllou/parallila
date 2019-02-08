#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NXPROB      20                 /* x dimension of problem grid */
#define NYPROB      20                 /* y dimension of problem grid */
#define STEPS       100                /* number of time steps */
#define UTAG        0                  /* message tag */
#define DTAG        1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define DONE        4                  /* message tag */

typedef enum positions{UP=0, DOWN, LEFT, RIGHT} positions;


class ParallelProcess {

    int x; // my  x
    int y; // my y
    float *u; // array for grid
    MPI_Comm comm; // MPI communicator
    MPI_Datatype row, column; // row and column MPI data types
    MPI_Request send[4], recv[4]; // MPI requests for send and receive
    int threads; // number of processes
    int coords[2]; // my coords in Cartesian
    int rank; // my rank in this communicator
    int positions[4]; // contains rank of the neighbour in every position
    // private function
    int getNeighbour(int position); // get neighbour's rank

public:

    // Constructor and Destructor
    ParallelProcess(int x, int y);
    virtual ~ParallelProcess();

    // in line functions
    int getX(){ return x; };
    int getY(){ return y; };
    int getDimension(){ return dimension; };
    int getRank() { return rank; };
    void sync() { MPI_Barrier(comm); };

    // Read / write from / in file
    //void read(char* file);
    //void write(char* file);

    // Sending and Receiving functions
    void iSend();
    void iRecv();
    void waitSend();
    void waitRecv();

    // Heat functions
    void inidat();

    //int isFinished();
    int reduce(int *flag);

};