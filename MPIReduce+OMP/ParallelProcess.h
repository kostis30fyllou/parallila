#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NXPROB      20                /* x dimension of problem grid */
#define NYPROB      20                 /* y dimension of problem grid */
#define STEPS       100000                /* number of time steps */
#define N       10000                   /* number of reduce check steps */
#define UTAG        0                  /* message tag */
#define DTAG        1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define CHARSPERNUM 7                  /* chars per number*/

typedef enum positions{UP=0, DOWN, LEFT, RIGHT} positions;

class ParallelProcess {

    int x; // my  x
    int y; // my y
    float *u; // array for grid
    MPI_Comm comm; // MPI communicator
    MPI_Datatype row, column; // row and column MPI data types
    MPI_Request send[2][4], recv[2][4]; // MPI requests for send and receive
    MPI_Status  status[8];
    int threads; // number of processes
    int coords[2]; // my coords in Cartesian
    int rank; // my rank in this communicator
    int positions[4]; // contains rank of the neighbour in every position
    int update(int ix, int iy, float* u1, float* u2);

public:

    // Constructor and Destructor
    ParallelProcess(int* argc, char** argv[]);
    ~ParallelProcess();

    // in line functions
    int getX(){ return x; };
    int getY(){ return y; };
    int getRank() { return rank; };
    void sync() { MPI_Barrier(comm); };

    //Write in file as string
    void write(const char* file, int iz);
    char* convertToString(int iz);

    // Sending and Receiving functions
    void Send_Init();
    void Recv_Init();
    void Start(int iz);
    void WaitRecv(int z);
    void WaitSend(int z);

    // Heat functions
    void inidat();
    int inner_update(int iz);
    int outer_update(int iz);

    //Other Functions
    void getParallelTime(double* global, double* local){ MPI_Reduce(local,global,1,MPI_DOUBLE,MPI_MAX,0,comm);};
    int reduce(int *flag);

};