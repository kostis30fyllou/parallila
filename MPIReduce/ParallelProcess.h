#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NXPROB      160                 /* x dimension of problem grid */
#define NYPROB      128                 /* y dimension of problem grid */
#define STEPS       500                /* number of time steps */
#define N           50                 /* number of reduce check steps */
#define UTAG        0                  /* message tag */
#define DTAG        1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */

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

    // inline functions
    int getX(){ return x; };
    int getY(){ return y; };
    int getRank() { return rank; };
    void sync() { MPI_Barrier(comm); };

    //Write in file
    void write(const char* file, int iz);

    // Sending and Receiving functions
    inline void Send_Init();
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

/**
 * Send, Receive, Start and Wait
 */
inline void ParallelProcess::Send_Init() {
    for(int iz=0; iz < 2; iz++) {
        MPI_Send_init(&u[iz*x*y + y + 1], 1, row, positions[UP], DTAG, comm, &send[iz][UP]);
        MPI_Send_init(&u[iz*x*y + (x-2)*y + 1], 1, row, positions[DOWN], UTAG, comm, &send[iz][DOWN]);
        MPI_Send_init(&u[iz*x*y + y + 1], 1, column, positions[LEFT], RTAG, comm, &send[iz][LEFT]);
        MPI_Send_init(&u[iz*x*y + 2*y - 2], 1, column, positions[RIGHT], LTAG, comm, &send[iz][RIGHT]);
    }
}

inline void ParallelProcess::Recv_Init() {
    for(int iz=0; iz < 2; iz++) {
        MPI_Recv_init(&u[iz*x*y + 1], 1, row, positions[UP], UTAG, comm, &recv[iz][UP]);
        MPI_Recv_init(&u[iz*x*y + (x-1)*y + 1], 1, row, positions[DOWN], DTAG, comm, &recv[iz][DOWN]);
        MPI_Recv_init(&u[iz*x*y + y], 1, column, positions[LEFT], LTAG, comm, &recv[iz][LEFT]);
        MPI_Recv_init(&u[iz*x*y + 2*y - 1], 1, column, positions[RIGHT], RTAG, comm, &recv[iz][RIGHT]);
    }
}

inline void ParallelProcess::Start(int iz) {
    for(int i = 0; i < 4; i++) {
        MPI_Start(&send[iz][i]);
        MPI_Start(&recv[iz][i]);
    }
}

inline void ParallelProcess::WaitRecv(int iz) {
    MPI_Waitall(4, recv[iz], MPI_STATUSES_IGNORE);
}

inline void ParallelProcess::WaitSend(int iz) {
    MPI_Waitall(4, send[iz], MPI_STATUSES_IGNORE);
}

inline int ParallelProcess::update(int ix, int iy, float* u1, float* u2) {
    u2[ix*y+iy] = u1[ix*y+iy]  +
                  0.1 * (u1[(ix+1)*y+iy] +
                              u1[(ix-1)*y+iy] -
                              2.0 * u1[ix*y+iy]) +
                  0.1 * (u1[ix*y+iy+1] +
                              u1[ix*y+iy-1] -
                              2.0 * u1[ix*y+iy]);
    if(u2[ix*y+iy] == u1[ix*y+iy])
        return 0;
    else return 1;
}

inline int ParallelProcess::inner_update(int iz) {
    int changes = 0;
    for (int ix = 2; ix < x - 2; ix++) {
        for (int iy = 2; iy < y - 2; iy++) {
            changes += update(ix, iy, &(u[iz * x * y]), &(u[(1 - iz) * x * y]));
        }
    }
    return changes;
}

inline int ParallelProcess::outer_update(int iz) {
    int changes = 0;
    for(int iy = 1; iy < y-1; iy++) {
        // UP
        changes += update(1, iy, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
        // DOWN
        changes += update(x-2, iy, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
    }

    for(int ix = 1; ix < x-1; ix++) {
        // LEFT
        changes += update(ix, 1, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
        // RIGHT
        changes += update(ix, y-2, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
    }
    return changes;
}

inline int ParallelProcess::reduce(int *flag) {
    int ret;
    MPI_Allreduce(flag, &ret, 1, MPI_INT, MPI_SUM, comm);
    return ret;
}