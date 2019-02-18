#include "ParallelProcess.h"

/**
 *
 * Constructor and Destructor
 */

struct Parms {
    float cx;
    float cy;
} parms = {0.1, 0.1};

ParallelProcess::ParallelProcess(int* argc, char** argv[]) {
    int periods[2];
    int dims[2]; //  2D matrix/grid
    int size;
    int reorder = 1;
    periods[0] = 0; // no row periodic
    periods[1] = 0; // no column periodic
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &threads);
    size = sqrt(threads);
    dims[0] = size;
    dims[1] = size;
    // Calculate dimensions of the grid
    while(dims[0]*dims[1] != threads) {
        if(dims[0]*dims[1] < threads) {
            dims[0]++;
        }
        else dims[1]--;
    }

    // Cartesian create with reorder on
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);
    if(NXPROB%dims[0] != 0 || NYPROB%dims[1] != 0){
        perror("Grid  cannot be divided with this number of processes");
        MPI_Abort(comm, 0);
        exit(1);
    }
    // my rank
    MPI_Comm_rank(comm, &rank);
    // my coords
    MPI_Cart_coords(comm, rank, 2, coords);
    // find neighbours positions
    MPI_Cart_shift(comm , 0 , 1, &positions[UP], &positions[DOWN]);
    MPI_Cart_shift(comm , 1 , 1, &positions[LEFT], &positions[RIGHT]);
    // initialize x and y of each block
    x = NXPROB/dims[0] + 2;
    y = NYPROB/dims[1] + 2;
    // initialize array of grid
    u = new float[2*x*y];
    for (int ix = 0; ix < x; ix++) {
        for (int iy = 0; iy < y; iy++) {
            u[ix * y + iy] = 0.0;
            u[x * y + ix * y + iy] = 0.0;
        }
    }
    // initiate my MPI Data types
    MPI_Type_contiguous(y-2, MPI_FLOAT, &row);
    MPI_Type_commit(&row);
    MPI_Type_vector(x-2, 1, y, MPI_FLOAT, &column);
    MPI_Type_commit(&column);
}

ParallelProcess::~ParallelProcess() {
    //delete arrays
    delete u;
    // free types
    MPI_Type_free(&row);
    MPI_Type_free(&column);
    MPI_Finalize();
}

/**
 * Send, Receive, Start and Wait
 */
void ParallelProcess::Send_Init() {
    for(int iz=0; iz < 2; iz++) {
        MPI_Send_init(&u[iz*x*y + y + 1], 1, row, positions[UP], DTAG, comm, &send[iz][UP]);
        MPI_Send_init(&u[iz*x*y + (x-2)*y + 1], 1, row, positions[DOWN], UTAG, comm, &send[iz][DOWN]);
        MPI_Send_init(&u[iz*x*y + y + 1], 1, column, positions[LEFT], RTAG, comm, &send[iz][LEFT]);
        MPI_Send_init(&u[iz*x*y + 2*y - 2], 1, column, positions[RIGHT], LTAG, comm, &send[iz][RIGHT]);
    }
}

void ParallelProcess::Recv_Init() {
    for(int iz=0; iz < 2; iz++) {
        MPI_Recv_init(&u[iz*x*y + 1], 1, row, positions[UP], UTAG, comm, &recv[iz][UP]);
        MPI_Recv_init(&u[iz*x*y + (x-1)*y + 1], 1, row, positions[DOWN], DTAG, comm, &recv[iz][DOWN]);
        MPI_Recv_init(&u[iz*x*y + y], 1, column, positions[LEFT], LTAG, comm, &recv[iz][LEFT]);
        MPI_Recv_init(&u[iz*x*y + 2*y - 1], 1, column, positions[RIGHT], RTAG, comm, &recv[iz][RIGHT]);
    }
}

void ParallelProcess::Start(int iz) {
    for(int i = 0; i < 4; i++) {
        MPI_Start(&send[iz][i]);
        MPI_Start(&recv[iz][i]);
    }
}

void ParallelProcess::WaitRecv(int iz) {
    MPI_Waitall(4, recv[iz], MPI_STATUSES_IGNORE);
}

void ParallelProcess::WaitSend(int iz) {
    MPI_Waitall(4, send[iz], MPI_STATUSES_IGNORE);
}

/**
 *  Write function
 */
void ParallelProcess::write(const char* file, int iz) {
    MPI_File fp;
    MPI_Datatype filetype, blocktype;
    MPI_Status status;
    if(MPI_File_open(comm, file, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &fp) > 0) {
        perror("Could not open this file");
        MPI_Abort(comm, 0);
    }
    int sizes[2] = {NXPROB, NYPROB};
    int local[2] = {x-2, y-2};
    int starts[2] = {coords[0]*local[0], coords[1]*local[1]};
    MPI_Type_create_subarray(2, sizes, local, starts, MPI_ORDER_C, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fp, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, &u[iz*x*y + y + 1], (x - 2) * (y - 2), MPI_FLOAT, &status);
    MPI_File_close(&fp);
    MPI_Type_free(&filetype);
}


/**
 *
 * Heat functions
 */
void ParallelProcess::inidat() {
    for (int ix = 1; ix <= x - 2; ix++)
        for (int iy = 1; iy <= y - 2; iy++)
            u[ix*y + iy] = (float)((ix-1) * ((x-2) - (ix-1) - 1) * (iy-1) * ((y-2) - (iy-1) - 1)) + 10;
}

int ParallelProcess::update(int ix, int iy, float* u1, float* u2) {
    u2[ix*y+iy] = u1[ix*y+iy]  +
                    parms.cx * (u1[(ix+1)*y+iy] +
                                u1[(ix-1)*y+iy] -
                                2.0 * u1[ix*y+iy]) +
                    parms.cy * (u1[ix*y+iy+1] +
                                u1[ix*y+iy-1] -
                                2.0 * u1[ix*y+iy]);
    if(u2[ix*y+iy] == u1[ix*y+iy])
        return 0;
    else return 1;
}

int ParallelProcess::inner_update(int iz) {
    int changes = 0;
#pragma omp parallel for reduction(+ : changes) schedule(static, CHUNK)
    for (int ix = 2; ix < x - 2; ix++) {
        for (int iy = 2; iy < y - 2; iy++) {
            changes += update(ix, iy, &(u[iz * x * y]), &(u[(1 - iz) * x * y]));
        }
    }
    return changes;
}

int ParallelProcess::outer_update(int iz) {
    int changes = 0;
#pragma omp parallel for reduction(+ : changes) schedule(static, CHUNK)
    for(int iy = 1; iy < y-1; iy++) {
        // UP
        changes += update(1, iy, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
        // DOWN
        changes += update(x-2, iy, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
    }

#pragma omp parallel for reduction(+ : changes) schedule(static, CHUNK)
    for(int ix = 1; ix < x-1; ix++) {
        // LEFT
        changes += update(ix, 1, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
        // RIGHT
        changes += update(ix, y-2, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
    }
    return changes;
}

int ParallelProcess::reduce(int *flag) {
    int ret;
    MPI_Allreduce(flag, &ret, 1, MPI_INT, MPI_SUM, comm);
    return ret;
}

