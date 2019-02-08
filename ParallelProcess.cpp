#include "ParallelProcess.h"

ParallelProcess::ParallelProcess() {
    int periods[2];
    int dims[2]; //  2D matrix/grid
    int size;
    int reorder = 1;
    int error;
    periods[0] = 0; // no row periodic
    periods[1] = 0; // no column periodic
    MPI_Comm_size(MPI_COMM_WORLD, &threads);
    size = sqrt(threads);
    dims[0] = size;
    dims[1] = size;
    // Cartesian create with reorder on
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);
    // my rank
    MPI_Comm_rank(comm, &rank);
    // my coords
    MPI_Cart_coords(comm, myrank, 2, coords);
    // find neighbours positions
    for(int i = 0; i < 4; i++) {
        positions[i] = getNeighbour(i);
    }
    if(NXPROB*NYPROB % threads != 0 ){
        MPI_Abort(newcomm, error);
        exit(1);
    }
    // initialize x and y of each block
    x = NXPROB/size + 2;
    y = NYPROB/size + 2;
    // initialize array of grid
    u = new float[2*x*y];
    for(int iz=0;iz<2;iz++){
        for(int ix=0;ix<x;ix++){
            for(int iy=0;iy<y;iy++){
                u[iz*x*y+ix*y+iy] = 0.0;
            }
        }
    }
    // initiate my MPI Data types
    MPI_Type_contiguous(x-2, MPI_FLOAT, &row);
    MPI_Type_commit(&row);
    MPI_Type_vector(y-2, sizeof(float), x, MPI_FLOAT, &column);
    MPI_Type_commit(&column);
}

ParallelProcess::~ParallelProcess() {
    // free types
    MPI_Type_free(&row);
    MPI_Type_free(&column);
}

void ParallelProcess::inidat() {
    for (int ix = 0; ix < x - 2; ix++)
        for (int iy = 0; iy < y - 2; iy++)
            u[ix*y + iy + y + 1] = (float)(ix * (x - ix - 1) * iy * (y - iy - 1));
}

int ParallelProcess::getNeighbour(int position) {
    int rank;
    int tempCoords[2];
    switch(position) {
        case UP:
            tempCoords[0] = coords[0] - 1;
            tempCoords[1] = coords[1];
            break;

        case DOWN:
            tempCoords[0] = coords[0] + 1;
            tempCoords[1] = coords[1];
            break;

        case LEFT:
            tempCoords[0] = coords[0];
            tempCoords[1] = coords[1] - 1;
            break;

        case RIGHT:
            tempCoords[0] = coords[0];
            tempCoords[1] = coords[1] + 1;
            break;

        default:
            perror("getNeighbour");
            MPI_Abort(comm, -1);
    }
    if(MPI_Cart_rank(comm, tempCoords, &rank)) {
        rank = MPI_PROC_NULL;
    }
    return rank;
}

