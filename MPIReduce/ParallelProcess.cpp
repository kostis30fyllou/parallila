#include "ParallelProcess.h"

/**
 *
 * Constructor and Destructor
 */


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
            dims[1]++;
        }
        else dims[0]--;
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


