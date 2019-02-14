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
    if(size * size != threads) {
        perror("Process number must have square root integer");
        MPI_Abort(MPI_COMM_WORLD, 0);
        exit(1);
    }
    dims[0] = size;
    dims[1] = size;

    // Cartesian create with reorder on
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);
    if(NXPROB%size != 0 || NYPROB%size != 0){
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
    x = NXPROB/size + 2;
    y = NYPROB/size + 2;
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
 *  Write function and convert to string
 */
void ParallelProcess::write(const char* file, int iz) {
    MPI_File fp;
    MPI_Datatype filetype, strtype, blocktype;
    MPI_Status status;
    char* str = convertToString(iz);
    if(MPI_File_open(comm, file, MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &fp) > 0) {
        perror("Could not open this file");
        MPI_Abort(comm, 0);
    }
    // initialize MPI string type
    MPI_Type_contiguous(CHARSPERNUM, MPI_CHAR, &strtype);
    MPI_Type_commit(&strtype);
    int sizes[2] = {NXPROB, NYPROB};
    int local[2] = {x-2, y-2};
    int starts[2] = {coords[0]*local[0], coords[1]*local[1]};
    MPI_Type_create_subarray(2, sizes, local, starts, MPI_ORDER_C, strtype, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fp, 0, strtype, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, str, (x - 2) * (y - 2), strtype, &status);
    MPI_File_close(&fp);
    MPI_Type_free(&filetype);
    MPI_Type_free(&strtype);
    delete str;
}

char* ParallelProcess::convertToString(int iz) {
    const char* fmt = "%6.1f ";
    const char* endfmt = "%6.1f\n";
    char *str = new char[(x - 2) * (y - 2) * CHARSPERNUM];
    int count = 0;
    for (int ix = 1; ix <= x - 2; ix++) {
        for (int iy = 1; iy < y - 2; iy++) {
            sprintf(&str[count * CHARSPERNUM], fmt, u[iz*x*y + ix*y + iy]);
            count++;
        }
        // if we are on the right side of file change line
        if(coords[1] == sqrt(threads) - 1)
            sprintf(&str[count * CHARSPERNUM], endfmt, u[iz*x*y + ix*y + y-2]);
        else sprintf(&str[count * CHARSPERNUM], fmt, u[iz*x*y + ix*y + y-2]);
        count++;
    }
    return str;
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

void ParallelProcess::update(int ix, int iy, float* u1, float* u2) {
    u2[ix*y+iy] = u1[ix*y+iy]  +
                    parms.cx * (u1[(ix+1)*y+iy] +
                                u1[(ix-1)*y+iy] -
                                2.0 * u1[ix*y+iy]) +
                    parms.cy * (u1[ix*y+iy+1] +
                                u1[ix*y+iy-1] -
                                2.0 * u1[ix*y+iy]);
}

void ParallelProcess::inner_update(int iz) {
    for (int ix = 2; ix < x-2; ix++) {
        for (int iy = 2; iy < y-2; iy++){
            update(ix, iy, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
        }
    }
}

void ParallelProcess::outer_update(int iz) {
    for(int iy = 1; iy < y-1; iy++) {
        // UP
        update(1, iy, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
        // DOWN
        update(x-2, iy, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
    }

    for(int ix = 1; ix < x-1; ix++) {
        // LEFT
        update(ix, 1, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
        // RIGHT
        update(ix, y-2, &(u[iz*x*y]), &(u[(1-iz)*x*y]));
    }
}

