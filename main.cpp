#include "ParallelProcess.h"

int main(int argc, char* argv[]) {
    double tstart, tend, local, global=0;
    int iz = 0;
    ParallelProcess *process;
    process = new ParallelProcess(&argc, &argv);
    process->inidat();
    process->write("initial.dat", iz);
    process->sync();
    tstart = MPI_Wtime();
    process->Send_Init();
    process->Recv_Init();
    for(int i = 0; i < STEPS; i++) {
        process->Start(iz);
        process->WaitAll(iz);
        iz = 1 - iz;
    }
    tend = MPI_Wtime();
    local = tend -tstart;
    MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,0,comm);
    process->sync();
    if(process->getRank() == 0) {
        printf("Total time elapsed = %lf \n",global);
    }
    delete process;
    return 0;
}

