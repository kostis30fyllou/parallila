#include "ParallelProcess.h"

int main(int argc, char* argv[]) {
    double tstart, tend, local, global=0;
    int iz = 0;
    int flag = 1;
    ParallelProcess *process;
    process = new ParallelProcess(&argc, &argv);
    process->inidat();
    //process->write("initial.dat", iz);
    process->sync();
    tstart = MPI_Wtime();
    process->Send_Init();
    process->Recv_Init();
#pragma opm parallel
    for(int i = 0; i < STEPS; i++) {
        int changes = 0;
        process->Start(iz);
        if(flag == 1)
            changes += process->inner_update(iz);
        process->WaitRecv(iz);
        if(flag == 1)
            changes += process->outer_update(iz);
        if(changes == 0)
            flag = 0;
        iz = 1 - iz;
        if(i%N == 0 && process->reduce(&flag) == 0) {
            i = STEPS;
        }
        process->WaitSend(1-iz);
    }
    tend = MPI_Wtime();
    //process->write("final.dat", iz);
    local = tend - tstart;
    process->getParallelTime(&global, &local);
    if(process->getRank() == 0) {
        printf("Total time elapsed = %lf \n",global);
    }
    delete process;
    return 0;
}

