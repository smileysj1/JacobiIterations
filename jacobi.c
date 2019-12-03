/* Steven Smiley | COMP233 | Jacobi Iterations
*  Based on work by Argonne National Laboratory.
*  https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MESHSIZE 800
#define NUMPROC 4
#define CHUNKROWS (MESHSIZE/NUMPROC)
#define CHUNKSIZE (CHUNKROWS * MESHSIZE)

void printMesh(float meshArray[][]);

int main( argc, argv )
int argc;
char **argv;
{
    const float northBound = 100.0;
    const float southBound = 100.0;
    const float eastBound = 75.0;
    const float westBound = 10.0;
    const float interiorInit = (northBound + southBound + eastBound + westBound) / 4.0;

    /*
    rank: MPI rank of the current process
    value:
    commSize: Number of processes in the MPI communicator
    errorCount:
    totalError:
    r: row iteration variable
    c: column iteration varible
    itrCount: total number of iterations performed
    rFirst: initial boundary for r
    rLast: end boundary for r
    */
    int        rank, value, commSize, errorCount, totalError, r, c, itrCount;
    int        rFirst, rLast;
    MPI_Status status;
    float     diffNorm, gDiffNorm;
    float     xLocal[CHUNKROWS + 2][MESHSIZE];
    float     xNew[CHUNKROWS + 2][MESHSIZE];

    float xFull[MESHSIZE][MESHSIZE];

    float epsilon;
    int maxIterations;

    if(argc < 3){   //check that we have enough command line arguments
        printf("Please specify the correct number of arguments.\n");
        printf("Usage: jacobi [epsilon] [max_iterations]\n");
        return 0;
    }

    MPI_Init( &argc, &argv );

    //Initialize MPI communicator rank and size variables.
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &commSize );

    //assign our epsilon and maxIteration variables using our command line arguments
    epsilon = strtod(argv[1], NULL);
    maxIterations = strtol(argv[2], NULL, 10);

    /* xlocal[][0] is lower ghostpoints, xlocal[][maxn+2] is upper */

    /* Note that top and bottom processes have one less row of interior
       points */
    rFirst = 1;
    rLast  = CHUNKROWS;
    if (rank == 0)        rFirst++;
    if (rank == commSize - 1) rLast--;

    /* Fill the data as specified */
    for (r = 1; r <= CHUNKROWS; r++) {
        for(c = 0; c < MESHSIZE; c++){
            xLocal[r][c] = interiorInit;
        }

        xLocal[r][MESHSIZE-1] = eastBound; //set value for east boundary
        xLocal[r][0] = westBound;          //set value for west boundary
    }
    for (c=0; c<MESHSIZE; c++) {
	    xLocal[rFirst-1][c] = northBound;   //set value for north boundary
	    xLocal[rLast+1][c] = southBound;    //set value for south boundary
    }


    itrCount = 0;
    do {
	/* Send up unless I'm at the top, then receive from below */
	/* Note the use of xlocal[i] for &xlocal[i][0] */
	if (rank < commSize - 1) 
	    MPI_Send( xLocal[MESHSIZE/commSize], MESHSIZE, MPI_FLOAT, rank + 1, 0, 
		      MPI_COMM_WORLD );
	if (rank > 0)
	    MPI_Recv( xLocal[0], MESHSIZE, MPI_FLOAT, rank - 1, 0, 
		      MPI_COMM_WORLD, &status );

	/* Send down unless I'm at the bottom */
	if (rank > 0) 
	    MPI_Send( xLocal[1], MESHSIZE, MPI_FLOAT, rank - 1, 1, 
		      MPI_COMM_WORLD );
	if (rank < commSize - 1) 
	    MPI_Recv( xLocal[MESHSIZE/commSize+1], MESHSIZE, MPI_FLOAT, rank + 1, 1, 
		      MPI_COMM_WORLD, &status );
	

	/* Compute new values (but not on boundary) */
	itrCount ++;
	diffNorm = 0.0;
	for (r=rFirst; r<=rLast; r++) 
	    for (c=1; c<MESHSIZE-1; c++) {
		xNew[r][c] = (xLocal[r][c+1] + xLocal[r][c-1] +     //new value computed as the average of its 4 neighbors
			      xLocal[r+1][c] + xLocal[r-1][c]) / 4.0;
		diffNorm += (xNew[r][c] - xLocal[r][c]) *           //compute diffNorm sum
		            (xNew[r][c] - xLocal[r][c]);
	    }


	/* Only transfer the interior points */
	for (r=rFirst; r<=rLast; r++) 
	    for (c=1; c<MESHSIZE-1; c++) 
		xLocal[r][c] = xNew[r][c];

	MPI_Allreduce( &diffNorm, &gDiffNorm, 1, MPI_FLOAT, MPI_SUM,
		       MPI_COMM_WORLD );
	gDiffNorm = sqrt( gDiffNorm );
	if (rank == 0) printf( "At iteration %d, diff is %e\n", itrCount, 
			       gDiffNorm );
    } while (gDiffNorm > epsilon && itrCount < maxIterations);  //keep doing Jacobi iterations until we hit our iteration or precision limit

    
    //send chunks to master
    MPI_Send(xLocal[1], (MESHSIZE / commSize) * MESHSIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    
    if(rank == 0){  //assemble chunks on master into full mesh
        int proc;
        for(proc = 0; proc < commSize; proc++){
            //recieve each array chunk from other processes
            MPI_Recv(xFull[CHUNKROWS * proc], CHUNKSIZE, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &status); 
        }

        printMesh(xFull);
    }

    MPI_Finalize( );
    return 0;
}

void printMesh(float meshArray[MESHSIZE][MESHSIZE]){
    int r, c;   //loop control variables

    for(r = 0; r < MESHSIZE; r++){
        for(c = 0; c < MESHSIZE; c++){
            //print cell with width 4 and 1 digit after the decimal
            printf("%4.1f ", meshArray[r][c]);
        }
        //print newline
        printf("\n");
    }
}