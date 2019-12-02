/* Steven Smiley | COMP233 | Jacobi Iterations
*  Based on work by Argonne National Laboratory.
*  https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html
*/

#include <stdio.h>
#include <math.h>
#include "mpi.h"

/* This example handles a 12 x 12 mesh, on 4 processors only. */
#define MESHSIZE 12
#define NUMPROC 4
#define MAXITR 100

int main( argc, argv )
int argc;
char **argv;
{
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
    double     diffNorm, gGiffNorm;
    double     xLocal[(MESHSIZE/NUMPROC)+2][MESHSIZE];
    double     xNew[(MESHSIZE/NUMPROC)+2][MESHSIZE];

    MPI_Init( &argc, &argv );

    //Initialize MPI communicator rank and size variables.
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &commSize );

    if (commSize != 4) MPI_Abort( MPI_COMM_WORLD, 1 );

    /* xlocal[][0] is lower ghostpoints, xlocal[][maxn+2] is upper */

    /* Note that top and bottom processes have one less row of interior
       points */
    rFirst = 1;
    rLast  = MESHSIZE/commSize;
    if (rank == 0)        rFirst++;
    if (rank == commSize - 1) rLast--;

    /* Fill the data as specified */
    for (r=1; r<=MESHSIZE/commSize; r++) {
        xLocal[r][MESHSIZE-1] = 75; //set value for east boundary
        xLocal[r][0] = 10;          //set value for west boundary
    }

    for (c=0; c<MESHSIZE; c++) {
	    xLocal[rFirst-1][c] = 100;   //set value for north boundary
	    xLocal[rLast+1][c] = 100;    //set value for south boundary
    }

    itrCount = 0;
    do {

	/* Send up unless I'm at the top, then receive from below */
	/* Note the use of xlocal[i] for &xlocal[i][0] */
	if (rank < commSize - 1) 
	    MPI_Send( xLocal[MESHSIZE/commSize], MESHSIZE, MPI_DOUBLE, rank + 1, 0, 
		      MPI_COMM_WORLD );
	if (rank > 0)
	    MPI_Recv( xLocal[0], MESHSIZE, MPI_DOUBLE, rank - 1, 0, 
		      MPI_COMM_WORLD, &status );

	/* Send down unless I'm at the bottom */
	if (rank > 0) 
	    MPI_Send( xLocal[1], MESHSIZE, MPI_DOUBLE, rank - 1, 1, 
		      MPI_COMM_WORLD );
	if (rank < commSize - 1) 
	    MPI_Recv( xLocal[MESHSIZE/commSize+1], MESHSIZE, MPI_DOUBLE, rank + 1, 1, 
		      MPI_COMM_WORLD, &status );
	

	/* Compute new values (but not on boundary) */
	itrCount ++;
	diffNorm = 0.0;
	for (r=rFirst; r<=rLast; r++) 
	    for (c=1; c<MESHSIZE-1; c++) {
		xNew[r][c] = (xLocal[r][c+1] + xLocal[r][c-1] +
			      xLocal[r+1][c] + xLocal[r-1][c]) / 4.0;
		diffNorm += (xNew[r][c] - xLocal[r][c]) * 
		            (xNew[r][c] - xLocal[r][c]);
	    }


	/* Only transfer the interior points */
	for (r=rFirst; r<=rLast; r++) 
	    for (c=1; c<MESHSIZE-1; c++) 
		xLocal[r][c] = xNew[r][c];

	MPI_Allreduce( &diffNorm, &gGiffNorm, 1, MPI_DOUBLE, MPI_SUM,
		       MPI_COMM_WORLD );
	gGiffNorm = sqrt( gGiffNorm );
	if (rank == 0) printf( "At iteration %d, diff is %e\n", itrCount, 
			       gGiffNorm );
    } while (gGiffNorm > 1.0e-2 && itrCount < MAXITR);

    MPI_Finalize( );
    return 0;
}

void printMesh(double** meshArray, int meshSize){
    int r, c;   //loop control variables

    for(r = 0; r < meshSize; r++){
        for(c = 0; c < meshSize; c++){
            //print cell
        }
        //print newline
    }
}