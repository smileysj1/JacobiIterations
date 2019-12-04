/* Steven Smiley | COMP233 | Jacobi Iterations
*  Based on work by Argonne National Laboratory.
*  https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MESHSIZE 16
#define NUMPROC 4
#define CHUNKROWS (MESHSIZE/NUMPROC)
#define CHUNKSIZE (CHUNKROWS * MESHSIZE)

void writeToPPM(float** mesh, int iterations);
void printMesh(float** meshArray);
float** makeContiguous2DArray(int r, int c);
void freeContiguous2DArray(float** ary);

int main( argc, argv )
int argc;
char **argv;
{
    
    const float NORTH_BOUND = 100.0;
    const float SOUTH_BOUND = 100.0;
    const float EAST_BOUND = 75.0;
    const float WEST_BOUND = 10.0;
    const float INTERIOR_AVG = (NORTH_BOUND + SOUTH_BOUND + EAST_BOUND + WEST_BOUND) / 4.0;


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
    float**     xLocal;
    float**     xNew;
    float**     xFull;

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

    //allocate memory to 2d arrays
    xLocal = makeContiguous2DArray(CHUNKROWS + 2, MESHSIZE);
    xNew = makeContiguous2DArray(CHUNKROWS + 2, MESHSIZE);
    xFull = makeContiguous2DArray(MESHSIZE, MESHSIZE);

    printf("succesfully allocated array's on proc %d\n", rank);

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
            xLocal[r][c] = INTERIOR_AVG;
        }

        xLocal[r][MESHSIZE-1] = EAST_BOUND; //set value for east boundary
        xLocal[r][0] = WEST_BOUND;          //set value for west boundary
    }
    for (c=0; c<MESHSIZE; c++) {
	    xLocal[rFirst-1][c] = NORTH_BOUND;   //set value for north boundary
	    xLocal[rLast+1][c] = SOUTH_BOUND;    //set value for south boundary
    }
    printf("Successfully initialized array data on proc %d\n", rank);


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

    //swap array pointers
    float** tmp = xLocal;
    xLocal = xNew;
    xNew = tmp;

	MPI_Allreduce( &diffNorm, &gDiffNorm, 1, MPI_FLOAT, MPI_SUM,
		       MPI_COMM_WORLD );
	gDiffNorm = sqrt( gDiffNorm );
	if (rank == 0) printf( "At iteration %d, diff is %e\n", itrCount, 
			       gDiffNorm );
    } while (gDiffNorm > epsilon && itrCount < maxIterations);  //keep doing Jacobi iterations until we hit our iteration or precision limit

    
    if(rank == 0){  //assemble chunks on master into full mesh
        memcpy(xFull[0], xLocal[1], CHUNKSIZE * sizeof(float));

        int proc;
        for(proc = 1; proc < commSize; proc++){
            //recieve each array chunk from other processes
            MPI_Recv(xFull[CHUNKROWS * proc], CHUNKSIZE, MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &status); 
        }

        printMesh(xFull);
        writeToPPM(xFull, itrCount);
    }
    else{
            //send chunks to master
            MPI_Send(xLocal[1], CHUNKSIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            printf("sent chunk from %d\n", rank);
    }
    
    freeContiguous2DArray(xLocal);
    freeContiguous2DArray(xNew);
    freeContiguous2DArray(xFull);

    MPI_Finalize( );
    return 0;
}

//Write our PPM image from a 2d array
void writeToPPM(float** mesh, int iterations) {
	//make a file stream for our ppm file
	FILE* fp = fopen("jacobi.ppm", "w");

	//write the data out to the file
		//output header information to ppm file
        fprintf(fp, "P3 %d %d 255\n", MESHSIZE, MESHSIZE);
        fprintf(fp, "#Steven Smiley | COMP233 | OpenMP Mandelbrot\n");
        fprintf(fp, "#This image took %d iterations to converge.\n", iterations);

        int r, c;
		for (r = 0; r < MESHSIZE; r++) {
			for (c = 0; c < MESHSIZE; c++) {
                int val = (int)(mesh[r][c] * (255.0/100.0));

				//write our color values to the buffer
				fprintf(fp, "%d 0 0 ", val);
				if (c % 5 == 4) {
					//write a newline every 5th rgb value
					fprintf(fp, "\n");
				}
				
			}
        }
		

		//close the file output stream
		fclose(fp);
	
}


void printMesh(float** meshArray){
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

float** makeContiguous2DArray(int r, int c){
    float** ary2D = (float**)malloc(sizeof(float*) * r);

    float* pool = (float*)malloc(sizeof(float) * r * c);

    int ri;
    for(ri = 0; ri < r; ri++){
        ary2D[ri] = pool + (ri * c);
    }

    return ary2D;
}

void freeContiguous2DArray(float** ary){
    free(ary[0]);   //free the pool
    free(ary);      //free the pointers
}