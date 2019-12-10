/* Steven Smiley | COMP233 | Jacobi Iterations
*  Based on work by Argonne National Laboratory.
*  https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

float lerp(float from, float to, float t);
void writeToPPM(float* mesh, int iterations, const int MESHSIZE);
int getChunkRows(int rank, int commSize, const int MESHSIZE);
int getChunkSize(int rank, int commSize, const int MESHSIZE);

int main( argc, argv )
int argc;
char **argv;
{
    const int MESHSIZE = 1000;                //size of the mesh to compute
    
    const float NORTH_BOUND = 100.0;        //north bounding value for the mesh
    const float SOUTH_BOUND = 100.0;        //south bounding value for the mesh
    const float EAST_BOUND = 0.0;          //east bounding value for the mesh
    const float WEST_BOUND = 0.0;          //west bounding value for the mesh
    const float INTERIOR_AVG =              //value to initialize interior mesh points with
        (NORTH_BOUND + SOUTH_BOUND + EAST_BOUND + WEST_BOUND) / 4.0;    //average the 4 bounds


    /*
    rank: MPI rank of the current process
    commSize: Number of processes in the MPI communicator
    r: row iteration variable
    c: column iteration varible
    itrCount: count of iterations performed
    rFirst: initial boundary for r
    rLast: end boundary for r
    status: used for storing the status reported from MPI communications
    diffNorm: used for local differential sum
    gDiffNorm: used for reduction of all the local diffNorms
    */
    int        rank, commSize, r, c, itrCount;
    int        rFirst, rLast;
    MPI_Status status;
    float     diffNorm, gDiffNorm;

    float*     xLocal;     //stores local chunk of mesh
    float*     xNew;       //stores new local chunk of mesh
    float*     xFull;      //used to assemble the chunks into a full mesh

    float epsilon;          //threshold for gDiffNorm; used to stop the Jacobi iteration loop
    int maxIterations;      //threshold for itrCount; used to stop the Jacobi iteration loop

    int reqThreads;         //number of threads the user requested for openmp

    double start, stop;     //timer variables

    MPI_Init( &argc, &argv );

    //Initialize MPI communicator rank and size variables.
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &commSize );

    //initialize constants dependent on communicator size
    const int CHUNKROWS = getChunkRows(rank, commSize, MESHSIZE);     //number of rows in a process chunk
    const int CHUNKSIZE = getChunkSize(rank, commSize, MESHSIZE);     //number of floats in a process chunk

    
    //check that we have enough command line arguments
    if(argc < 4){
        //print usage information to head
        if(rank == 0){
            printf("Please specify the correct number of arguments.\n");
            printf("Usage: jacobi [epsilon] [max_iterations] [threads]\n");
        }

        //exit the program
        MPI_Finalize();
        return 0;
    }

    //allocate memory to 2d arrays
    xLocal = (float*)malloc((CHUNKROWS + 2) * MESHSIZE * sizeof(float));
    xNew = (float*)malloc((CHUNKROWS + 2) * MESHSIZE * sizeof(float));
    if(rank == 0) xFull = (float*)malloc(MESHSIZE * MESHSIZE * sizeof(float));  //only the master needs to use this

    //assign our epsilon and maxIteration variables using our command line arguments
    epsilon = strtod(argv[1], NULL);
    maxIterations = strtol(argv[2], NULL, 10);
    reqThreads = strtol(argv[3], NULL, 10);
    
    //print the standard header
    printf("Steven Smiley | COMP233 | Jacobi Iterations (MPI)\n");

    /* Note that top and bottom processes have one less row of interior
       points */
    rFirst = 1;
    rLast  = CHUNKROWS;
    if (rank == 0)        rFirst++;
    if (rank == commSize - 1) rLast--;

    /* Fill the data as specified */
    for (r = 1; r <= CHUNKROWS; r++) {
        for(c = 0; c < MESHSIZE; c++){
            xLocal[r * MESHSIZE + c] = INTERIOR_AVG;    //set value for interior points
        }

        xLocal[r * MESHSIZE + MESHSIZE-1] = EAST_BOUND; //set value for east boundary
        xLocal[r * MESHSIZE + 0] = WEST_BOUND;          //set value for west boundary
    }
    for (c=0; c<MESHSIZE; c++) {
	    xLocal[(rFirst-1) * MESHSIZE + c] = NORTH_BOUND;   //set value for north boundary
	    xLocal[(rLast+1) * MESHSIZE + c] = SOUTH_BOUND;    //set value for south boundary
    }
    
    for (r=0; r<CHUNKROWS+2; r++) 
	    for (c=0; c<MESHSIZE; c++) {
         xNew[r * MESHSIZE + c] = xLocal[r * MESHSIZE + c];
         }
         
         	
    if(rank == 0)   //start the timer on the master
        start = MPI_Wtime();

    //Jacobi iteration computation loop
    itrCount = 0;   //initialize iteration count
    do {
	/* Send up unless I'm at the top, then receive from below */
	/* Note the use of xlocal[i] for &xlocal[i][0] */
	if (rank < commSize - 1) 
	    MPI_Send( xLocal + (CHUNKROWS * MESHSIZE), MESHSIZE, MPI_FLOAT, rank + 1, 0, 
		      MPI_COMM_WORLD );
	if (rank > 0)
	    MPI_Recv( xLocal, MESHSIZE, MPI_FLOAT, rank - 1, 0, 
		      MPI_COMM_WORLD, &status );

	/* Send down unless I'm at the bottom */
	if (rank > 0) 
	    MPI_Send( xLocal + (1 * MESHSIZE), MESHSIZE, MPI_FLOAT, rank - 1, 1, 
		      MPI_COMM_WORLD );
	if (rank < commSize - 1) 
	    MPI_Recv( xLocal + ((CHUNKROWS+1) * MESHSIZE), MESHSIZE, MPI_FLOAT, rank + 1, 1, 
		      MPI_COMM_WORLD, &status );


	/* Compute new values (but not on boundary) */
	itrCount ++;
	diffNorm = 0.0;
    
#pragma omp parallel for private(r,c) reduction(+:diffNorm) num_threads(reqThreads)
	for (r=rFirst; r<=rLast; r++) 
	    for (c=1; c<MESHSIZE-1; c++) {
		xNew[r * MESHSIZE + c] = (xLocal[r * MESHSIZE + c+1] + xLocal[r * MESHSIZE + c-1] +     //new value computed as the average of its 4 neighbors
			      xLocal[(r+1) * MESHSIZE + c] + xLocal[(r-1) * MESHSIZE + c]) / 4.0;
		diffNorm += (xNew[r * MESHSIZE + c] - xLocal[r * MESHSIZE + c]) *           //compute diffNorm sum
		            (xNew[r * MESHSIZE + c] - xLocal[r * MESHSIZE + c]);
	    }

    //swap new to local using pointers
    float* tmp = xLocal;
    xLocal = xNew;
    xNew = tmp;

    //reduce value for diffNorm
	MPI_Allreduce( &diffNorm, &gDiffNorm, 1, MPI_FLOAT, MPI_SUM,
		       MPI_COMM_WORLD );
	gDiffNorm = sqrt( gDiffNorm );  //finish computation on the sum
	if (rank == 0 && itrCount % 1000 == 0) printf( "At iteration %d, diff is %e\n", itrCount, 
			       gDiffNorm );
    } while (gDiffNorm > epsilon && itrCount < maxIterations);  //keep doing Jacobi iterations until we hit our iteration or precision limit

    
    //assemble mesh into full mesh and write to ppm
    if(rank == 0){//master
        //copy the masters chunk into the full mesh
        memcpy(xFull, xLocal + (1 * MESHSIZE), CHUNKSIZE * sizeof(float));

        int proc;
        int pointerOffset = 0;
        for(proc = 1; proc < commSize; proc++){
            //adjust the pointer offset based on how many elements the previous process put in the array
            pointerOffset += getChunkSize(proc - 1, commSize, MESHSIZE);

            //recieve each array chunk from other processes and assemble into full mesh
            //getChunkSize is used here instead of local CHUNKSIZE, because the chunk size can vary by process
            MPI_Recv(xFull + pointerOffset, getChunkSize(proc, commSize, MESHSIZE), MPI_FLOAT, proc, 0, MPI_COMM_WORLD, &status); 
        }

        //stop the timer, because the calculation is complete
        stop = MPI_Wtime();

        printf("%d Jacobi iterations took %f seconds.\n", itrCount, stop - start);

        //write the full mesh to ppm output file
        writeToPPM(xFull, itrCount, MESHSIZE);
    }
    else{//slaves
            //send chunks to master
            MPI_Send(xLocal + (1 * MESHSIZE), CHUNKSIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    

    //free our dynamic memory
    free(xLocal);
    free(xNew);
    if(rank == 0) free(xFull);  //only free from master because we only initialized on master

    //display normal termination message and exit
    if(rank == 0) printf("<normal termination>\n");
    
    MPI_Finalize();
    return 0;
}

//Write our PPM image from a 2d array
void writeToPPM(float* mesh, int iterations, const int MESHSIZE) {
	//make a file for our ppm data
	FILE* fp = fopen("jacobi.ppm", "w");

	//write the data out to the file
	//output header information to ppm file
    fprintf(fp, "P3 %d %d 255\n", MESHSIZE, MESHSIZE);
    fprintf(fp, "#Steven Smiley | COMP233 | OpenMP Mandelbrot\n");
    fprintf(fp, "#This image took %d iterations to converge.\n", iterations);

    int r, c;
	for (r = 0; r < MESHSIZE; r++) {
		for (c = 0; c < MESHSIZE; c++) {
            float temp = mesh[r * MESHSIZE + c];

            int red = lerp(0.0, 255.0, temp);
            int blue = lerp(255.0, 0.0, temp);

			//write our color values to the buffer
			fprintf(fp, "%d 0 %d ", red, blue);
			if (c % 5 == 4) {
				//write a newline every 5th rgb value
				fprintf(fp, "\n");
			}
			
		}
    }
	
	//close the output file
	fclose(fp);
}

//linear interpolation between 2 values.
//t is the point to interpolate at (divided by 100 because that is our maximum temp)
float lerp(float from, float to, float t) {
    return from + (t / 100.0f) * (to - from);   
}

//calculates and returns how many rows that a process with a certain rank should get
int getChunkRows(int rank, int commSize, const int MESHSIZE){
    int chunkRows = MESHSIZE / commSize;
    int remainder = MESHSIZE % commSize;

    if(rank < remainder){   //the first remainder processes will get one extra row of data
        return chunkRows + 1;
    }

    return chunkRows;
}

//calculates and returns how many elements that a process with a certain rank has in its chunk
int getChunkSize(int rank, int commSize, const int MESHSIZE){
    return getChunkRows(rank, commSize, MESHSIZE) * MESHSIZE;
}