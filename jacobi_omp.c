/* Steven Smiley | COMP233 | Jacobi Iterations Open MP
*  Based on work by Argonne National Laboratory.
*  https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void writeToPPM(float* mesh, int iterations, const int MESHSIZE);
float lerp(float from, float to, float t);
void printMesh(float* meshArray, const int MESHSIZE);

int main(argc, argv)
int argc;
char** argv;
{
    const int MESHSIZE = 1000;                //size of the mesh to compute

    const float NORTH_BOUND = 100.0;        //north bounding value for the mesh
    const float SOUTH_BOUND = 100.0;        //south bounding value for the mesh
    const float EAST_BOUND = 0.0;          //east bounding value for the mesh
    const float WEST_BOUND = 0.0;          //west bounding value for the mesh
    const float INTERIOR_AVG =              //value to initialize interior mesh points with
        (NORTH_BOUND + SOUTH_BOUND + EAST_BOUND + WEST_BOUND) / 4.0;    //average the 4 bounds


    /*
    r: row iteration variable
    c: column iteration varible
    itrCount: count of iterations performed
    gDiffNorm: used for reduction of all the local diffNorms
    */
    int        r, c, itrCount;
    float     gDiffNorm;
    
    float* xNew;
    float* xFull;      //used to assemble the chunks into a full mesh

    float epsilon;          //threshold for gDiffNorm; used to stop the Jacobi iteration loop
    int maxIterations;      //threshold for itrCount; used to stop the Jacobi iteration loop
    
    int reqThreads;        //requested number of threads to be used for openmp

    double start, stop;     //timer variables

    //check that we have enough command line arguments
    if (argc < 4) {
        //print usage information to head
        
        printf("Please specify the correct number of arguments.\n");
        printf("Usage: jacobi_openmp [epsilon] [max_iterations] [threads]\n");
       
        return 0;
    }

    //allocate memory to 2d array
    xFull = (float*)malloc(MESHSIZE * MESHSIZE * sizeof(float));
    xNew = (float*)malloc(MESHSIZE * MESHSIZE * sizeof(float));

    //assign our epsilon and maxIteration variables using our command line arguments
    epsilon = strtod(argv[1], NULL);
    maxIterations = strtol(argv[2], NULL, 10);
    reqThreads = strtol(argv[3], NULL, 10);

    //print the standard header
    printf("Steven Smiley | COMP233 | Jacobi Iterations (OpenMP)\n");

    /* Fill the data as specified */
    for (r = 0; r < MESHSIZE; r++) {
        for (c = 0; c < MESHSIZE; c++) {
            xFull[r * MESHSIZE + c] = INTERIOR_AVG;    //set value for interior points
        }

        xFull[r * MESHSIZE + MESHSIZE - 1] = EAST_BOUND; //set value for east boundary
        xFull[r * MESHSIZE + 0] = WEST_BOUND;          //set value for west boundary
    }
    for (c = 0; c < MESHSIZE; c++) {
        xFull[0 * MESHSIZE + c] = NORTH_BOUND;   //set value for north boundary
        xFull[(MESHSIZE - 1) * MESHSIZE + c] = SOUTH_BOUND;    //set value for south boundary
    }

    for (r = 0; r < MESHSIZE; r++) {  //copy full into new array so that initial values in new array are correct
        for (c = 0; c < MESHSIZE; c++) {
            xNew[r * MESHSIZE + c] = xFull[r * MESHSIZE + c];
        }
    }


    start = omp_get_wtime(); //start the timer

    //Jacobi iteration computation loop
    itrCount = 0;   //initialize iteration count

    do {

        /* Compute new values (but not on boundary) */
        itrCount++;
        gDiffNorm = 0.0;

#pragma omp parallel for private(r,c) reduction(+:gDiffNorm) num_threads(reqThreads)
            for (r = 1; r < MESHSIZE - 1; r++)
                for (c = 1; c < MESHSIZE - 1; c++) {
                    xNew[r * MESHSIZE + c] = (xFull[r * MESHSIZE + c + 1] + xFull[r * MESHSIZE + c - 1] +     //new value computed as the average of its 4 neighbors
                        xFull[(r + 1) * MESHSIZE + c] + xFull[(r - 1) * MESHSIZE + c]) / 4.0;

                    gDiffNorm += (xNew[r * MESHSIZE + c] - xFull[r * MESHSIZE + c]) *           //compute gDiffNorm sum
                        (xNew[r * MESHSIZE + c] - xFull[r * MESHSIZE + c]);
                }
        

        //swap new to local using pointers
        float* tmp = xFull;
        xFull = xNew;
        xNew = tmp;
        
        gDiffNorm = sqrt(gDiffNorm);  //finish computation on the sum

        if(itrCount % 1000 == 0)
        printf("At iteration %d, diff is %e\n", itrCount,
            gDiffNorm);
        
    } while (gDiffNorm > epsilon && itrCount < maxIterations);  //keep doing Jacobi iterations until we hit our iteration or precision limit


    stop = omp_get_wtime(); //stop the timer

    printf("%d Jacobi iterations took %f seconds.\n", itrCount, stop - start);

    writeToPPM(xFull, itrCount, MESHSIZE);

    //free our dynamic memory
    free(xNew);
    free(xFull);

    //display normal termination message and exit
    printf("<normal termination>\n");
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

//print contents of 2d array to console (for testing purposes)
void printMesh(float* meshArray, const int MESHSIZE) {
    int r, c;   //loop control variables

    for (r = 0; r < MESHSIZE; r++) {
        for (c = 0; c < MESHSIZE; c++) {
            //print cell with width 5 and 1 digit after the decimal
            printf("%5.1f ", meshArray[r * MESHSIZE + c]);
        }
        //print newline
        printf("\n");
    }
}