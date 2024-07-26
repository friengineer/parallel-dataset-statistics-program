#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "dataset_reading_method.h"

int main( int argc, char **argv )
{
    int i;

    // Initialise MPI and get the rank of this process, and the total number of processes.
    int rank, numProcs;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numProcs );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank     );

    // Check that the number of processes is a power of 2, but <= 256, so the data set, which is a multiple of 256 in length,
    // is also a multiple of the number of processes. If using OpenMPI, you may need to add the argument '--oversubscribe'
    // when launching the executable, to allow more processes than you have cores
    if ( (numProcs & (numProcs - 1)) != 0 || numProcs > 256 )
    {
        // Only display the error message from one process, but finalise and quit all of them
        if ( rank == 0 ) printf( "ERROR: Launch with a number of processes that is a power of 2 (i.e. 2, 4, 8, ...) and <= 256.\n" );

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Load the full data set onto rank 0
    float *globalData = NULL;
    int globalSize = 0;
    
    if ( rank == 0 )
    {
        // globalData must be freed on rank 0 before quitting
        globalData = readDataFromFile( &globalSize );
        if ( globalData == NULL )
        {
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        printf( "Rank 0: Read in data set with %d floats.\n", globalSize );
    }

    // Calculate the number of floats per process assuming that globalSize is a multiple of numProcs
    int localSize = globalSize / numProcs;

    // Start the timing now, after the data has been loaded (will only output on rank 0)
    double startTime = MPI_Wtime();

    // Inform processes of their data portion sizes using collective communication
    // and each process allocates memory for their data
    MPI_Bcast( &localSize, 1, MPI_INT, 0, MPI_COMM_WORLD );

    float *localData = (float*) malloc( localSize * sizeof(float) );

  	if ( !localData )
  	{
  		printf( "Could not allocate memory for the local data array on rank %d.\n", rank );
  		MPI_Finalize();
  		return EXIT_FAILURE;
  	}

    float globalMean;

    // Distribute data between processes using collective communication
    MPI_Scatter( globalData, localSize, MPI_FLOAT,
  		           localData, localSize, MPI_FLOAT,
  							 0, MPI_COMM_WORLD );

    float localSum = 0.0;

    // Each process calculates the mean of its portion of data
    for ( i = 0; i < localSize; i++ ) localSum += localData[i];
    float localMean = localSum / (localSize * numProcs);

    // Reduce all the local means into a single global mean
    MPI_Reduce( &localMean, &globalMean, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Inform processes what the calculated global mean is using collective communication
    MPI_Bcast( &globalMean, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );

    float globalVariance;

    localSum = 0.0;

    // Each process calculates the variance of its portion of data
    for ( i = 0; i < localSize; i++ ) localSum += (localData[i] - globalMean) * (localData[i] - globalMean);
    float localVariance = localSum / (localSize * numProcs);

    // Determine the number of levels in the binary tree
    int lev = 0;

    while ( 1 << lev < numProcs ) lev++;

    // Use point-to-point communication to reduce all processes' variances using a
    // binary tree
    for ( i = 1; i <= lev; i++ )
    {
      if ( rank >= (numProcs / (1 << i)) && rank < (2 * (numProcs / (1 << i))) )
      {
        MPI_Send( &localVariance, 1, MPI_FLOAT, rank - (numProcs / (1 << i)), 0, MPI_COMM_WORLD );
      }
      else if ( rank < (numProcs / (1 << i)) )
      {
        float next;
        MPI_Recv( &next, 1, MPI_FLOAT, rank + (numProcs / (1 << i)), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        localVariance += next;
      }
    }

    if ( rank == 0 ) globalVariance = localVariance;

    // Output the results alongside a serial check
    if ( rank == 0 )
    {
        // Output the results of the timing now, before moving onto other calculations
        printf( "Total time taken: %g s\n", MPI_Wtime() - startTime );

        finalMeanAndVariance( globalMean, globalVariance );

        // Mean
        float sum = 0.0;
        for ( i = 0; i < globalSize; i++ ) sum += globalData[i];
        float mean = sum / globalSize;

        // Variance
        float sumSqrd = 0.0;
        for ( i = 0; i < globalSize; i++ ) sumSqrd += ( globalData[i] - mean ) * ( globalData[i] - mean );
        float variance = sumSqrd / globalSize;

        printf( "SERIAL CHECK: Mean = %g and Variance = %g.\n", mean, variance );

    }

    // Free all resources (including any memory dynamically allocated), then quit
    if ( rank == 0 ) free( globalData );

    free( localData );

    MPI_Finalize();

    return EXIT_SUCCESS;
}
