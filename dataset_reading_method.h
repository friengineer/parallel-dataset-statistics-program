#include <stdio.h>
#include <stdlib.h>

// Reads data from the file "dataSet.txt", allocating memory and returning a pointer to the array - this memory will
// need to be freed elsewhere. Also changes the value of size in-place to reflect the total number of floats in the data set
#define MAX_LINE_LENGTH 20
float* readDataFromFile( int *size )
{
    // Try to open up the file for reading
    FILE *fileIn = fopen( "dataSet.txt", "rt" );
    if( !fileIn )
    {
        printf( "Could not open the local file 'dataSet.txt' for reading.\n" );
        return NULL;
    }

    // Assume all lines are less than MAX_LINE_LENGTH bytes
    char line[MAX_LINE_LENGTH];

    // The first line should be a single integer, corresponding to the total number of data items
    fgets( line, MAX_LINE_LENGTH, fileIn );
    *size = atoi( line );
    if( *size <= 0 )
    {
        printf( "Could not parse first line of 'dataSet.txt' as a positive integer; has the file been corrupted?\n" );
        fclose( fileIn );
        return NULL;
    }

    // Allocate memory for the full data list
    float *data = (float*) malloc( (*size)*sizeof(float) );
    if( data==NULL )
    {
        printf( "Could not allocate memory for the %i entries expected for the data list.\n", *size );
        fclose( fileIn );
        return NULL;
    }

    // Read in the floats line by line
    int lineNum;
    for( lineNum=0; lineNum<*size; lineNum++ )
    {
        fgets( line, MAX_LINE_LENGTH, fileIn );
        data[lineNum] = atof( line );
    }

    // Close the file and return with the pointer to the data (which will need to be freed somewhere else)
    fclose( fileIn );
    return data;
}

// Displays the output of the mean and variance
void finalMeanAndVariance( float mean, float variance )
{
    printf( "FINAL RESULT: Mean=%g and Variance=%g.\n", mean, variance );
}
