#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    // Check if the number of arguments is correct
    if (argc < 3 || argc != 3 + atoi(argv[1]) * atoi(argv[2]))
    {
        printf("arguments are not correct");
        return 1;
    }

    // Get the number of rows and columns from the arguments
    int numRows = atoi(argv[1]);
    int numCols = atoi(argv[2]);

    // Allocate memory for the matrix
    int **matrix = (int **)malloc(numRows * sizeof(int *));
    for (int i = 0; i < numRows; i++)
    {
        matrix[i] = (int *)malloc(numCols * sizeof(int));
    }

    // Fill the matrix with values from the arguments
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            matrix[i][j] = atoi(argv[3 + i * numCols + j]);
        }
    }

    int result = 0;

    // Concatenate the values in each column and convert to integer
    for (int i = 0; i < numCols; i++)
    {
        char concatenatedValues[30] = "";
        for (int j = 0; j < numRows; j++)
        {
            char str[13];
            sprintf(str, "%d", matrix[j][i]);
            strcat(concatenatedValues, str);
        }
        result += atoi(concatenatedValues);
    }

    // Print the result
    printf("%d", result);

    // Free the allocated memory
    for (int i = 0; i < numRows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);

    return 0;
}