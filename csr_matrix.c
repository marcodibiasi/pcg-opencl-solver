#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "csr_matrix.h"

CSRMatrix csrmatrix_from_file(FILE *file) {
    CSRMatrix matrix;
    int cols, rows, nnz;

    // Read the size of the matrix
    if (fscanf(file, "%d %d %d", &cols, &rows, &nnz) != 3) {
        fprintf(stderr, "Error reading matrix size\n");
        exit(EXIT_FAILURE);
    }

    matrix.cols = cols;    
    matrix.rows = rows; 
    matrix.nnz = nnz;

    matrix.row_ptr = (int *)malloc((rows + 1) * sizeof(int));
    matrix.col_ind = (int *)malloc(nnz * sizeof(int));
    matrix.values = (float *)malloc(nnz * sizeof(float));

    if (!matrix.row_ptr || !matrix.col_ind || !matrix.values) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // BUild the CSR matrix
    int value_count = 0;
    matrix.row_ptr[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value;
            if (fscanf(file, "%f", &value) != 1) {
                fprintf(stderr, "Error reading matrix value at (%d, %d)\n", i, j);
                exit(EXIT_FAILURE);
            }

            if (value != 0.0f) {
                if (value_count >= nnz) {
                    fprintf(stderr, "Too many non-zero values compared to header\n");
                    exit(EXIT_FAILURE);
                }
                matrix.col_ind[value_count] = j;
                matrix.values[value_count] = value;
                value_count++;
            }
        }
        matrix.row_ptr[i + 1] = value_count;
    }

    if (value_count != nnz) {
        printf("\nNon-zero counted: %d, Expected: %d", value_count, nnz);
        fprintf(stderr, "Number of non-zero elements does not match expected count\n");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

void csrmatrix_free(CSRMatrix *matrix) {
    if (matrix) {
        free(matrix->row_ptr);
        free(matrix->col_ind);
        free(matrix->values);
        matrix->row_ptr = NULL;
        matrix->col_ind = NULL;
        matrix->values = NULL;
    }
}

void csrmatrix_print(const CSRMatrix *matrix, int length) {
    if (!matrix) {
        fprintf(stderr, "Matrix is NULL\n");
        return;
    }

    printf("CSR Matrix:\n");
    printf("Rows: %d, Cols: %d, Non-zero elements: %d\n", matrix->rows, matrix->cols, matrix->nnz);
    
    for (int i = 0; i < length; i++) {
        printf("Row %d: ", i);
        for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; j++) {
            printf("(%d, %.2f) ", matrix->col_ind[j], matrix->values[j]);
        }
        printf("\n");
    }
}