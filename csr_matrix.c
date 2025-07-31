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
    matrix.values = (double *)malloc(nnz * sizeof(double));

    if (!matrix.row_ptr || !matrix.col_ind || !matrix.values) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // BUild the CSR matrix
    int value_count = 0;
    matrix.row_ptr[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value;
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

CSRMatrix csrmatrix_from_sparse_file(FILE *file) {
    CSRMatrix matrix;
    int rows, cols, nnz;

    if (fscanf(file, "%d %d %d", &rows, &cols, &nnz) != 3) {
        fprintf(stderr, "Error reading matrix size\n");
        exit(EXIT_FAILURE);
    }

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;

    matrix.row_ptr = (int *)calloc((rows + 1), sizeof(int));
    matrix.col_ind = (int *)malloc(nnz * sizeof(int));
    matrix.values  = (double *)malloc(nnz * sizeof(double));

    if (!matrix.row_ptr || !matrix.col_ind || !matrix.values) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Temporary array to hold row indices
    int *temp_row = (int *)malloc(nnz * sizeof(int));
    if (!temp_row) {
        fprintf(stderr, "Memory allocation failed (temp_row)\n");
        exit(EXIT_FAILURE);
    }

    // Step 1: Read all triplets
    for (int i = 0; i < nnz; i++) {
        int row, col;
        double val;
        if (fscanf(file, "%d %d %lf", &row, &col, &val) != 3) {
            fprintf(stderr, "Error reading triplet at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        temp_row[i] = row;
        matrix.col_ind[i] = col;
        matrix.values[i] = val;
        matrix.row_ptr[row + 1]++;  // count entries per row
    }

    // Step 2: Cumulative sum to build row_ptr
    for (int i = 0; i < rows; i++) {
        matrix.row_ptr[i + 1] += matrix.row_ptr[i];
    }

    // Step 3 (optional): If needed, reorder values by row -> col within each row

    free(temp_row);
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
        if(i >= matrix->rows) {
            break; // Avoid printing beyond the matrix size
        }
        printf("Row %d: ", i);
        for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; j++) {
            printf("(%d, %.2f) ", matrix->col_ind[j], matrix->values[j]);
        }
        printf("\n");
    }
}