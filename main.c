#include <stdio.h>
#include <stdlib.h>
#include "conjugate_gradient.h"
#include "csr_matrix.h"

float* vector_from_file(FILE *file, int size);

int main(int argc, char *argv[]){
    /*
    A: text file with first line containing the size of the matrix and the number of non-zero elements
    b: text file with first line containing the size of the vector
    
    !! the matrix must be square and the vector must have the same number of rows as the matrix !!
    */

    if (argc < 3) {
        fprintf(stderr, "Wrong arguments\n");
        return 1;
    }

    FILE *A_file = fopen(argv[1], "r");
    if (!A_file) { 
        fprintf(stderr, "Failed to open matrix file: %s\n", argv[1]);
        return 1;
    }

    FILE *b_file = fopen(argv[2], "r");
    if (!b_file) {
        fprintf(stderr, "Failed to open vector file: %s\n", argv[2]);
        fclose(A_file);
        return 1;
    }

    // Create the elements for the conjugate gradient algorithm

    // MATRIX A
    CSRMatrix A = csrmatrix_from_file(A_file);
    if (A.rows != A.cols) {
        fprintf(stderr, "Matrix must be square\n");
        fclose(A_file);
        fclose(b_file);
        return 1;
    }

    // VECTOR b
    int b_size;
    if (fscanf(b_file, "%d", &b_size) != 1 || b_size != A.rows) {
        fprintf(stderr, "Vector size not supported\n");
        fclose(A_file);
        fclose(b_file);
        return 1;
    }
    float *b = vector_from_file(b_file, b_size);

    // VECTOR x TO DEBUG
    int initial_x_size;
    FILE *initial_x_file = fopen("test-cases/sin_vector.txt", "r");
    if (fscanf(initial_x_file, "%d", &initial_x_size) != 1 || initial_x_size != A.rows) {
        fprintf(stderr, "Initial vector x size not supported\n");
        free(b);
        fclose(A_file);
        fclose(b_file);
        return 1;
    }
    float *initial_x = vector_from_file(initial_x_file, initial_x_size);

    // CONJUGATE GRADIENT
    Solver solver = setup_solver(A.rows, A, b, initial_x);
    conjugate_gradient(&solver);

    // Free memory
    free_solver(&solver);
    csrmatrix_free(&A);
    free(b);
    fclose(initial_x_file);
    fclose(A_file);
    fclose(b_file);

    return 1;
}

float* vector_from_file(FILE *file, int size) {
    float *vector = (float *)malloc(size * sizeof(float));
    if (!vector) {
        fprintf(stderr, "Memory allocation for vector failed\n");
        exit(EXIT_FAILURE);
    }      

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &vector[i]) != 1) {
            fprintf(stderr, "Error reading vector value at index %d\n", i);
            free(vector);
            exit(EXIT_FAILURE);
        }
    }

    return vector;
}