#include <stdio.h>
#include <stdlib.h>
#include "conjugate_gradient.h"
#include "csr_matrix.h"
#include "cbm.h"


double* vector_from_file(FILE *file, int size);

int main(int argc, char *argv[]){
    // VECTOR x FOR DEBUGGING
    int initial_x_size;
    FILE *initial_x_file = fopen("test-cases/sin_vector.txt", "r");
    if (fscanf(initial_x_file, "%d", &initial_x_size) != 1) {
        fprintf(stderr, "Initial vector x size not supported\n");
        return 1;
    }
    double *initial_x = vector_from_file(initial_x_file, initial_x_size);

    // Create the elements for the conjugate gradient algorithm
    CBMatrix A_cbm = {
        .rows = initial_x_size,
        .bandwidth = 5,
        .values = (double[]){-1, -1, 5, -1, -1}
    };
    

    // Vector b
    double *b = calloc(initial_x_size, sizeof(double));
    if (!b) {
        fprintf(stderr, "Memory allocation for vector b failed\n");
        free(initial_x);
        fclose(initial_x_file);
        return 1;
    }

    // CONJUGATE GRADIENT
    Solver solver = setup_solver(initial_x_size, A_cbm, b, initial_x);
    conjugate_gradient(&solver);

    // Free memory
    free_solver(&solver);
    free(b);
    fclose(initial_x_file);

    return 1;
}

double* vector_from_file(FILE *file, int size) {
    double *vector = (double *)malloc(size * sizeof(double));
    if (!vector) {
        fprintf(stderr, "Memory allocation for vector failed\n");
        exit(EXIT_FAILURE);
    }      

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &vector[i]) != 1) {
            fprintf(stderr, "Error reading vector value at index %d\n", i);
            free(vector);
            exit(EXIT_FAILURE);
        }
    }

    return vector;
}