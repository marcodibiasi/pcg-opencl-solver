#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void zero_vector(char* fname, int size);
void identity_matrix(char* fname, int size);
void sin_vector(char* fname, int size);
void random_diagonal_matrix(char* fname, int size);

int main(){
    char *filename = "test-cases/zero_vector.txt";
    char *identity_filename = "test-cases/identity_matrix.txt";
    char *sin_filename = "test-cases/sin_vector.txt";
    char *random_diagonal_filename = "test-cases/random_diagonal_matrix.txt";
    int size = 1024; // Size of the vector

    zero_vector(filename, size);
    identity_matrix(identity_filename, size);
    sin_vector(sin_filename, size);
    random_diagonal_matrix(random_diagonal_filename, size);
    printf("Zero vector of size %d written to %s\n", size, filename);
    printf("Identity matrix of size %d written to %s\n", size, identity_filename);
    printf("Sine vector of size %d written to %s\n", size, sin_filename);
    printf("Random diagonal matrix of size %d written to %s\n", size, random_diagonal_filename);

    return 0;
}

void zero_vector(char* fname, int size) {
    FILE *file = fopen(fname, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", fname);
        exit(EXIT_FAILURE);
    }

    fprintf(file, "%d\n", size);
    for (int i = 0; i < size; i++) {
        fprintf(file, "0.0\n");
    }

    fclose(file);
}

void sin_vector(char* fname, int size) {
    FILE *file = fopen(fname, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", fname);
        exit(EXIT_FAILURE);
    }

    fprintf(file, "%d\n", size);
    for (int i = 0; i < size; i++) {
        fprintf(file, "%f\n", sin(i * 0.1));
    }

    fclose(file);
}

void identity_matrix(char* fname, int size) {
    FILE *file = fopen(fname, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", fname);
        exit(EXIT_FAILURE);
    }

    fprintf(file, "%d %d %d\n", size, size, size); // Size and number of non-zero elements
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                fprintf(file, "1.0 "); // Diagonal elements
            } else {
                fprintf(file, "0.0 "); // Off-diagonal elements
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void random_diagonal_matrix(char* fname, int size) {
    FILE *file = fopen(fname, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", fname);
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    fprintf(file, "%d %d %d\n", size, size, size); // nrows, ncols, non-zero elements

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                double value = 1.0 + (rand() % 10); // valori tra 1 e 10
                fprintf(file, "%.2f ", value);
            } else {
                fprintf(file, "0.0 ");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}