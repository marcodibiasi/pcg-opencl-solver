#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

void zero_vector(char* fname, int size);
void identity_matrix(char* fname, int size);
void sin_vector(char* fname, int size);
void random_diagonal_matrix(char* fname, int size);
void random_banded_matrix(char* fname, int size, int bandwidth);
void banded_matrix(char* fname, int size, int bandwidth);

int main(){
    char *filename = "test-cases/zero_vector.txt";
    char *identity_filename = "test-cases/identity_matrix.txt";
    char *sin_filename = "test-cases/sin_vector.txt";
    char *random_diagonal_filename = "test-cases/random_diagonal_matrix.txt";
    char *banded_filename = "test-cases/banded_matrix.txt";
    int size = 65536/16; // Size of the vector

    zero_vector(filename, size);
    // identity_matrix(identity_filename, size);
    sin_vector(sin_filename, size);
    // random_diagonal_matrix(random_diagonal_filename, size);
    banded_matrix(banded_filename, size, 2);
    printf("Zero vector of size %d written to %s\n", size, filename);
    // printf("Identity matrix of size %d written to %s\n", size, identity_filename);
    printf("Sine vector of size %d written to %s\n", size, sin_filename);
    // printf("Random diagonal matrix of size %d written to %s\n", size, random_diagonal_filename);
    printf("Random banded matrix of size %d written to %s\n", size, banded_filename);

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
                fprintf(file, "10000.0 "); // Diagonal elements
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


void banded_matrix(char* fname, int size, int bandwidth) {
    FILE *file = fopen(fname, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", fname);
        exit(EXIT_FAILURE);
    }

    int nvals = 0;
    for (int i = 0; i < size; i++) {
        int start = MAX(0, i - bandwidth);
        int end = MIN(size - 1, i + bandwidth);
        nvals += end - start + 1;
    }

    fprintf(file, "%d %d %d\n", size, size, nvals);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (j >= MAX(0, i - bandwidth) && j <= MIN(size - 1, i + bandwidth)) {
                float value = (i == j) ? 5.0f : -1.0f;  // 1.0 sulla diagonale
                fprintf(file, "%.2f ", value);
            } else {
                fprintf(file, "0.0 ");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void random_banded_matrix(char* fname, int size, int bandwidth) {
    float** L = malloc(size * sizeof(float*));
    for (int i = 0; i < size; i++) {
        L[i] = calloc(size, sizeof(float));
        for (int j = MAX(0, i - bandwidth); j <= i; j++) {
            if (i == j)
                L[i][j] = 2.0 + (rand() % 1);  // garantisce valori positivi sulla diagonale
            else
                L[i][j] = ((float)rand() / RAND_MAX); // piccoli valori off-diagonali
        }
    }

    FILE *file = fopen(fname, "w");
    if (!file) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    // Scrivi intestazione provvisoria (nnz da aggiornare dopo)
    long header_pos = ftell(file);
    fprintf(file, "%d %d %06d\n", size, size, 0);

    int nnz = -2;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0;
            for (int k = 0; k <= MIN(i, j); k++) {
                sum += L[i][k] * L[j][k];
            }

            if (sum != 0.0f) nnz++;  // conteggio preciso dei non nulli

            fprintf(file, "%.2f ", sum);
        }
        fprintf(file, "\n");
    }

    // Sovrascrivi il valore corretto di nnz
    fseek(file, header_pos, SEEK_SET);
    fprintf(file, "%d %d %06d\n", size, size, nnz);

    fclose(file);

    for (int i = 0; i < size; i++) free(L[i]);
    free(L);
}