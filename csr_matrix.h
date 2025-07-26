#ifndef MATRIX_H
#define MATRIX_H

typedef struct CSRMatrix {
    int rows, cols;
    int nnz; 
    int *row_ptr;
    int *col_ind;
    float *values;
} CSRMatrix;

CSRMatrix csrmatrix_from_file(FILE *file);
void csrmatrix_free(CSRMatrix *matrix);
void csrmatrix_print(const CSRMatrix *matrix, int length);

#endif // MATRIX_H