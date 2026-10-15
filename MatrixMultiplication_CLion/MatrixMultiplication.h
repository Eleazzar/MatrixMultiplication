#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

/**
 * Result structure returning kernel elapsed time and result matrix pointer.
 */
typedef struct {
    double elapsed_time;
    double** C;
} Result;

double** allocate_matrix(int n);
void free_matrix(double** mat, int n);
double** generate_matrix(int n);
Result multiplication(double** A, double** B, int n);

#endif // MATRIX_MULTIPLICATION_H