#include "csr_matrix.h"
#include "cbm.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

typedef struct{
    cl_kernel partial_sum_reduction;
    cl_kernel reduce_sum4_double4_sliding;
    cl_kernel dot_product;
    cl_kernel update_x;
    cl_kernel update_r;
    cl_kernel update_p;
    cl_kernel dot_product_vec4;
    cl_kernel mat_vec_multiply;
    cl_kernel compressed_matvec_mult;
    cl_kernel mult_vectors;
    cl_kernel sum_vectors;
    cl_kernel scale_vector;
    cl_kernel get_inverted_diagonal;
    cl_kernel get_inverted_diagonal_compressed;
} OpenCLKernels;

typedef struct{
    cl_platform_id p;
	cl_device_id d;
	cl_context ctx;
	cl_command_queue q;
	cl_program prog;

    cl_mem x_buffer; 
    cl_mem b_buffer;

    // CSRMatrix buffers
    cl_mem csr_row_ptr_buffer;
    cl_mem csr_col_ind_buffer;
    cl_mem csr_values_buffer;

    // CBM representation
    cl_int cbm_rows;
    cl_int cbm_bandwidth;
    cl_mem cbm_values_buffer;

    size_t lws;

    OpenCLKernels kernels;
} OpenCLContext;


typedef struct {
    int size;

    CSRMatrix A; 
    CBMatrix A_cbm;

    double *x; 
    double *b;    
    OpenCLContext cl;
} Solver;

typedef struct{
    cl_mem Ap;      // Used in alpha_calculate
    cl_mem x;  // Used in update_x
    cl_mem p;  // Used in update_p
} TemporaryBuffers;

Solver setup_solver(int size, CSRMatrix A, CBMatrix A_cbm, double *b, double *initial_x);
OpenCLContext setup_opencl_context(Solver solver);
TemporaryBuffers init_buffers(Solver* solver, int length);
void conjugate_gradient(Solver* solver);
double alpha_calculate(Solver* solver, cl_mem* r, cl_mem* z, cl_mem* p, TemporaryBuffers* temp);
double beta_calculate(Solver* solver, cl_mem* r_next, cl_mem* z_next, cl_mem* r, cl_mem* z); 
// void update_x(Solver* solver, cl_mem* p, double alpha, int length, TemporaryBuffers* temp);
void update_r(Solver* solver, cl_mem* r, cl_mem* p, cl_mem* r_next, double alpha, int length);
// void update_p(Solver* solver, cl_mem* r, cl_mem* p, double beta, int length, TemporaryBuffers* temp);
double dot_product_handler(Solver* solver, cl_mem* vec1, cl_mem* vec2, int lenght);
cl_event dot_product(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length);
cl_event dot_product_vec4(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length);
cl_event partial_sum_reduction(Solver *solver, cl_mem* in_buf, cl_mem* out_buf, int num_groups);
cl_event get_inverted_diagonal(Solver* solver, cl_mem* diagonal, int length);
cl_event sum_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int lenght);
cl_event scale_vector(Solver* solver, cl_mem* vec, double scale, cl_mem* result, int lenght);
cl_event mat_vec_multiply(Solver *solver, cl_mem* vec, cl_mem* result);
cl_event compressed_matvec_mult(Solver* solver, cl_mem* vec, cl_mem* result);
cl_event mult_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length);
cl_event update_x(Solver *solver, cl_mem* p, double alpha, int length);
cl_event update_p(Solver *solver, cl_mem* p, cl_mem* z, double beta, int length);
cl_event update_r_evt(Solver *solver, cl_mem* r, cl_mem* r_next, double alpha, int length);
cl_event get_inverted_diagonal_compressed(Solver* solver, cl_mem* diagonal_buffer);
void free_solver(Solver* solver);
void free_temporarybuffers(TemporaryBuffers* temp);
void print_buffer(OpenCLContext *cl, cl_mem buf, size_t size, int n);
double profiling_event(cl_event event);
