#include "csr_matrix.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

typedef struct{
    cl_kernel partial_sum_reduction;
    cl_kernel dot_product;
    cl_kernel mat_vec_multiply;
    cl_kernel mult_vectors;
    cl_kernel sum_vectors;
    cl_kernel scale_vector;
    cl_kernel get_inverted_diagonal;
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

    size_t lws;

    OpenCLKernels kernels;
} OpenCLContext;


typedef struct {
    int size;

    CSRMatrix A; 
    float *x; 
    float *b;    
    OpenCLContext cl;
} Solver;

Solver setup_solver(int size, CSRMatrix A, float *b, float *initial_x);
OpenCLContext setup_opencl_context(Solver solver);
void conjugate_gradient(Solver* solver);
float alpha_calculate(Solver* solver, cl_mem* r, cl_mem* z, cl_mem* p);
float beta_calculate(Solver* solver, cl_mem* r_next, cl_mem* z_next, cl_mem* r, cl_mem* z); 
void update_x(Solver* solver, cl_mem* p, float alpha, int length);
void update_r(Solver* solver, cl_mem* r, cl_mem* p, cl_mem* r_next, float alpha, int length);
void update_p(Solver* solver, cl_mem* r, cl_mem* p, float beta, int length);
float dot_product_handler(Solver* solver, cl_mem* vec1, cl_mem* vec2, int lenght);
cl_event dot_product(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length);
cl_event partial_sum_reduction(Solver *solver, cl_mem* in_buf, cl_mem* out_buf, int num_groups);
cl_event get_inverted_diagonal(Solver* solver, cl_mem* diagonal, int length);
cl_event sum_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int lenght);
cl_event scale_vector(Solver* solver, cl_mem* vec, float scale, cl_mem* result, int lenght);
cl_event mat_vec_multiply(Solver *solver, cl_mem* vec, cl_mem* result);
cl_event mult_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length);
void free_solver(Solver* solver);
void print_buffer(OpenCLContext *cl, cl_mem buf, size_t size, int n);
