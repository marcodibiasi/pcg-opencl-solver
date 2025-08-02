#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>
#include "conjugate_gradient.h"
#include "ocl_boiler.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

Solver setup_solver(int size, CBMatrix A_cbm, double *b, double *initial_x) {
    Solver solver;
    solver.size = size;
    // solver.A = A;
    solver.A_cbm = A_cbm;
    solver.b = b;
    solver.x = initial_x;

    if (!solver.x) {
        solver.x = malloc(size * sizeof(double));
        if (!solver.x) {
            fprintf(stderr, "Failed to allocate memory for x vector\n");
            exit(1);
        }
        for (int i = 0; i < size; i++) {
            solver.x[i] = 0.0;  // Initialize x to zero
        }
    }
    solver.cl = setup_opencl_context(solver);  
    return solver;
}

OpenCLContext setup_opencl_context(Solver solver) {
    OpenCLContext cl;

    cl.p = select_platform();
	cl.d = select_device(cl.p);
	cl.ctx = create_context(cl.p, cl.d); 
	cl.q = create_queue(cl.ctx, cl.d);
	cl.prog = create_program("kernels.cl", cl.ctx, cl.d);

	cl_int err;

    // Allocate OpenCL buffers
	cl.b_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver.size * sizeof(double), solver.b, &err);
	ocl_check(err, "clCreateBuffer failed for b_buffer");

    cl.x_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        solver.size * sizeof(double), solver.x, &err);
    ocl_check(err, "clCreateBuffer failed for x_buffer");

    //CSR Matrix buffers
    // cl.csr_row_ptr_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    //     (solver.A.rows + 1) * sizeof(int), solver.A.row_ptr, &err);
    // ocl_check(err, "clCreateBuffer failed for csr_row_ptr_buffer");

    // cl.csr_col_ind_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    //     solver.A.nnz * sizeof(int), solver.A.col_ind, &err);
    // ocl_check(err, "clCreateBuffer failed for csr_col_ind_buffer");

    // cl.csr_values_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    //     solver.A.nnz * sizeof(double), solver.A.values, &err);
    // ocl_check(err, "clCreateBuffer failed for csr_values_buffer");

    // CBM representation
    cl.cbm_rows = solver.A_cbm.rows;
    cl.cbm_bandwidth = solver.A_cbm.bandwidth;
    cl.cbm_values_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver.A_cbm.bandwidth * sizeof(double), solver.A_cbm.values, &err);
    
    // Create kernels
    cl.kernels.partial_sum_reduction = clCreateKernel(cl.prog, "partial_sum_reduction", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.reduce_sum4_double4_sliding = clCreateKernel(cl.prog, "reduce_sum4_double4_sliding", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.dot_product = clCreateKernel(cl.prog, "dot_product", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.dot_product_vec4 = clCreateKernel(cl.prog, "dot_product_vec4", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.mat_vec_multiply = clCreateKernel(cl.prog, "mat_vec_multiply", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.compressed_matvec_mult = clCreateKernel(cl.prog, "compressed_matvec_mult", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.mult_vectors = clCreateKernel(cl.prog, "mult_vectors", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.sum_vectors = clCreateKernel(cl.prog, "sum_vectors", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.scale_vector = clCreateKernel(cl.prog, "scale_vector", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.get_inverted_diagonal = clCreateKernel(cl.prog, "get_inverted_diagonal", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.get_inverted_diagonal_compressed = clCreateKernel(cl.prog, "get_inverted_diagonal_compressed", &err);
    ocl_check(err, "clCreateKernel failed for get_inverted_diagonal_compressed");

    cl.kernels.update_x = clCreateKernel(cl.prog, "update_x", &err);
    ocl_check(err, "clCreateKernel failed for update_x");

    cl.kernels.update_r = clCreateKernel(cl.prog, "update_r_and_z", &err);
    ocl_check(err, "clCreateKernel failed for update_r");

    cl.kernels.update_p = clCreateKernel(cl.prog, "update_p", &err);
    ocl_check(err, "clCreateKernel failed for update_p");

    cl.lws = 32;

    return cl;
}

TemporaryBuffers init_buffers(Solver* solver, int length) {
    cl_int err;
    TemporaryBuffers temp;

    temp.Ap = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for temp.Ap");

    temp.x = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for temp.x");

    temp.p = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for temp.p");

    return temp;
}

void free_temporarybuffers(TemporaryBuffers* temp) {
    if (temp->Ap) clReleaseMemObject(temp->Ap);
    if (temp->x)  clReleaseMemObject(temp->x);
    if (temp->p)  clReleaseMemObject(temp->p);
}

void conjugate_gradient(Solver* solver) {
    cl_int err;
    OpenCLContext* cl = &solver->cl;

    // SETUP 
    int length = solver->size;
    double r_norm;   // Residue norm
    double epsilon = 1e-10;  // Convergence threshold
    int max_iter = length;   // Maximum iterations
    double alpha;    // Step size along the search direction
    int k = 0;  // Iteration counter

    TemporaryBuffers temp = init_buffers(solver, length);

    /*
    STEP ZERO: Precondition with Jacobi by extracting the diagonal of the matrix A
    and inverting it
    */
    cl_mem diagonal_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err); 
    ocl_check(err, "clCreateBuffer failed for diagonal_buffer");

    // cl_event inverted_diagonal = get_inverted_diagonal(solver, &diagonal_buffer, length);
    // cl_event inverted_diagonal = get_inverted_diagonal_compressed(solver, &diagonal_buffer, length);
    cl_event inverted_diagonal = get_inverted_diagonal_compressed(solver, &diagonal_buffer);
    clWaitForEvents(1, &inverted_diagonal);
    clReleaseEvent(inverted_diagonal);

    /*
    STEP ONE: Calculate initial residue r = b - Ax
    */
    cl_mem r_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_buffer");

    //cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, &cl->x_buffer, &r_buffer);
    cl_event mat_vec_multiply_evt = compressed_matvec_mult(solver, &cl->x_buffer, &r_buffer);
    clWaitForEvents(1, &mat_vec_multiply_evt);
    clReleaseEvent(mat_vec_multiply_evt);

    cl_event scale_vector_evt = scale_vector(solver, &r_buffer, -1, &r_buffer, length);
    clWaitForEvents(1, &scale_vector_evt);
    clReleaseEvent(scale_vector_evt);

    cl_event sum_vectors_evt = sum_vectors(solver, &cl->b_buffer, &r_buffer, &r_buffer, length);
    clWaitForEvents(1, &sum_vectors_evt);
    clReleaseEvent(sum_vectors_evt);

    /*
    STEP TWO: Preconditioned residue z = D^(-1) * r
    where D is the diagonal of the matrix A
    */
    cl_mem z_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for z_buffer");

    cl_event preconditioned_residue_evt = mult_vectors(solver, &diagonal_buffer, &r_buffer, &z_buffer, length);
    ocl_check(err, "mult_vectors failed for preconditioned_residue");
    clWaitForEvents(1, &preconditioned_residue_evt);
    clReleaseEvent(preconditioned_residue_evt);

    /*
    STEP THREE: set first search direction p = z 
    */
    cl_mem direction_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
	ocl_check(err, "clCreateBuffer failed for direction_buffer");

    cl_event copy_buffer_evt;
    err = clEnqueueCopyBuffer(cl->q, z_buffer, direction_buffer, 0, 0, 
        length * sizeof(double), 0, NULL, &copy_buffer_evt);
    ocl_check(err, "clEnqueueCopyBuffer failed for direction_buffer");

    clWaitForEvents(1, &copy_buffer_evt);
    clReleaseEvent(copy_buffer_evt);

    /*
    STEP FOUR: main loop of the Conjugate Gradient algorithm
    */
    // Allocate buffers for the next iteration
    cl_mem r_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_next_buffer");

    cl_mem z_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(double), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_next_buffer");

    double r_dot_z;

    // profiling 
    struct timespec start, end;
    struct timespec iter_start, iter_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    do {
        clock_gettime(CLOCK_MONOTONIC, &iter_start);

        printf("\033[1;32mITERATION %d\033[0m\n", k);
        // alpha = dot(r, z) / dot(p, mat_vec(A, p))
        printf("ALPHA CALCULATE\n");
        double alpha = alpha_calculate(solver, &r_buffer, &z_buffer, &direction_buffer, &temp, &r_dot_z);
        printf("\n\tAlpha = %g\n", alpha);

        // Update the solution vector x = x + alpha * p
        printf("\nUPDATE X\n");
        cl_event upd_x = update_x(solver, &direction_buffer, alpha, length);
        // USELESS WAIT FOR EVENT
        // clWaitForEvents(1, &upd_x);
        // double updx_t = profiling_event(upd_x);
        // printf("%-40s %-6.3f ms\n", "\tupdate_x kernel:", updx_t);
        // clReleaseEvent(upd_x);

        // Calculate the new residue r_(k+1) = r - alpha * mat_vec(A, p)
        // Calculate the new preconditioned residue z_(k+1) = D^(-1) * r_(k+1)
        printf("\nUPDATE R AND Z\n");
        cl_event next_r_z = update_r_and_z(solver, &r_buffer, &temp.Ap, &diagonal_buffer, &r_next_buffer, &z_next_buffer, alpha, length);
        clWaitForEvents(1, &next_r_z);
        double nextrz_t = profiling_event(next_r_z);
        printf("%-40s %-6.3f ms\n", "\tupdate_r_and_z kernel:", nextrz_t);
        clReleaseEvent(next_r_z);
        
        // printf("\nCALCULATE ||r_(k+1)||^2\n");    
        // Calculate the norm of the new residue ||r_(k+1)||^2
        r_norm = dot_product_handler(solver, &r_next_buffer, &r_next_buffer, length);
        printf("\n\tResidue norm = %g\n", r_norm);

        // beta = dot(r_(k+1), z_(k+1)) / dot(r, z)
        printf("\nBETA CALCULATE\n");
        printf("\t(r_(k+1) · z_(k+1))\n");
        double nextr_dot_nextz = dot_product_handler(solver, &r_next_buffer, &z_next_buffer, solver->size); 
        double beta = nextr_dot_nextz / r_dot_z;
        printf("\n\tBeta = %g\n", beta);

        // Update the search direction p = z_(k+1) + beta * p
        printf("\nUPDATE P\n");
        cl_event upd_p = update_p(solver, &direction_buffer, &z_next_buffer, beta, length);
        clWaitForEvents(1, &upd_p);
        double updp_t = profiling_event(upd_p);
        printf("%-40s %-6.3f ms\n", "\tupdate_p kernel:", updp_t);
        clReleaseEvent(upd_p);

        // Swap the buffers for the next iteration
        cl_mem tmp;
        tmp = r_buffer;
        r_buffer = r_next_buffer;
        r_next_buffer = tmp;

        tmp = z_buffer;
        z_buffer = z_next_buffer;
        z_next_buffer = tmp;

        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        double iter_elapsed = (iter_end.tv_sec - iter_start.tv_sec) + (iter_end.tv_nsec - iter_start.tv_nsec) / 1e9;
        printf("\nIteration %d time: %.3f s\n", k, iter_elapsed);

        printf("\n");
        k++;

    } while(r_norm > epsilon && k < max_iter);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\nX (snippet): ");
    print_buffer(cl, cl->x_buffer, length * sizeof(double), 20);

    clReleaseMemObject(diagonal_buffer);
    clReleaseMemObject(r_buffer);
    clReleaseMemObject(z_buffer);
    clReleaseMemObject(direction_buffer);
    clReleaseMemObject(r_next_buffer);
    clReleaseMemObject(z_next_buffer);
    free_temporarybuffers(&temp);

    printf("\nConjugate Gradient converged after %d iterations with norm %g\n", k, r_norm);
    printf("Total time: %.3f s\n", elapsed);
    return;
}

double alpha_calculate(Solver* solver, cl_mem *r, cl_mem *z, cl_mem *p, TemporaryBuffers* temp, double *r_dot_z) {
    cl_int err;
    OpenCLContext *cl = &solver->cl;
    int length = solver->size;

    // r * z
    printf("\t(r · z)\n");
    *r_dot_z = dot_product_handler(solver, r, z, length);

    // p * A * p
    printf("\n\t(p * A * p)\n");
    //cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, p, &temp->Ap);
    cl_event mat_vec_multiply_evt = compressed_matvec_mult(solver, p, &temp->Ap);

    //USELESS WAIT FOR EVENT
    clWaitForEvents(1, &mat_vec_multiply_evt);
    double t = profiling_event(mat_vec_multiply_evt);
    printf("%-40s %-6.3f ms\n", "\tmat_vec_multiply kernel:", t);
    clReleaseEvent(mat_vec_multiply_evt);

    double denominator = dot_product_handler(solver, p, &temp->Ap, length);

    // printf("Alpha -> Numerator: %g, Denominator: %g\n", numerator, denominator);

    if (denominator == 0) {
        fprintf(stderr, "Denominator is zero, cannot compute alpha.\n");
        exit(EXIT_FAILURE);
    }

    return *r_dot_z / denominator;
}

cl_event partial_sum_reduction(Solver *solver, cl_mem* vec_in, cl_mem* vec_out, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, sizeof(cl_mem), vec_in);
    ocl_check(err, "clSetKernelArg failed for vec_in");
    arg++;

    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, sizeof(cl_mem), vec_out);
    ocl_check(err, "clSetKernelArg failed for vec_out");
    arg++;

    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, cl->lws * sizeof(double), NULL);
    ocl_check(err, "clSetKernelArg failed for local_memory");
    arg++;

    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.partial_sum_reduction, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");

    return event;
}

cl_event reduce_sum4_double4_sliding(Solver *solver, cl_mem* vec_in, cl_mem* vec_out, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, sizeof(cl_mem), vec_in);
    ocl_check(err, "clSetKernelArg failed for vec_in");
    arg++;

    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, sizeof(cl_mem), vec_out);
    ocl_check(err, "clSetKernelArg failed for vec_out");
    arg++;

    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, cl->lws * sizeof(double), NULL);
    ocl_check(err, "clSetKernelArg failed for local_memory");
    arg++;

    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length_vec4");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.partial_sum_reduction, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");

    return event;
}

cl_event dot_product(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(cl_mem), vec1);
    ocl_check(err, "clSetKernelArg failed for vec1");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(cl_mem), vec2);
    ocl_check(err, "clSetKernelArg failed for vec2");
    arg++;  

    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product, arg, cl->lws * sizeof(double), NULL);
    ocl_check(err, "clSetKernelArg failed for local_memory");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.dot_product, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");    

    return event;
}

cl_event dot_product_vec4(Solver *solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    if (length % 4 != 0) {
        fprintf(stderr, "Length must be multiple of 4 to use double4\n");
        exit(EXIT_FAILURE);
    }
    int length_vec4 = length / 4;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg, sizeof(cl_mem), vec1);
    ocl_check(err, "clSetKernelArg failed for vec1");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg, sizeof(cl_mem), vec2);
    ocl_check(err, "clSetKernelArg failed for vec2");
    arg++;  

    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.dot_product_vec4, arg, sizeof(int), &length_vec4);
    ocl_check(err, "clSetKernelArg failed for lenght (dot4)");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length_vec4, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.dot_product_vec4, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed");    

    return event;
}

cl_event update_x(Solver* solver, cl_mem* p, double alpha, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(cl_mem), &cl->x_buffer);
    ocl_check(err, "clSetKernelArg failed for x_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(cl_mem), p);
    ocl_check(err, "clSetKernelArg failed for p");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(double), &alpha);
    ocl_check(err, "clSetKernelArg failed for alpha");
    arg++;

    err = clSetKernelArg(cl->kernels.update_x, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.update_x, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for update_x");

    return event;
} 

cl_event update_p(Solver* solver, cl_mem* p, cl_mem* z, double beta, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(cl_mem), p);
    ocl_check(err, "clSetKernelArg failed for p");
    arg++;

    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(cl_mem), z);
    ocl_check(err, "clSetKernelArg failed for z");
    arg++;

    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(double), &beta);
    ocl_check(err, "clSetKernelArg failed for beta");
    arg++;

    err = clSetKernelArg(cl->kernels.update_p, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.update_p, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for update_p");

    return event;
} 

cl_event update_r_and_z(Solver* solver, cl_mem* r, cl_mem* Ap, cl_mem* precond, cl_mem* r_next, cl_mem* z_next, double alpha, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.update_r, arg, sizeof(cl_mem), r);
    ocl_check(err, "clSetKernelArg failed for r");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r, arg, sizeof(cl_mem), Ap);
    ocl_check(err, "clSetKernelArg failed for Ap");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r, arg, sizeof(cl_mem), precond);
    ocl_check(err, "clSetKernelArg failed for precond");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r, arg, sizeof(cl_mem), r_next);
    ocl_check(err, "clSetKernelArg failed for r_next");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r, arg, sizeof(cl_mem), z_next);
    ocl_check(err, "clSetKernelArg failed for z_next");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r, arg, sizeof(double), &alpha);
    ocl_check(err, "clSetKernelArg failed for alpha");
    arg++;

    err = clSetKernelArg(cl->kernels.update_r, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for length");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.update_r, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for update_r");

    return event;
}

cl_event compressed_matvec_mult(Solver* solver, cl_mem* vec, cl_mem* result) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.compressed_matvec_mult, arg, sizeof(cl_mem), &cl->cbm_values_buffer);
    ocl_check(err, "clSetKernelArg failed for cbm_values_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.compressed_matvec_mult, arg, sizeof(cl_int), &cl->cbm_bandwidth);
    ocl_check(err, "clSetKernelArg failed for cbm_bandwidth");
    arg++;

    // err = clSetKernelArg(cl->kernels.compressed_matvec_mult, arg, sizeof(double) * cl->cbm_bandwidth, NULL);
    // ocl_check(err, "clSetKernelArg failed for local_memory");
    // arg++;

    err = clSetKernelArg(cl->kernels.compressed_matvec_mult, arg, sizeof(cl_mem), vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl->kernels.compressed_matvec_mult, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.compressed_matvec_mult, arg, sizeof(int), &solver->size);
    ocl_check(err, "clSetKernelArg failed for size");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = solver->size * cl->lws;
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.compressed_matvec_mult, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for compressed_matvec_mult");

    return event;
}

cl_event get_inverted_diagonal_compressed(Solver* solver, cl_mem* diagonal_buffer){
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.get_inverted_diagonal_compressed, arg, sizeof(cl_mem), &cl->cbm_values_buffer);
    ocl_check(err, "clSetKernelArg failed for cbm_values_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.get_inverted_diagonal_compressed, arg, sizeof(cl_int), &cl->cbm_bandwidth);
    ocl_check(err, "clSetKernelArg failed for cbm_bandwidth");
    arg++;

    err = clSetKernelArg(cl->kernels.get_inverted_diagonal_compressed, arg, sizeof(cl_mem), diagonal_buffer);
    ocl_check(err, "clSetKernelArg failed for diagonal_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.get_inverted_diagonal_compressed, arg, sizeof(int), &solver->size);
    ocl_check(err, "clSetKernelArg failed for size");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = solver->size * cl->lws;
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.get_inverted_diagonal_compressed, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for get_inverted_diagonal_compressed");

    return event;
}

void free_solver(Solver* solver) {
    OpenCLContext* cl = &solver->cl;

    // Release OpenCL resources
    clReleaseMemObject(cl->b_buffer);
    clReleaseMemObject(cl->x_buffer);
    // clReleaseMemObject(cl->csr_row_ptr_buffer);
    // clReleaseMemObject(cl->csr_col_ind_buffer);
    // clReleaseMemObject(cl->csr_values_buffer);

    clReleaseKernel(cl->kernels.partial_sum_reduction);
    clReleaseKernel(cl->kernels.dot_product);
    clReleaseKernel(cl->kernels.mat_vec_multiply);
    clReleaseKernel(cl->kernels.mult_vectors);
    clReleaseKernel(cl->kernels.sum_vectors);
    clReleaseKernel(cl->kernels.scale_vector);
    clReleaseKernel(cl->kernels.get_inverted_diagonal);
    clReleaseKernel(cl->kernels.update_x);
    clReleaseKernel(cl->kernels.update_p);
    clReleaseKernel(cl->kernels.update_r);
    clReleaseKernel(cl->kernels.compressed_matvec_mult);
    clReleaseKernel(cl->kernels.get_inverted_diagonal_compressed);


    clReleaseProgram(cl->prog);
    clReleaseCommandQueue(cl->q);
    clReleaseContext(cl->ctx);

    // Free the solver structure
    free(solver->x);
}   

void print_buffer(OpenCLContext *cl, cl_mem buf, size_t size, int n) {
    double* temp = malloc(size);
    if (!temp) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    clFinish(cl->q);  // Assicura che tutte le operazioni siano terminate
    cl_int err = clEnqueueReadBuffer(cl->q, buf, CL_TRUE, 0, size, temp, 0, NULL, NULL);
    ocl_check(err, "print_buffer read");

    for (int i = 0; i < n; i++) {
        printf(" %.2f ", temp[i]);
    }
    printf("\n");
    free(temp);
}

double profiling_event(cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    return (end - start) / 1e6; // Convert to milliseconds
}

double dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    //Partial dot product
    cl_mem partial_dot_product = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(double) * length, NULL, &err);
    cl_event dot_event = dot_product(solver, vec1, vec2, &partial_dot_product, length);
    clWaitForEvents(1, &dot_event);

    // Profiling
    double t = profiling_event(dot_event);
    printf("%-40s %-6.3f ms\n", "\tdot_product kernel:", t);

    clReleaseEvent(dot_event);

    size_t nels_vec4 = (solver->size + 3) / 4;
    size_t num_groups = round_div_up(nels_vec4, cl->lws);

    cl_mem temp_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, num_groups * sizeof(double), NULL, NULL);

    cl_mem *in_buf = &partial_dot_product;
    cl_mem *out_buf = &temp_buffer;

    double total_time = 0.0;
    t = 0.0;
    while(num_groups > 1) {
        // Perform partial sum reduction
        cl_event partial_sum_evt = reduce_sum4_double4_sliding(solver, in_buf, out_buf, num_groups);
        clWaitForEvents(1, &partial_sum_evt);

        // Profiling
        t = profiling_event(partial_sum_evt);
        total_time += t;

        clReleaseEvent(partial_sum_evt);

        // Swap buffers
        cl_mem temp = *in_buf;
        *in_buf = *out_buf;
        *out_buf = temp;

        num_groups = round_div_up(num_groups, 4 * cl->lws);
    }

    printf("%-40s %-6.3f ms\n", "\treduce_sum4_double4_sliding kernel (total):", total_time);

    double final_result;
    clEnqueueReadBuffer(cl->q, *in_buf, CL_TRUE, 0, sizeof(double), &final_result, 0, NULL, NULL);
    
    clReleaseMemObject(partial_dot_product);
    clReleaseMemObject(temp_buffer);

    return final_result;
}

cl_event sum_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(cl_mem), vec1);
    ocl_check(err, "clSetKernelArg failed for vec1");
    arg++;      

    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(cl_mem), vec2);
    ocl_check(err, "clSetKernelArg failed for vec2");
    arg++;  

    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.sum_vectors, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel 
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.sum_vectors, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for sum_vectors");    

    return event;
}

cl_event scale_vector(Solver* solver, cl_mem* vec, double scale, cl_mem* result, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(cl_mem), vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(double), &scale);
    ocl_check(err, "clSetKernelArg failed for scale");
    arg++;

    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel 
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.scale_vector, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for scale_vector");    

    return event;
}

cl_event mult_vectors(Solver* solver, cl_mem* vec1, cl_mem* vec2, cl_mem* result, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.mult_vectors, arg, sizeof(cl_mem), vec1);
    ocl_check(err, "clSetKernelArg failed for vec1");
    arg++;      

    err = clSetKernelArg(cl->kernels.mult_vectors, arg, sizeof(cl_mem), vec2);
    ocl_check(err, "clSetKernelArg failed for vec2");
    arg++;  

    err = clSetKernelArg(cl->kernels.mult_vectors, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    err = clSetKernelArg(cl->kernels.mult_vectors, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel 
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.mult_vectors, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for mult_vectors");    

    return event;
}

// DEPRECATED (IDK YET)
// double dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int length) {
//     OpenCLContext *cl = &solver->cl;
//     cl_int err;

//     //Partial dot product
//     size_t num_groups = round_div_up(solver->size, cl->lws);
//     cl_mem partial_dot_product = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(double) * num_groups, NULL, &err);
//     cl_event dot_event = dot_product(solver, vec1, vec2, &partial_dot_product, length);
//     clWaitForEvents(1, &dot_event);

//     // Profiling
//     double t = profiling_event(dot_event);
//     printf("%-40s %-6.3f ms\n", "\tdot_product kernel:", t);

//     clReleaseEvent(dot_event);

//     cl_mem temp_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, num_groups * sizeof(double), NULL, NULL);

//     cl_mem *in_buf = &partial_dot_product;
//     cl_mem *out_buf = &temp_buffer;

//     double total_time = 0.0;
//     t = 0.0;
//     while(num_groups > 1) {
//         // Perform partial sum reduction
//         cl_event partial_sum_evt = partial_sum_reduction(solver, in_buf, out_buf, num_groups);
//         clWaitForEvents(1, &partial_sum_evt);

//         // Profiling
//         t = profiling_event(partial_sum_evt);
//         total_time += t;

//         clReleaseEvent(partial_sum_evt);

//         // Swap buffers
//         cl_mem temp = *in_buf;
//         *in_buf = *out_buf;
//         *out_buf = temp;

//         num_groups = round_div_up(num_groups, cl->lws);
//     }

//     printf("%-40s %-6.3f ms\n", "\tpartial_sum_reduction kernel (total):", total_time);

//     double final_result;
//     clEnqueueReadBuffer(cl->q, *in_buf, CL_TRUE, 0, sizeof(double), &final_result, 0, NULL, NULL);
    
//     clReleaseMemObject(partial_dot_product);
//     clReleaseMemObject(temp_buffer);

//     return final_result;
// }
