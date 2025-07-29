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

Solver setup_solver(int size, CSRMatrix A, float *b, float *initial_x) {
    Solver solver;
    solver.size = size;
    solver.A = A;
    solver.b = b;
    solver.x = initial_x;

    if (!solver.x) {
        solver.x = malloc(size * sizeof(float));
        if (!solver.x) {
            fprintf(stderr, "Failed to allocate memory for x vector\n");
            exit(1);
        }
        for (int i = 0; i < size; i++) {
            solver.x[i] = 0.0f;  // Initialize x to zero
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
        solver.size * sizeof(float), solver.b, &err);
	ocl_check(err, "clCreateBuffer failed for b_buffer");

    cl.x_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        solver.size * sizeof(float), solver.x, &err);
    ocl_check(err, "clCreateBuffer failed for x_buffer");

    //CSR Matrix buffers
    cl.csr_row_ptr_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (solver.A.rows + 1) * sizeof(int), solver.A.row_ptr, &err);
    ocl_check(err, "clCreateBuffer failed for csr_row_ptr_buffer");

    cl.csr_col_ind_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver.A.nnz * sizeof(int), solver.A.col_ind, &err);
    ocl_check(err, "clCreateBuffer failed for csr_col_ind_buffer");

    cl.csr_values_buffer = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        solver.A.nnz * sizeof(float), solver.A.values, &err);
    ocl_check(err, "clCreateBuffer failed for csr_values_buffer");
    
    // Create kernels
    cl.kernels.partial_sum_reduction = clCreateKernel(cl.prog, "partial_sum_reduction", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.dot_product = clCreateKernel(cl.prog, "dot_product", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.mat_vec_multiply = clCreateKernel(cl.prog, "mat_vec_multiply", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.mult_vectors = clCreateKernel(cl.prog, "mult_vectors", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.sum_vectors = clCreateKernel(cl.prog, "sum_vectors", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.scale_vector = clCreateKernel(cl.prog, "scale_vector", &err);  
    ocl_check(err, "clCreateKernel failed");

    cl.kernels.get_inverted_diagonal = clCreateKernel(cl.prog, "get_inverted_diagonal", &err);
    ocl_check(err, "clCreateKernel failed");

    cl.lws = 64;

    return cl;
}

TemporaryBuffers init_buffers(Solver* solver, int length) {
    cl_int err;
    TemporaryBuffers temp;

    temp.Ap = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for temp.Ap");

    temp.x = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for temp.x");

    temp.p = clCreateBuffer(solver->cl.ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
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
    int length = solver->A.rows;
    float r_norm;   // Residue norm
    float epsilon = 1e-5f;  // Convergence threshold
    int max_iter = length;   // Maximum iterations
    float alpha;    // Step size along the search direction
    int k = 0;  // Iteration counter

    TemporaryBuffers temp = init_buffers(solver, length);

    // printf("x: ");
    // print_buffer(cl, cl->x_buffer, length * sizeof(float), 20);
    // printf("b: ");
    // print_buffer(cl, cl->b_buffer, length * sizeof(float), 20);
    // printf("\n");


    /*
    STEP ZERO: Precondition with Jacobi by extracting the diagonal of the matrix A
    and inverting it
    */
    cl_mem diagonal_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err); 
    ocl_check(err, "clCreateBuffer failed for diagonal_buffer");

    cl_event inverted_diagonal = get_inverted_diagonal(solver, &diagonal_buffer, length);
    clWaitForEvents(1, &inverted_diagonal);
    clReleaseEvent(inverted_diagonal);

    /*
    STEP ONE: Calculate initial residue r = b - Ax
    */
    cl_mem r_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_buffer");

    cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, &cl->x_buffer, &r_buffer);
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
    cl_mem z_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for z_buffer");

    cl_event preconditioned_residue_evt = mult_vectors(solver, &diagonal_buffer, &r_buffer, &z_buffer, length);
    ocl_check(err, "mult_vectors failed for preconditioned_residue");
    clWaitForEvents(1, &preconditioned_residue_evt);
    clReleaseEvent(preconditioned_residue_evt);

    /*
    STEP THREE: set first search direction p = z 
    */
    cl_mem direction_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
	ocl_check(err, "clCreateBuffer failed for direction_buffer");

    cl_event copy_buffer_evt;
    err = clEnqueueCopyBuffer(cl->q, z_buffer, direction_buffer, 0, 0, 
        length * sizeof(float), 0, NULL, &copy_buffer_evt);
    ocl_check(err, "clEnqueueCopyBuffer failed for direction_buffer");

    clWaitForEvents(1, &copy_buffer_evt);
    clReleaseEvent(copy_buffer_evt);

    /*
    STEP FOUR: main loop of the Conjugate Gradient algorithm
    */

    //DEBUG 
    // printf("\nInitial x: ");
    // print_buffer(cl, cl->x_buffer, length * sizeof(float), 20);
    // cl_mem temp = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    // cl_event mat_vec_evt = mat_vec_multiply(solver, &cl->x_buffer, &temp);
    // clWaitForEvents(1, &mat_vec_evt);   
    // clReleaseEvent(mat_vec_evt);
    // printf("\nAx: ");
    // print_buffer(cl, temp, length * sizeof(float), 50);
    // printf("\nDiagonal: ");
    // print_buffer(cl, diagonal_buffer, length * sizeof(float), 50);
    // printf("\nInitial r: ");
    // print_buffer(cl, r_buffer, length * sizeof(float), 20);
    // printf("\nInitial z: ");
    // print_buffer(cl, z_buffer, length * sizeof(float), 20);
    

    // Allocate buffers for the next iteration
    cl_mem r_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_next_buffer");

    cl_mem z_next_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, length * sizeof(float), NULL, &err);
    ocl_check(err, "clCreateBuffer failed for r_next_buffer");


    // profiling 
    struct timespec start, end;
    struct timespec iter_start, iter_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    do {
        clock_gettime(CLOCK_MONOTONIC, &iter_start);

        printf("\033[1;32mITERATION %d\033[0m\n", k);
        // alpha = dot(r, z) / dot(p, mat_vec(A, p))
        printf("ALPHA CALCULATE\n");
        alpha = alpha_calculate(solver, &r_buffer, &z_buffer, &direction_buffer, &temp);
        printf("\n\tAlpha = %g\n", alpha);

        // Update the solution vector x = x + alpha * p
        printf("\nUPDATE X\n");
        update_x(solver, &direction_buffer, alpha, length, &temp);

        // Calculate the new residue r_(k+1) = r - alpha * mat_vec(A, p)
        printf("\nUPDATE R\n");
        update_r(solver, &r_buffer, &direction_buffer, &r_next_buffer, alpha, length);

        // Calculate the norm of the new residue ||r_(k+1)||
        r_norm = sqrt(dot_product_handler(solver, &r_next_buffer, &r_next_buffer, length));
        printf("\n\tResidue norm = %g\n", r_norm);

        // Calculate the new preconditioned residue z_(k+1) = D^(-1) * r_(k+1)
        printf("\nUPDATE Z\n\t(D^(-1) * r_(k+1))\n");
        cl_event z_evt = mult_vectors(solver, &diagonal_buffer, &r_next_buffer, &z_next_buffer, length);
        clWaitForEvents(1, &z_evt);

        // profiling
        double t = profiling_event(z_evt);
        printf("%-40s %-6.3f ms\n", "\tmult_vectors kernel:", t);
        clReleaseEvent(z_evt);

        // beta = dot(r_(k+1), z_(k+1)) / dot(r, z)
        printf("\nBETA CALCULATE\n");
        float beta = beta_calculate(solver, &r_next_buffer, &z_next_buffer, &r_buffer, &z_buffer);  
        printf("\n\tBeta = %g\n", beta);

        // Update the search direction p = z_(k+1) + beta * p
        printf("\nUPDATE P\n");
        update_p(solver, &z_next_buffer, &direction_buffer, beta, length, &temp);

        // Update the residue for the next iteration
        printf("\nCOPY R BUFFER\n");
        cl_event rcopy_evt;
        err = clEnqueueCopyBuffer(cl->q, r_next_buffer, r_buffer, 0, 0, length * sizeof(float), 0, NULL, &rcopy_evt);
        clWaitForEvents(1, &rcopy_evt);

        // profiling
        t = profiling_event(rcopy_evt);
        printf("%-40s %-6.3f ms\n", "\tclEnqueueCopyBuffer:", t);
        clReleaseEvent(rcopy_evt);
        ocl_check(err, "clEnqueueCopyBuffer failed for r_buffer");

        // Update the preconditioned residue for the next iteration
        printf("\nCOPY Z BUFFER\n");
        cl_event zcopy_evt;
        err = clEnqueueCopyBuffer(cl->q, z_next_buffer, z_buffer, 0, 0, length * sizeof(float), 0, NULL, &zcopy_evt);
        clWaitForEvents(1, &zcopy_evt);

        // profiling
        t = profiling_event(zcopy_evt);
        printf("%-40s %-6.3f ms\n", "\tclEnqueueCopyBuffer:", t);
        clReleaseEvent(zcopy_evt);
        ocl_check(err, "clEnqueueCopyBuffer failed for z_buffer");

        // printf("\nX: ");
        // print_buffer(cl, cl->x_buffer, length * sizeof(float), 20);
        clock_gettime(CLOCK_MONOTONIC, &iter_end);
        double iter_elapsed = (iter_end.tv_sec - iter_start.tv_sec) + (iter_end.tv_nsec - iter_start.tv_nsec) / 1e9;
        printf("\nIteration %d time: %.3f s\n", k, iter_elapsed);

        printf("\n");
        k++;

    } while(r_norm > epsilon && k < max_iter);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\nX (snippet): ");
    print_buffer(cl, cl->x_buffer, length * sizeof(float), 20);

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

float alpha_calculate(Solver* solver, cl_mem *r, cl_mem *z, cl_mem *p, TemporaryBuffers* temp) {
    cl_int err;
    OpenCLContext *cl = &solver->cl;
    int length = solver->size;

    // r * z
    printf("\t(r · z)\n");
    float numerator = dot_product_handler(solver, r, z, length);

    // p * A * p
    printf("\n\t(p * A * p)\n");
    cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, p, &temp->Ap);
    clWaitForEvents(1, &mat_vec_multiply_evt);

    // profiling
    double t = profiling_event(mat_vec_multiply_evt);
    printf("%-40s %-6.3f ms\n", "\tmat_vec_multiply kernel:", t);
    clReleaseEvent(mat_vec_multiply_evt);

    float denominator = dot_product_handler(solver, p, &temp->Ap, length);

    // printf("Alpha -> Numerator: %g, Denominator: %g\n", numerator, denominator);

    if (denominator == 0) {
        fprintf(stderr, "Denominator is zero, cannot compute alpha.\n");
        exit(EXIT_FAILURE);
    }

    return numerator / denominator;
}

float beta_calculate(Solver* solver, cl_mem *r_next, cl_mem *z_next, cl_mem *r, cl_mem *z) {
    printf("\t(r_(k+1) · z_(k+1))\n");
    float nextr_dot_nextz = dot_product_handler(solver, r_next, z_next, solver->size);
    printf("\t(r · z)\n");
    float r_dot_z = dot_product_handler(solver, r, z, solver->size);
    return nextr_dot_nextz / r_dot_z;
}

void update_x(Solver *solver, cl_mem* p, float alpha, int length, TemporaryBuffers* temp) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    // alpha * p
    printf("\t(alpha * p)\n");
    cl_event scale_vector_evt = scale_vector(solver, p, alpha, &temp->x, length);
    clWaitForEvents(1, &scale_vector_evt);

    // profiling
    double t = profiling_event(scale_vector_evt);
    printf("%-40s %-6.3f ms\n", "\tscale_vector kernel:", t);
    clReleaseEvent(scale_vector_evt);

    // Add the scaled p to x
    printf("\t(x + alpha * p)\n");
    cl_event sum_vectors_evt = sum_vectors(solver, &cl->x_buffer, &temp->x, &cl->x_buffer, length);
    clWaitForEvents(1, &sum_vectors_evt);

    // profiling
    t = profiling_event(sum_vectors_evt);
    printf("%-40s %-6.3f ms\n", "\tsum_vectors kernel:", t);    
    clReleaseEvent(sum_vectors_evt);
}   

void update_r(Solver *solver, cl_mem* r, cl_mem* p, cl_mem* r_next, float alpha, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    // A * p
    printf("\t(A * p)\n");
    cl_event mat_vec_multiply_evt = mat_vec_multiply(solver, p, r_next);
    clWaitForEvents(1, &mat_vec_multiply_evt);

    // profiling    
    double t = profiling_event(mat_vec_multiply_evt);
    printf("%-40s %-6.3f ms\n", "\tmat_vec_multiply kernel:", t);
    clReleaseEvent(mat_vec_multiply_evt);

    // Scale the result by -alpha
    printf("\t(-alpha * A * p)\n");
    cl_event scale_vector_evt = scale_vector(solver, r_next, -alpha, r_next, length);
    clWaitForEvents(1, &scale_vector_evt);

    // profiling
    t = profiling_event(scale_vector_evt);
    printf("%-40s %-6.3f ms\n", "\tscale_vector kernel:", t);
    clReleaseEvent(scale_vector_evt);

    // Add the scaled result to the original residue
    printf("\t(r - alpha * A * p)\n");
    cl_event sum_vectors_evt = sum_vectors(solver, r, r_next, r_next, length);
    clWaitForEvents(1, &sum_vectors_evt);

    // profiling
    t = profiling_event(sum_vectors_evt);
    printf("%-40s %-6.3f ms\n", "\tsum_vectors kernel", t);
    clReleaseEvent(sum_vectors_evt);
}

void update_p(Solver *solver, cl_mem* z, cl_mem* p, float beta, int length, TemporaryBuffers* temp) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    
    // Scale the previous search direction p by beta
    printf("\t(beta * p)\n");
    cl_event scale_vector_evt = scale_vector(solver, p, beta, &temp->p, length);
    clWaitForEvents(1, &scale_vector_evt);

    // profiling
    double t = profiling_event(scale_vector_evt);
    printf("%-40s %-6.3f ms\n", "\tscale_vector kernel:", t);
    clReleaseEvent(scale_vector_evt);

    // Add the new residue r to the scaled search direction
    printf("\t(z + beta * p)\n");
    cl_event sum_vectors_evt = sum_vectors(solver, z, &temp->p, p, length);
    clWaitForEvents(1, &sum_vectors_evt);

    // profiling
    t = profiling_event(sum_vectors_evt);
    printf("%-40s %-6.3f ms\n", "\tsum_vectors kernel:", t);
    clReleaseEvent(sum_vectors_evt);
}

float dot_product_handler(Solver *solver, cl_mem *vec1, cl_mem *vec2, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;

    //Partial dot product
    size_t num_groups = round_div_up(solver->size, cl->lws);
    cl_mem partial_dot_product = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, sizeof(float) * num_groups, NULL, &err);
    cl_event dot_event = dot_product(solver, vec1, vec2, &partial_dot_product, length);
    clWaitForEvents(1, &dot_event);

    // Profiling
    double t = profiling_event(dot_event);
    printf("%-40s %-6.3f ms\n", "\tdot_product kernel:", t);

    clReleaseEvent(dot_event);

    cl_mem temp_buffer = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, num_groups * sizeof(float), NULL, NULL);

    cl_mem *in_buf = &partial_dot_product;
    cl_mem *out_buf = &temp_buffer;

    double total_time = 0.0;
    t = 0.0;
    while(num_groups > 1) {
        // Perform partial sum reduction
        cl_event partial_sum_evt = partial_sum_reduction(solver, in_buf, out_buf, num_groups);
        clWaitForEvents(1, &partial_sum_evt);

        // Profiling
        t = profiling_event(partial_sum_evt);
        total_time += t;

        clReleaseEvent(partial_sum_evt);

        // Swap buffers
        cl_mem temp = *in_buf;
        *in_buf = *out_buf;
        *out_buf = temp;

        num_groups = round_div_up(num_groups, cl->lws);
    }

    printf("%-40s %-6.3f ms\n", "\tpartial_sum_reduction kernel (total):", total_time);

    float final_result;
    clEnqueueReadBuffer(cl->q, *in_buf, CL_TRUE, 0, sizeof(float), &final_result, 0, NULL, NULL);
    
    clReleaseMemObject(partial_dot_product);
    clReleaseMemObject(temp_buffer);

    return final_result;
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

    err = clSetKernelArg(cl->kernels.partial_sum_reduction, arg, cl->lws * sizeof(float), NULL);
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

    err = clSetKernelArg(cl->kernels.dot_product, arg, cl->lws * sizeof(float), NULL);
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

cl_event get_inverted_diagonal(Solver* solver, cl_mem* diagonal, int length) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.get_inverted_diagonal, arg, sizeof(cl_mem), &cl->csr_values_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_values_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.get_inverted_diagonal, arg, sizeof(cl_mem), &cl->csr_col_ind_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_col_ind_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.get_inverted_diagonal, arg, sizeof(cl_mem), &cl->csr_row_ptr_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_row_ptr_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.get_inverted_diagonal, arg, sizeof(cl_mem), diagonal);
    ocl_check(err, "clSetKernelArg failed for diagonal");
    arg++;

    err = clSetKernelArg(cl->kernels.get_inverted_diagonal, arg, sizeof(int), &length);
    ocl_check(err, "clSetKernelArg failed for lenght");
    arg++;

    // Launch the kernel
    cl_event event;
    size_t gws = round_mul_up(length, cl->lws);
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.get_inverted_diagonal, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for get_inverted_diagonal");

    return event;
}

cl_event mat_vec_multiply(Solver *solver, cl_mem* vec, cl_mem* result) {
    OpenCLContext *cl = &solver->cl;
    cl_int err;
    cl_int arg = 0;

    // Set kernel arguments
    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &cl->csr_values_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_values_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &cl->csr_col_ind_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_col_ind_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), &cl->csr_row_ptr_buffer);
    ocl_check(err, "clSetKernelArg failed for csr_row_ptr_buffer");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), vec);
    ocl_check(err, "clSetKernelArg failed for vec");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(cl_mem), result);
    ocl_check(err, "clSetKernelArg failed for result");
    arg++;

    size_t local_mem_size = sizeof(float) * cl->lws;
    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, local_mem_size, NULL);
    ocl_check(err, "clSetKernelArg failed for local memory");
    arg++;

    err = clSetKernelArg(cl->kernels.mat_vec_multiply, arg, sizeof(int), &solver->size);
    ocl_check(err, "clSetKernelArg failed for size");
    arg++;

    //Launch the kernel
    cl_event event;
    size_t gws = solver->size * cl->lws;
    err = clEnqueueNDRangeKernel(cl->q, cl->kernels.mat_vec_multiply, 1, NULL,
            &gws, &cl->lws, 0, NULL, &event);
    ocl_check(err, "clEnqueueNDRangeKernel failed for mat_vec_multiply");

    return event;
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

cl_event scale_vector(Solver* solver, cl_mem* vec, float scale, cl_mem* result, int length) {
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

    err = clSetKernelArg(cl->kernels.scale_vector, arg, sizeof(float), &scale);
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

void free_solver(Solver* solver) {
    OpenCLContext* cl = &solver->cl;

    // Release OpenCL resources
    clReleaseMemObject(cl->b_buffer);
    clReleaseMemObject(cl->x_buffer);
    clReleaseMemObject(cl->csr_row_ptr_buffer);
    clReleaseMemObject(cl->csr_col_ind_buffer);
    clReleaseMemObject(cl->csr_values_buffer);

    clReleaseKernel(cl->kernels.partial_sum_reduction);
    clReleaseKernel(cl->kernels.dot_product);
    clReleaseKernel(cl->kernels.mat_vec_multiply);
    clReleaseKernel(cl->kernels.mult_vectors);
    clReleaseKernel(cl->kernels.sum_vectors);
    clReleaseKernel(cl->kernels.scale_vector);
    clReleaseKernel(cl->kernels.get_inverted_diagonal);

    clReleaseProgram(cl->prog);
    clReleaseCommandQueue(cl->q);
    clReleaseContext(cl->ctx);

    // Free the solver structure
    free(solver->x);
}   

void print_buffer(OpenCLContext *cl, cl_mem buf, size_t size, int n) {
    float* temp = malloc(size);
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