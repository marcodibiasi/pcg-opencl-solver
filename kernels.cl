__kernel void partial_sum_reduction(
    __global const double* vec_in,
    __global double* vec_out,

    __local double* local_memory,

    const int length
)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    if(global_id < length)
        local_memory[local_id] = vec_in[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = local_size / 2; stride > 0; stride /= 2){
        if(local_id < stride)
            local_memory[local_id] = local_memory[local_id] + local_memory[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0)
        vec_out[group_id] = local_memory[0];
}


__kernel void reduce_sum4_double4_sliding(
    __global const double4* restrict in,
    __global double* restrict out,
    __local double* restrict local_memory,
    const int n_els  // total number
) {
    const int li = get_local_id(0);
    const int lws = get_local_size(0);
    const int group_id = get_group_id(0);
    const int num_groups = get_num_groups(0);

    const int nels_vec = (n_els + 3) / 4; 
    const int vecs_per_group_min = (nels_vec - 1) / num_groups + 1;
    const int vecs_per_group = lws * ((vecs_per_group_min - 1) / lws + 1);

    int gi = group_id * vecs_per_group + li;
    const int end = (group_id + 1) * vecs_per_group;

    double acc = 0.0;

    while (gi < end) {
        double val = 0.0;
        if (gi < nels_vec) {
            double4 d = in[gi];
            int base = gi * 4;
            if (base + 3 < n_els)
                val = d.x + d.y + d.z + d.w;
            else {
                if (base + 0 < n_els) val += d.x;
                if (base + 1 < n_els) val += d.y;
                if (base + 2 < n_els) val += d.z;
                if (base + 3 < n_els) val += d.w;
            }
        }

        local_memory[li] = val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = lws / 2; stride > 0; stride >>= 1) {
            if (li < stride)
                local_memory[li] += local_memory[li + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (li == 0)
            acc += local_memory[0];

        gi += lws;
    }

    if (li == 0)
        out[group_id] = acc;
}


__kernel void dot_product(
    __global const double* a,
    __global const double* b,

    __global double* result,

    __local double* local_memory,
    const int length
)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    double local_product = 0.0;
    if(global_id < length)
        local_product = a[global_id] * b[global_id];

    local_memory[local_id] = local_product;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = local_size / 2; stride > 0; stride /= 2){
        if(local_id < stride)
            local_memory[local_id] = local_memory[local_id] + local_memory[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0)
        result[group_id] = local_memory[0];

}

__kernel void dot_product_vec4(
    __global const double4* a,
    __global const double4* b,
    __global double* result,
    const int length 
){
    int global_id = get_global_id(0);
    if(global_id >= length) return;

    double local_product = 0.0;
    double4 a_val = a[global_id];
    double4 b_val = b[global_id];

    local_product += (a_val.x * b_val.x) + (a_val.y * b_val.y) +
        (a_val.z * b_val.z) + (a_val.w * b_val.w);

    result[global_id] = local_product;
}


__kernel void mat_vec_multiply(

    //CSR Matrix format
    __global const double* values,
    __global const int* col_ind,
    __global const int* row_ptr,

    __global const double* vec_in,
    __global double* vec_out,

    __local double* local_memory,

    const int rows
)
{
    int row = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    if(row >= rows) return;

    // Get the non-zero values for each row
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    int row_lenght = row_end - row_start;

    double sum = 0.0;

    for (int idx = local_id; idx < row_lenght; idx += local_size) {
        int val_idx = row_start + idx;  // Get the CSR Matrix value of non-zero element
        sum += values[val_idx] * vec_in[col_ind[val_idx]];
    }

    local_memory[local_id] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride)
            local_memory[local_id] += local_memory[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        vec_out[row] = local_memory[0];
}


__kernel void compressed_matvec_mult(

    //CB Matrix format
    __global const double* values,
    const int bandwidth,

    __global const double* vec_in,
    __global double* vec_out,

    const int rows
)
{
    int global_id = get_global_id(0);
    if (global_id >= rows) return;

    int half_bw = bandwidth / 2;
    double sum = 0.0;

    for (int d = -half_bw; d <= half_bw; d++) {
        int j = global_id + d;
        if (j >= 0 && j < rows) {
            int val_index = d + half_bw;
            sum += values[val_index] * vec_in[j];
        }
    }

    vec_out[global_id] = sum;
}


__kernel void mult_vectors(
    __global const double* vec1,
    __global const double* vec2,
    __global double* result,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        result[global_id] = vec1[global_id] * vec2[global_id];
}


__kernel void sum_vectors(
    __global const double* vec1,
    __global const double* vec2,
    __global double* result,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        result[global_id] = vec1[global_id] + vec2[global_id];
}


__kernel void scale_vector(
    __global const double* vec_in,
    __global double* vec_out,
    const double scale,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        vec_out[global_id] = vec_in[global_id] * scale;
}


__kernel void get_inverted_diagonal(
    __global const double* values,
    __global const int* col_ind,
    __global const int* row_ptr,
    __global double* diagonal,
    const int rows
)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int idx = row_start; idx < row_end; ++idx) {
        if (col_ind[idx] == row) {
            diagonal[row] = 1.0 / values[idx];
            return;
        }
    }
}

__kernel void get_inverted_diagonal_compressed(
    __global const double* values,
    const int bandwidth,
    __global double* diagonal,
    const int rows
)
{
    int global_id = get_global_id(0);
    if (global_id >= rows) return;

    int half_bw = bandwidth / 2;
    int val_index = half_bw;  // The value at the center of the band

    diagonal[global_id] = 1.0 / values[val_index];
}

__kernel void update_x(
    __global double* x,
    __global const double* p,
    const double alpha,
    const int length
)
{
    int global_id = get_global_id(0);
    if (global_id < length) 
        x[global_id] += alpha * p[global_id];
}  

__kernel void update_r_and_z(
    __global const double* r,
    __global const double* Ap,
    __global const double* precond,
    __global double* r_next,  // at the start it contains A * p  
    __global double* z_next,
    const double alpha,
    const int length
)
{
    int global_id = get_global_id(0);
    double r_i;

    if (global_id < length) {
        r_i = r[global_id] - alpha * Ap[global_id];
        r_next[global_id] = r_i;
        z_next[global_id] = r_i * precond[global_id];
    }
}  

__kernel void update_p(
    __global double* p,
    __global const double* z,
    const double beta,
    const int length
)
{
    int global_id = get_global_id(0);
    if (global_id < length) {
        
        p[global_id] = z[global_id] + beta * p[global_id];
    }
} 