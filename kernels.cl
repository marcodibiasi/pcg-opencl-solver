__kernel void partial_sum_reduction(
    __global const float* vec_in,
    __global float* vec_out,

    __local float* local_memory,

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


__kernel void dot_product(
    __global const float* a,
    __global const float* b,

    __global float* result,

    __local float* local_memory,
    const int length
)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    float local_product = 0.0f;
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


__kernel void mat_vec_multiply(

    //CSR Matrix format
    __global const float* values,
    __global const int* col_ind,
    __global const int* row_ptr,

    __global const float* vec_in,
    __global float* vec_out,

    __local float* local_memory,

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

    float sum = 0.0f;

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


__kernel void mult_vectors(
    __global const float* vec1,
    __global const float* vec2,
    __global float* result,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        result[global_id] = vec1[global_id] * vec2[global_id];
}


__kernel void sum_vectors(
    __global const float* vec1,
    __global const float* vec2,
    __global float* result,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        result[global_id] = vec1[global_id] + vec2[global_id];
}


__kernel void scale_vector(
    __global const float* vec_in,
    __global float* vec_out,
    const float scale,
    const int lenght
)
{
    int global_id = get_global_id(0);
    if (global_id < lenght) 
        vec_out[global_id] = vec_in[global_id] * scale;
}


__kernel void get_inverted_diagonal(
    __global const float* values,
    __global const int* col_ind,
    __global const int* row_ptr,
    __global float* diagonal,
    const int rows
)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int idx = row_start; idx < row_end; ++idx) {
        if (col_ind[idx] == row) {
            diagonal[row] = 1.0f / values[idx];
            return;
        }
    }
}