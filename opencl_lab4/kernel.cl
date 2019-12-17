__kernel void iterate(__global float *A, __global float *b, __global float *x0, 
    __global float *x1, __global float *norm, const uint size)
{
	const size_t i = get_global_id(0);
	if (i >= size) {
		return;
	}
	float acc = 0.0f;
	for (size_t j = 0; j < size; j++) {
		acc += A[j * size + i] * x0[j] * (float)(i != j);
	}
	x1[i] = (b[i] - acc) / A[i * size + i];
	norm[i] = x0[i] - x1[i];
}