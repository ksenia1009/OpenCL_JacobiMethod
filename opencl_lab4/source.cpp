#include <cstdlib>
#include <fstream>
#include <CL/cl.h>
#include <cassert>
#include <iostream>

const float epsilon = 0.0000001f; 
const size_t size = 2048;
const size_t iterations = 200;

std::string content = "__kernel void iterate(__global float *A, __global float *b, __global float *x0,\n" 
"__global float *x1, __global float *norm, const uint size)                                           \n"
"{                                                                                                    \n"
"    const size_t i = get_global_id(0);                                                               \n"
"    if (i >= size) {                                                                                 \n"
"        return;                                                                                      \n"
"    }                                                                                                \n"
"    float acc = 0.0f;                                                                                \n"
"    for (size_t j = 0; j < size; j++) {                                                              \n"
"        acc += A[j * size + i] * x0[j] * (float)(i != j);                                            \n"
"    }                                                                                                \n"
"    x1[i] = (b[i] - acc) / A[i * size + i];                                                          \n"
"    norm[i] = x0[i] - x1[i];                                                                         \n"
"}";

void fillMatrix(float* A, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		float acc = 0.0;
		for (size_t j = 0; j < size; ++j) {
			A[i * size + j] = rand() % 6 + 1;
			acc += A[i * size + j];
		}
		A[i * size + i] += acc * size * size;
	}
}

void fillVector(float* vector, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		vector[i] = rand() % 6 + 1;
	}
}

void printSystem(float* A, float* b, size_t size) {
	if (size <= 10) {
		std::cout << "System of equations:" << std::endl;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				std::cout << A[i * size + j] << "\t";
			}
			std::cout << "|\t" << b[i] << std::endl;
		}
		std::cout << std::endl;
	}
}

void printVector(float* vector, size_t size) {
	if (size <= 10) {
		std::cout << "Solution:" << std::endl;
		for (int i = 0; i < size; i++) {
			std::cout << vector[i] << "\t";
		}
		std::cout << std::endl;
	}
}

void setKernelArguments(const size_t size, float* A, float* b, float* x0, float* x1, float* norm,
	cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
	cl_mem& buffer_A, cl_mem& buffer_b, cl_mem& buffer_x0, cl_mem& buffer_x1, cl_mem& buffer_norm) {
	cl_int error = 0;
	buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * size * sizeof(float), A, &error);
	assert(error == CL_SUCCESS && "Create input buffer A failed");
	buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), b, &error);
	assert(error == CL_SUCCESS && "Create input buffer b failed");
	buffer_x0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size * sizeof(float), x0, &error);
	assert(error == CL_SUCCESS && "Create input buffer x0 failed");
	buffer_x1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size * sizeof(float), x1, &error);
	assert(error == CL_SUCCESS && "Create input buffer x1 failed");
	buffer_norm = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size * sizeof(float), norm, &error);
	assert(error == CL_SUCCESS && "Create input buffer norm failed");
	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
	assert(error == CL_SUCCESS && "clSetKernelArg 0-arg failed");
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
	assert(error == CL_SUCCESS && "clSetKernelArg 1-arg failed");
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_x0);
	assert(error == CL_SUCCESS && "clSetKernelArg 2-arg failed");
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_x1);
	assert(error == CL_SUCCESS && "clSetKernelArg 3-arg failed");
	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_norm);
	assert(error == CL_SUCCESS && "clSetKernelArg 4-arg failed");
	cl_uint size_copy = size;
	error = clSetKernelArg(kernel, 5, sizeof(cl_uint), &size_copy);
	assert(error == CL_SUCCESS && "clSetKernelArg 5-arg failed");
}

void initializeKernel(std::string kernal_name, cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
	cl_device_id& device, cl_program& program, cl_device_type device_type) {
	cl_uint platformsCount = 0;
	cl_int error = 0;
	clGetPlatformIDs(0, nullptr, &platformsCount);
	assert(platformsCount > 0 && "Not platforms");
	cl_platform_id* platforms = new cl_platform_id[platformsCount];
	clGetPlatformIDs(platformsCount, platforms, nullptr);
	cl_platform_id platform = platforms[1];
	char platform_name[128];
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
	std::cout << platform_name << std::endl;
	delete[] platforms;
	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM,
										(cl_context_properties)platform, 0 };
	context = clCreateContextFromType((nullptr == platform) ? nullptr : properties,
		device_type, nullptr, nullptr, &error);
	assert(error == CL_SUCCESS && "Create context from type failed");
	size_t device_count = 0;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &device_count);
	assert(device_count > 0 && "Not device");
	cl_device_id* devices = new cl_device_id[device_count];
	clGetContextInfo(context, CL_CONTEXT_DEVICES, device_count, devices, NULL);
	device = devices[0];
	char device_name[128];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
	std::cout << device_name << std::endl;
	cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	queue =	clCreateCommandQueueWithProperties(context, device, props, &error);
	assert(error == CL_SUCCESS && "Create command queue with properties failed");
	const char* kernelSource = content.c_str();
	size_t kernelLen[] = { strlen(kernelSource) };
	program = clCreateProgramWithSource(context, 1, &kernelSource, kernelLen, &error);
	assert(error == CL_SUCCESS && "Create program with source failed");
	error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cout << "Build prog failed" << std::endl;
		size_t logSize = 0;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0,
			nullptr, &logSize);
		char* log = new char[logSize];
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, logSize,
			log, nullptr);
		std::cout << log;
		assert(false && "Build failed");
	}
	kernel = clCreateKernel(program, kernal_name.c_str(), &error);
	assert(error == CL_SUCCESS && "Create kernel failed");
}

size_t upperBound(const size_t const_size, size_t size) {
	while (size < const_size) {
		size *= 2;
	}
	return size;
}

void solve(const char* kernel_name, size_t size, float* A, float* b, float* x0, float* x1, float* norm, cl_device_type device_type, double* time) {
	cl_mem buffer_A;
	cl_mem buffer_b;
	cl_mem buffer_x0;
	cl_mem buffer_x1;
	cl_mem buffer_norm;

	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_device_id device;
	cl_program program;

	initializeKernel(kernel_name, kernel, context, queue, device, program, device_type);
	setKernelArguments(size, A, b, x0, x1, norm, kernel, context, queue, buffer_A,
		buffer_b, buffer_x0, buffer_x1, buffer_norm);
	size_t group_size = 0;
	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(size_t), &group_size, NULL);
	size_t global_size = upperBound(size, group_size);
	size_t count = 0;
	cl_int error = 0;
	
	while (true) {
		cl_event event;
		error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &group_size, 0, NULL, &event);
		assert(error == CL_SUCCESS && "Read NDRange failed");
		clWaitForEvents(1, &event);
		cl_ulong time_start;
		cl_ulong time_end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		*time += (time_end - time_start) / 1.0e6;
		clEnqueueReadBuffer(queue, buffer_x1, CL_TRUE, 0, sizeof(float) * size, x1, 0, NULL, NULL);
		clEnqueueReadBuffer(queue, buffer_norm, CL_TRUE, 0, sizeof(float) * size, norm, 0, NULL, NULL);
		float sum = 0.0f;
		for (size_t k = 0; k < size; k++) {
			sum += norm[k] * norm[k];
		}
		count++;
		std::swap(x0, x1);
		std::cout << "Iteration: " << count << "; Norm: " << sqrt(sum) << std::endl;
		if (count > iterations || sqrt(sum) < epsilon) {
			break;
		}
		clEnqueueWriteBuffer(queue, buffer_x0, CL_TRUE, 0, sizeof(float) * size, x0, 0, NULL, NULL);
	}
	clReleaseMemObject(buffer_A);
	clReleaseMemObject(buffer_b);
	clReleaseMemObject(buffer_x0);
	clReleaseMemObject(buffer_x1);
	clReleaseMemObject(buffer_norm);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

void checkSolution(size_t size, float* A, float* b, float* solution) {
	float* check = new float[size];
	for (int i = 0; i < size; i++) {
		check[i] = 0;
		for (int j = 0; j < size; j++) {
			check[i] += A[i * size + j] * solution[j];
		}
	}
	float sum = 0.0f;
	for (size_t k = 0; k < size; k++) {
		sum += (check[k] - b[k]) * (check[k] - b[k]);
	}
	delete[] check;
	std::cout << "Norm of difference " << sqrt(sum) << std::endl;
}

int main() {
	float* A = new float[size * size];
	fillMatrix(A, size);
	float *b = new float[size];
	fillVector(b, size);
	float *x0 = new float[size];
	fillVector(x0, size);
	float *x1 = new float[size];
	float *norm = new float[size];
	double time_gpu = 0;
	float* x0_gpu = new float[size];
	for (size_t index = 0; index < size; ++index) {
		x0_gpu[index] = x0[index];
	}
	printSystem(A, b, size);
	solve("iterate", size, A, b, x0_gpu, x1, norm, CL_DEVICE_TYPE_GPU, &time_gpu);
	checkSolution(size, A, b, x0_gpu);
	printVector(x0_gpu, size);
	std::cout << "OpenCL GPU duration : " << time_gpu << "ms" << std::endl << std::endl;
	delete[] x0_gpu;
	
	double time_cpu = 0;
	float* x0_cpu = new float[size];
	for (size_t index = 0; index < size; ++index) {
		x0_cpu[index] = x0[index];
	}
	solve("iterate", size, A, b, x0_cpu, x1, norm, CL_DEVICE_TYPE_CPU, &time_cpu);
	checkSolution(size, A, b, x0_cpu);
	printVector(x0_cpu, size);
	std::cout << "OpenCL CPU duration : " << time_cpu << "ms" << std::endl << std::endl;
	delete[] x0_cpu;

	delete[] A;
	delete[] b;
	delete[] x0;
	delete[] x1;
	delete[] norm;
	system("pause");
}
