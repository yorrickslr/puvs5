#include "CL/cl.h"                              
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------------
// Speicheranforderung fuer eine leere Matrix A[row][col]. 

float **alloc_mat(int row, int col)
{
	float **A;

	A = (float **)calloc(row, sizeof(float *));           // Zeiger auf die Zeilen
	if (A) {
		A[0] = (float *)calloc(row*col, sizeof(float));         // Alle Matrixelemente
		if (A[0]) {
			for (int i = 1; i < row; i++)
				A[i] = A[i - 1] + col;
			return A;
		}
	}
	perror("out of memory!"); exit(1);
}

// ----------------------------------------------------------------------------------
// Zufaellige Initialisierung einer Matrix mit den Werten [0..9]. 

void init_mat(float **A, int row, int col)
{
	for (int i = 0; i < row*col; i++)
		A[0][i] = (float)(rand() % 10);
}

// ----------------------------------------------------------------------------------
// Sequentielle Matrixmultiplikation C = A*B. 

float **mult_mat(float **A, float **B, int d1, int d2, int d3)
{
	float **C = alloc_mat(d1, d3);                            // Erzeugt neue Matrix
	int i, j, k;

	for (i = 0; i < d1; i++)
		for (j = 0; j < d3; j++)
			for (k = 0; k < d2; k++)
				C[i][j] += A[i][k] * B[k][j];                 // Matrixmultiplikation

	return C;
}

void mult_mat(float **A, float **B, float **C, int d0, int d1, int d2, int d3)
{
	int i, j, k;

	for (i = d0; i < d1; i++)          // Multipliziert nur Teile einer großen Matrix
		for (j = 0; j < d3; j++)
			for (k = 0; k < d2; k++)
				C[i][j] += A[i][k] * B[k][j];          // Füllt existierende Matrix C

}

// ----------------------------------------------------------------------------------
// Tested die Gleichheit von Matrizen  

void is_correct(float **A, float **B, int row, int col)
{
	int i, j;

	for (i = 0; i < row; i++)
		for (j = 0; j < col; j++)
			if (A[i][j] != B[i][j])
				printf("error!\n");

	printf("ok.\n");
}

// ---------------------------------------------------------------------------
// Ausgabe der Matrixelemente fuer Debugzwecke

void print_mat(float **A, int row, int col, char *tag)
{
	int i, j;

	printf("Matrix %s:\n", tag);
	for (i = 0; i < row; i++) {
		for (j = 0; j < col; j++)
			printf("%6.1f   ", A[i][j]);
		printf("\n");
	}
}

const char *KernelSource =
"__kernel void matmult_ocl(__global float *A, __global float *B, __global int *d, __global float *C) { \n"
"	size_t id = get_global_id(0);                                                                      \n"
"	int shiftA = ((int) id/d[2]) * d[2];                                                               \n"
"	int shiftB = id%d[3];                                                                              \n"
"	for(int i=0; i<d[2]; i++) {                                                                        \n"
"		C[id] += A[shiftA+i] * B[shiftB+i*d[3]];                                                       \n"
"	}                                                                                                  \n"
"}                                                                                                     \n"
"\n";

int main(int argc, char** argv)
{
	cl_int err;
	cl_platform_id* platforms = NULL;
	char platform_name[1024];
	cl_device_id device_id = NULL;
	cl_uint	num_of_platforms = 0;
	cl_uint num_of_devices = 0;
	cl_context context;
	cl_kernel kernel;
	cl_command_queue command_queue;
	cl_program program;
	cl_mem input1, input2, input3, output;
	float **A, **B, **C, **serialC;	// matrices
	int d1, d2, d3;         // dimensions of matrices

							/* print user instruction */
	if (argc != 4)
	{
		printf("Matrix multiplication: C = A x B\n");
		printf("Usage: %s <NumRowA> <NumColA> <NumColB>\n", argv[0]);
		return 0;
	}

	/* read user input */
	d1 = atoi(argv[1]);		// rows of A and C
	d2 = atoi(argv[2]);     // cols of A and rows of B
	d3 = atoi(argv[3]);     // cols of B and C
	int d[4] = { 0, d1, d2, d3};
	size_t global[1] = { (size_t) d1*d3 };

	printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1, d2, d2, d3);

	/* prepare matrices */
	A = alloc_mat(d1, d2);
	init_mat(A, d1, d2);
	B = alloc_mat(d2, d3);
	init_mat(B, d2, d3);
	C = alloc_mat(d1, d3);
	serialC = alloc_mat(d1, d3);

	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS) {
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	platforms = (cl_platform_id *)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS) {
		printf("No platforms found. Error: %d\n", err);
		return 0;
	} else {
		int nvidia_platform = 0;
		for (unsigned int i = 0; i<num_of_platforms; i++) {
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (err != CL_SUCCESS) {
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}
			if (strstr(platform_name, "NVIDIA") != NULL) {
				nvidia_platform = i;
				break;
			}
		}
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS) {
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS) {
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	kernel = clCreateKernel(program, "matmult_ocl", &err);
	if (err != CL_SUCCESS) {
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}

	input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, d1*d2*sizeof(float), NULL, &err);
	input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, d2*d3*sizeof(float), NULL, &err);
	input3 = clCreateBuffer(context, CL_MEM_READ_ONLY, 4 * sizeof(int), NULL, &err);

	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, d1*d3*sizeof(float), NULL, &err);

	clEnqueueWriteBuffer(command_queue, input1, CL_TRUE, 0, d1*d2*sizeof(float), *A, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, input2, CL_TRUE, 0, d2*d3*sizeof(float), *B, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, input3, CL_TRUE, 0, 4 * sizeof(int), d, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &input3);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);

	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, d1*d3*sizeof(float), *C, 0, NULL, NULL);
	// for (unsigned int i = 0; i < (unsigned int) d1*d3; i++)
	//	printf("%f\n", C[0][i]);

	clReleaseMemObject(input1);
	clReleaseMemObject(input2);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	printf("Running serial algorithm...\n");
	serialC = mult_mat(A, B, d1, d2, d3);

	printf("Checking results... ");
	is_correct(C, serialC, d1, d3);

	printf("Finishing...\n");

	return 0;
}
