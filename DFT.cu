//Code for Discrete Fourier Transform

#define N 64

#include "timerc.h"
#include <stdio.h>
#include <complex.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuComplex.h>

//the multiplication operator in cuComplex.h library is cuCmul, but it seems to be problematic because it always gives meaningless result, so I decided to rewrite the operator
__host__ __device__ cuDoubleComplex mult(cuDoubleComplex a, cuDoubleComplex b)
{
	double a1 = cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b);
	double a2 = cuCreal(a) * cuCimag(b) + cuCreal(b) * cuCimag(a);
	return make_cuDoubleComplex(a1, a2);
}

__host__ __device__ cuDoubleComplex plus(cuDoubleComplex a, cuDoubleComplex b)
{
	double a1 = cuCreal(a) + cuCreal(b);
	double a2 = cuCimag(a) + cuCimag(b);
	return make_cuDoubleComplex(a1, a2);

}

__host__ __device__ int comp(cuDoubleComplex *vec1, cuDoubleComplex *vec2, int n)
{
	int error = 0;
	for(int i = 0; i < n; i++)
		if(((int)cuCreal(vec1[i]) != (int)cuCreal(vec2[i])) || ((int)cuCimag(vec1[i]) != (int)cuCimag(vec2[i])))
			error += 1;
	return error;

}

__global__ void naive_mult(cuDoubleComplex *matrix, cuDoubleComplex *vector, int n)
{
	matrix[threadIdx.x * n + blockIdx.x] = mult(matrix[threadIdx.x * n + blockIdx.x], vector[blockIdx.x]);
}



__global__ void naive_sum(cuDoubleComplex *matrix_in, cuDoubleComplex *vector_out, int n)
{
	int id = n * blockIdx.x + threadIdx.x;
	int space = n/2;
	while(space >= 1){
		if((id % n + space) < n){
			matrix_in[id + space] = cuCadd(matrix_in[id + space], matrix_in[id]);	
			__syncthreads();		
		}
		id += space;
		space /= 2;	
	}
	if(threadIdx.x == 0) 
		vector_out[blockIdx.x] = matrix_in[blockIdx.x * n + n - 1];
}

__global__ void naive_sum_shared(cuDoubleComplex *matrix_in, cuDoubleComplex *vector_out, int n)
{
	__shared__ cuDoubleComplex temp[N];
	int id = n * blockIdx.x + threadIdx.x;
	int space = n/2;
	if(threadIdx.x * 2 < n){
		temp[threadIdx.x] = matrix_in[id];
		temp[threadIdx.x + space] = matrix_in[id + space];	
	}
	__syncthreads();
	while(space >= 1){
		if((id % n + space) < n){
			temp[id % n + space] = cuCadd(temp[id % n + space], temp[id % n]);
			__syncthreads();
		}
		id += space;
		space /= 2;
	}
	if(threadIdx.x == 0)
		vector_out[blockIdx.x] = temp[n-1];
}

cuDoubleComplex calc (int p, int n)
{
	double theta = 2 * M_PI / (double)n;
	cuDoubleComplex ori = make_cuDoubleComplex(1.0, 0.0); 
	cuDoubleComplex w = make_cuDoubleComplex(cos(theta), sin(theta));	
 	for(int i = 1; i <= p; i++)
		ori = mult(ori, w);
	return ori;
}

int main()
{
	float host_time, dev_time;	
	srand(time(NULL));
	cuDoubleComplex *dev_m, *m = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N * N);
	
	cudaMalloc(&dev_m, sizeof(cuDoubleComplex) * N * N);
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			m[N * i + j] = calc(i*j, N);
	cudaMemcpy(dev_m, m, sizeof(cuDoubleComplex) * N * N, cudaMemcpyHostToDevice);

/*
	cuDoubleComplex *m2 = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N * N);
	cudaMemcpy(m2, dev_m, sizeof(cuDoubleComplex) * N * N, cudaMemcpyDeviceToHost);
	printf("\ncopied Transformation matrix is \n");
	for(int i = 0; i < N; i++){
		printf("\n");		
		for(int j = 0; j < N; j++)
				printf("%.0f + %.0fi ", cuCreal(m2[N * i + j]), cuCimag(m2[N * i + j]));
	}
	printf("\n\n\n");
*/

	cuDoubleComplex *vec_in = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N);
	cuDoubleComplex *vec_out = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N);
	cuDoubleComplex *dev_vec_in;
	cuDoubleComplex *dev_vec_out;
	cudaMalloc(&dev_vec_in, sizeof(cuDoubleComplex) * N);
	cudaMalloc(&dev_vec_out, sizeof(cuDoubleComplex) * N);
	cuDoubleComplex *vec_out2 = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N);
	for(int i = 0; i < N; i++){
		vec_out[i] = make_cuDoubleComplex(0.0, 0.0);
		vec_in[i] = make_cuDoubleComplex((double)(rand()%10), (double)(rand()%10));}

/*	
	printf("\nrandomly generated vector is\n");
	for(int i = 0; i < N; i++)
		printf("%.0f + %.0fi\n", cuCreal(vec_in[i]), cuCimag(vec_in[i]));
	printf("\n\n");
*/


	cudaMemcpy(dev_vec_in, vec_in, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec_out, vec_out, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);


	cstart();
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			vec_out[i] =cuCadd(vec_out[i], cuCmul(m[i * N + j], vec_in[j]));
	cend(&host_time);

	gstart();
	naive_mult<<<N, N>>>(dev_m, dev_vec_in, N);
/*	
	printf("\nafter multiplication \n");
	cuDoubleComplex *dev_m_2 = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N * N);	
	cudaMemcpy(dev_m_2, dev_m, sizeof(cuDoubleComplex) * N * N, cudaMemcpyDeviceToHost);
	for(int i = 0; i < N; i++){
		printf("\n");		
		for(int j = 0; j < N; j++)
				printf("%.0f + %.0fi \n\n", cuCreal(dev_m_2[N * i + j]), cuCimag(dev_m_2[N * i + j]));
	}
	printf("\n\n");
*/
	
	naive_sum<<<N, N/2>>>(dev_m, dev_vec_out, N);
//	naive_sum_shared<<<N, N/2>>>(dev_m, dev_vec_out, N);
	gend(&dev_time);
	cudaMemcpy(vec_out2, dev_vec_out, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
/*	
	// Testing the output of the original complex-valued matrix

	for(int i = 0; i < N; i++){
		printf("\n");		
		for(int j = 0; j < N; j++)
			if(cimag(m[N * i + j]))
				printf("%.0fi ", cimag(m[N * i + j]));
			else 
				printf("%.0f ", creal(m[N * i + j]));
	}
*/	
/*
	for(int i = 0; i < N; i++)
		printf("%.0f + %.0fi\n", cuCreal(vec_out2[i]), cuCimag(vec_out2[i]));
	printf("\n\n\n");
	for(int i = 0; i < N; i++)
		printf("%.0f + %.0fi\n", cuCreal(vec_out[i]), cuCimag(vec_out[i]));
*/	
	if(comp(vec_out, vec_out2, N) == 0)
		printf("Two vectors computed by CPU and GPU matches!\n");
	else 
		printf("Two vectors computed by CPU and GPU does not match!\n");
	printf("CPU Time: %f\n GPU Time: %f\n", host_time, dev_time);

	free(m);
	free(vec_in);
	free(vec_out);
	free(vec_out2);
	cudaFree(dev_m);
	cudaFree(dev_vec_in);
	cudaFree(dev_vec_out);	
	return 0;
}
