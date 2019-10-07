//The Code for Fast Fourier Transforms

#define N 64

#include "timerc.h"
#include <stdio.h>
#include <complex.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuComplex.h>

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

__host__ __device__ cuDoubleComplex minus(cuDoubleComplex a, cuDoubleComplex b)
{
	double a1 = cuCreal(a) - cuCreal(b);
	double a2 = cuCimag(a) - cuCimag(b);
	return make_cuDoubleComplex(a1, a2);

}

__host__ __device__ cuDoubleComplex pwcom(cuDoubleComplex a, int b)
{
	cuDoubleComplex ori = make_cuDoubleComplex(1.0, 0.0);
	for(int i = 0; i < b; i++)
		ori = mult(ori, a);
	return ori;
	
}

__host__ __device__ int comp(cuDoubleComplex *vec1, cuDoubleComplex *vec2, int n)
{
	int error = 0;
	for(int i = 0; i < n; i++)
		if(((int)cuCreal(vec1[i]) != (int)cuCreal(vec2[i])) || ((int)cuCimag(vec1[i]) != (int)cuCimag(vec2[i])))
			error += 1;
	return error;

}

__global__ void cumerge(cuDoubleComplex* in, int n)
{
	int half = n/2;
	cuDoubleComplex temp;
	temp = in[threadIdx.x];
	__syncthreads();
	if(threadIdx.x < half)
		in[threadIdx.x * 2] = temp;
	else
		in[(threadIdx.x - half) * 2 + 1] = temp;
}

//Dyanmic Parallelism
__global__ void dynamic(cuDoubleComplex* in, cuDoubleComplex w, int n, int deg, int start)
{
	cuDoubleComplex temp1, temp2;	
	temp1 = in[threadIdx.x + start];
	temp2 = in[threadIdx.x + start + n/2];
	in[threadIdx.x + start] = plus(temp1, temp2);
	in[threadIdx.x + start + n/2] = mult(pwcom(w, threadIdx.x * deg), minus(temp1, temp2));
	__syncthreads();
	if(threadIdx.x == 0 && n >= 4){
		dynamic<<<1, n/4>>>(in, w, n/2, deg*2, start);
		dynamic<<<1, n/4>>>(in, w, n/2, deg*2, start + n/2);
		cudaDeviceSynchronize();
		cumerge<<<1, n>>>(&in[start], n);
	}
}

__global__ void rearrange(cuDoubleComplex* in, int n)
{
	for(int i = 4; i <= n; i *= 2){
		cuDoubleComplex temp;
		temp = in[threadIdx.x];
		__syncthreads();
		if(threadIdx.x % i < i/2)
			in[threadIdx.x + threadIdx.x % i] = temp;
		else
			in[threadIdx.x - i/2 + threadIdx.x % (i/2) + 1] = temp;
		__syncthreads();
	}
}

__global__ void one_kernel(cuDoubleComplex* in, cuDoubleComplex w, int n)
{
	__shared__ cuDoubleComplex temp[N];
	temp[threadIdx.x] = in[threadIdx.x];
	temp[threadIdx.x + n/2] = in[threadIdx.x + n/2];
	__syncthreads();	
	int degree = 1;
	for(int i = n; i >= 2; i/=2){
		int half = i/2;
		int id = threadIdx.x + (threadIdx.x / half) * half;
		cuDoubleComplex temp1, temp2;
		temp1 = temp[id];
		temp2 = temp[id + half];
		temp[id] = plus(temp1, temp2);
		temp[id + half] = mult(pwcom(w, ((threadIdx.x % half) * degree)), minus(temp1, temp2));
		degree *= 2;
		__syncthreads();
	}
	in[threadIdx.x] = temp[threadIdx.x];
	in[threadIdx.x + n/2] = temp[threadIdx.x + n/2];
	__syncthreads();	
}

void comb(cuDoubleComplex *res, int n, int a, int b)
{
	cuDoubleComplex re1[n], re2[n];
	for(int i = 0; i < n; i++){
		re1[i] = res[a + i];
		re2[i] = res[b + i];
	}	
	for(int i = 0; i < 2 * n; i += 2){
		res[i] = re1[i/2];
		res[i + 1] = re2[i/2];
	}
}


void trans(cuDoubleComplex *a, int n, int deg, cuDoubleComplex w)
{	
	cuDoubleComplex t;	
	if(n == 2){
			t = a[0];
			a[0] = plus(a[0], a[1]); 
			a[1] = minus(t, a[1]);			
	}
	else{
		cuDoubleComplex temp1, temp2;
		for(int i = 0; i < n / 2; i++){
			temp1 = a[i];
			temp2 = a[i + n/2];
			a[i] = plus(temp1, temp2);
			a[n/2 + i] = mult(pwcom(w, i * deg), minus(temp1, temp2));
		}
/*
		printf("Test\n");
		for(int i = 0; i < n; i++)
			printf("%.0f + %.0fi\n", cuCreal(a[i]), cuCimag(a[i]));
		printf("Test finished\n");
*/
		trans(&a[0], n/2, deg*2, w);
		trans(&a[n/2], n/2, deg*2, w);
		comb(a, n/2, 0, n/2);
	}
}

cuDoubleComplex calc (int n)
{
	double theta = 2 * M_PI / (double)n;
	cuDoubleComplex w = make_cuDoubleComplex(cos(theta), sin(theta));	
	return w;
}

int main()
{
	float ctime, gtime;
	srand(time(NULL));
	cuDoubleComplex *vec_host = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N);
	cuDoubleComplex *vec_host_out = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N);
	cuDoubleComplex *vec_dev;
	cudaMalloc(&vec_dev, sizeof(cuDoubleComplex) * N);
	for(int i = 0; i < N; i++)
		vec_host[i] = make_cuDoubleComplex((double)(rand()%10), (double)(rand()%10));
/*
	printf("Vector:\n");
	for(int i = 0; i < N; i++)
		printf("%.0f + %.0fi\n", cuCreal(vec_host[i]), cuCimag(vec_host[i]));
	printf("\n\n");
*/
	cudaMemcpy(vec_dev, vec_host, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);
	gstart();
	//dynamic<<<1, N/2>>>(vec_dev, make_cuDoubleComplex(0.0, 1.0), N, 1, 0);
	//dynamic<<<1, N/2>>>(vec_dev, calc(N), N, 1, 0);
	one_kernel<<<1, N/2>>>(vec_dev, make_cuDoubleComplex(0.0, 1.0), N);
	//one_kernel<<<1, N/2>>>(vec_dev, calc(N), N);
	rearrange<<<1, N>>>(vec_dev, N);
	gend(&gtime);
	cudaMemcpy(vec_host_out, vec_dev, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
	cstart();
	trans(vec_host, N, 1, make_cuDoubleComplex(0.0, 1.0));
	//trans(vec_host, N, 1, calc(N));
	cend(&ctime);
/*
	printf("GPU:\n");
	for(int i = 0; i < N; i++)
		printf("%.0f + %.0fi\n", cuCreal(vec_host_out[i]), cuCimag(vec_host_out[i]));
	printf("CPU:\n");
	for(int i = 0; i < N; i++)
		printf("%.0f + %.0fi\n", cuCreal(vec_host[i]), cuCimag(vec_host[i]));
*/
	if(comp(vec_host, vec_host_out, N) == 0)
		printf("Two vectors computed by CPU and GPU matches!\n");
	else 
		printf("Two vectors computed by CPU and GPU does not match!\n");
	printf("CPU Time: %f GPU Time: %f\n", ctime, gtime);
	free(vec_host);
	free(vec_host_out);
	cudaFree(vec_dev);
	return 0;
}
