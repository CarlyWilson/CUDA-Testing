#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void mykernel(void){}

__global__ void add(int *n, int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	c[blockIdx.x+(n[0]/10)] = a[blockIdx.x+(n[0]/10)] + b[blockIdx.x+(n[0]/10)];
	c[blockIdx.x+2*(n[0]/10)] = a[blockIdx.x+2*(n[0]/10)] + b[blockIdx.x+2*(n[0]/10)];
	c[blockIdx.x+3*(n[0]/10)] = a[blockIdx.x+3*(n[0]/10)] + b[blockIdx.x+3*(n[0]/10)];
	c[blockIdx.x+4*(n[0]/10)] = a[blockIdx.x+4*(n[0]/10)] + b[blockIdx.x+4*(n[0]/10)];
	c[blockIdx.x+5*(n[0]/10)] = a[blockIdx.x+5*(n[0]/10)] + b[blockIdx.x+5*(n[0]/10)];
	c[blockIdx.x+6*(n[0]/10)] = a[blockIdx.x+6*(n[0]/10)] + b[blockIdx.x+6*(n[0]/10)];
	c[blockIdx.x+7*(n[0]/10)] = a[blockIdx.x+7*(n[0]/10)] + b[blockIdx.x+7*(n[0]/10)];
	c[blockIdx.x+8*(n[0]/10)] = a[blockIdx.x+8*(n[0]/10)] + b[blockIdx.x+8*(n[0]/10)];
	c[blockIdx.x+9*(n[0]/10)] = a[blockIdx.x+9*(n[0]/10)] + b[blockIdx.x+9*(n[0]/10)];
}

void random_ints(int* a, int N){
	int i;
	for(i = 0; i < N; i++){
		//a[i] = rand();
		a[i] = i;
	}
}

#define N (100)
#define THREADS_PER_BLOCK 4

int main(void){
	mykernel<<<1,1>>>();
	printf("Hello World!\n");

	int *a, *b, *c;
	int *d_a, *d_b, *d_c, *d_n;
	int size = N * sizeof(int);

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMalloc((void **)&d_n, sizeof(int));

	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);
	int *N2 = (int*)malloc(sizeof(int));
	N2[0] = N;

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, N2, sizeof(int), cudaMemcpyHostToDevice);

	add<<<N/10, 1>>>(d_n, d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++){
		cout<<c[i]<<endl;
	}

	free(a); free(b); free(c); free(N2);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}
