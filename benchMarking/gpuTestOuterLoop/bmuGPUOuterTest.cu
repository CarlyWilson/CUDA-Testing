#include <time.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define W (8192)
#define N (8192)

#define THREADS_PER_BLOCK (1)
#define NUMBER_BLOCKS (N/THREADS_PER_BLOCK)

typedef float myFloat;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void DistanceForBMUCalcBlocks(myFloat *input, myFloat *v, myFloat *x){
	myFloat d = 0;
	for(long long int i = 0; i < W; i++){
		d += (v[i+W*blockIdx.x] - input[i]) * (v[blockIdx.x*W+i] - input[i]);
	}
	x[blockIdx.x] = sqrt(d);
}

__global__ void DistanceForBMUCalcBlocksAndThreads(myFloat *input, myFloat *v, myFloat *x){
	myFloat d = 0;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for(long long int i = 0; i < W; i++){
		d += (v[i+W*index] - input[i]) * (v[index*W+i] - input[i]);
	}
	x[index] = sqrt(d);
}

int main(int argc, char* argv[]){
	steady_clock::time_point t_i = steady_clock::now();
	srand(0);
	
	myFloat *v;
	myFloat *d_v;
	long long int size = N*W * sizeof(myFloat);

	long long int d_vSize = N*W * sizeof(myFloat);
	gpuErrchk(cudaMalloc((void **)&d_v, d_vSize));
	v = (myFloat *)malloc(size); 
	
	for(int i = 0; i < N*W; i++){
		v[i] = rand();
	}
	
	myFloat *distances;
	myFloat *d_distances;
	long long int distanceArraySize = N * sizeof(myFloat);
	gpuErrchk(cudaMalloc((void **)&d_distances, distanceArraySize));
	distances = (myFloat *)malloc(distanceArraySize);

	myFloat *training;
	myFloat *d_training;
	long long int trainingSize = W * sizeof(myFloat); 
	gpuErrchk(cudaMalloc((void **)&d_training, trainingSize));
	training = (myFloat *)malloc(trainingSize);

	int index = 0;
	for(int i = 0; i < W; i++){
		training[i] = rand();
	}

	steady_clock::time_point workI = steady_clock::now();

	for(int i = 0; i < 1; i++){
		//steady_clock::time_point t_CPUToGPUI = steady_clock::now();
		gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(d_training, training, trainingSize, cudaMemcpyHostToDevice));

		//int t_CPUToGPUF = time(NULL);
		//cout<<"Finished copying to device "<<t_CPUToGPUF - t_CPUToGPUI<<endl;		

		//DistanceForBMUCalcBlocks<<<NUMBER_BLOCKS, THREADS_PER_BLOCK>>>(d_training, d_v, d_distances);
		DistanceForBMUCalcBlocksAndThreads<<<NUMBER_BLOCKS, THREADS_PER_BLOCK>>>(d_training, d_v, d_distances);
		
		cudaThreadSynchronize();

		//cout<<"Finished distance calc"<<endl;
		//int t_GPUToCPUI = time(NULL);

		gpuErrchk(cudaMemcpy(distances, d_distances, distanceArraySize, cudaMemcpyDeviceToHost));
		
		//int t_GPUToCPUF = time(NULL);
		//cout<<"Finished Device to CPU copy "<<t_GPUToCPUF - t_GPUToCPUI<<endl;

		myFloat dmin = distances[0];
		for(int j = 0; j < N; j++){
			if(distances[j] < dmin){ 
				dmin = distances[j];
				index = j;
			}
		}
	}
	steady_clock::time_point workF = steady_clock::now();
	cout<<"Total work execution time "<<duration_cast<milliseconds>(workF - workI).count()<<endl;
	cout<<"BMU is "<<index<<endl;

	steady_clock::time_point t_f = steady_clock::now();
	cout<<"Total Execution Time: "<<duration_cast<milliseconds>(t_f - t_i).count()<<endl;
}
