#include <time.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define W (5000)
#define N (5000)


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void DistanceForBMUCalc(double *input, double *v, double *x){
	double d = 0;
	for(long long int i = 0; i < W; i++){
		d += (v[i+W*blockIdx.x] - input[i]) * (v[blockIdx.x*W+i] - input[i]);
	}
	x[blockIdx.x] = sqrt(d);
}

int main(int argc, char* argv[]){
	int t_i = time(NULL);
	srand(0);
	
	double *v;
	double *d_v;
	long long int size = N*W * sizeof(double);

	long long int d_vSize = N*W * sizeof(double);
	gpuErrchk(cudaMalloc((void **)&d_v, d_vSize));
	v = (double *)malloc(size); 
	
	for(int i = 0; i < N*W; i++){
		v[i] = rand();
	}
	
	double *distances;
	double *d_distances;
	long long int distanceArraySize = N * sizeof(double);
	gpuErrchk(cudaMalloc((void **)&d_distances, distanceArraySize));
	distances = (double *)malloc(distanceArraySize);

	double *training;
	double *d_training;
	long long int trainingSize = W * sizeof(double); 
	gpuErrchk(cudaMalloc((void **)&d_training, trainingSize));
	training = (double *)malloc(trainingSize);

	for(int i = 0; i < W; i++){
		training[i] = rand();
	}

	for(int i = 0; i < 1; i++){
		gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(d_training, training, trainingSize, cudaMemcpyHostToDevice));

		DistanceForBMUCalc<<<N, 192>>>(d_training, d_v, d_distances);

		gpuErrchk(cudaMemcpy(distances, d_distances, distanceArraySize, cudaMemcpyDeviceToHost));

		double dmin = distances[0];
		//int index = 0;
		for(int j = 0; j < N; j++){
			//cout<<i<<" "<<distances[i]<<endl;
			if(distances[j] < dmin){ 
				dmin = distances[j];
				//index = j;
			}
		}
	}
	//cout<<"BMU is "<<index<<endl;

	int t_f = time(NULL);
	int total_time = t_f - t_i;
	cout<<"Total Execution Time: "<<total_time<<endl;
}
