#include <time.h>
#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

double Distance(vector<double> a, vector<double> b){
	if(a.size() != b.size()){
		cout<<"Error! Cannot do distance between vectors!"<<endl;
		return -1;
	}
	double d = 0;
	for(int i = 0; i < a.size(); i++){
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(d); //to CPU and stored in an array
}

__global__ void FindBMU(double *input, double **v, int *x){
	x = 0;

	double dmin = Distance(input, v[0]);
	double d = 100 * fabs(dmin);
//GPU starts here
	for(int i = 0; i < v.size(); i++){
		d = Distance(v[i], input);
		if(d < dmin){ // CPU will compare everything in the array
			dmin = d;
			x = i;
		}
	}
}

#define C(1920)

int main(int argc, char* argv[]){
	int t_i = time(NULL);
	srand(0);

	double **v;
	double **d_v;
	int size = C * sizeof(int);

	cudaMalloc((void **)&d_v, size);
	v = (double *)malloc(size); 

	for(int i = 0; i < v.size(); i++){
		for(int j = 0; j < C; j++){
			v[i][j] = rand();
		}
	}

	int *BMUx;
	cudaMalloc((void **)&d_BMUx, size);
	BMUx = (int *)malloc(size);

	double *training;
	cudaMalloc((void **)&d_training, size);
	training = (double *)malloc(size);

	for(int l = 0; l < training.size(); l++){
		training[l] = rand();
	}

	cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_training, training, size, cudaMemcpyHostToDevice);

	for(int k = 0; k < C; k++){
		FindBMU<<<C, 192>>>(d_training, d_v, d_BMUx);
	}

	int t_f = time(NULL);
	int total_time = t_f - t_i;
	cout<<"Total Execution Time: "<<total_time<<endl;
}
