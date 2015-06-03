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
	return sqrt(d);
}

void FindBMU(vector<double> input, vector<vector<double> > v, int &x){
	x = 0;

	double dmin = Distance(input, v[0]);
	double d = 100 * fabs(dmin);

	for(int i = 0; i < v.size(); i++){
		d = Distance(v[i], input);
		if(d < dmin){
			dmin = d;
			x = i;
		}
	}
}

int main(int argc, char* argv[]){
	int t_i = time(NULL);
	srand(0);

	int c = 1920;

	vector<vector<double> > v(c);
	for(int i = 0; i < v.size(); i++){
		v[i].resize(c);
		for(int j = 0; j < c; j++){
			v[i][j] = rand();
		}
	}

	int BMUx;

	vector<double> training(c);
	for(int l = 0; l < training.size(); l++){
		training[l] = rand();
	}

	for(int k = 0; k < c; k++){
		FindBMU(training, v, BMUx);
	}

	int t_f = time(NULL);
	int total_time = t_f - t_i;
	cout<<"Total Execution Time: "<<total_time<<endl;
}
