

#include <ctime>     // for a random seed
#include <fstream>   // for file-reading
#include <iostream>  // for file-reading
#include <sstream>   // for file-reading
#include <vector>
#include <algorithm> // for the sort function
#include <random>
#include <numeric> // iota

#include <Eigen/Dense> // ATTENTION : nécessite la commande  -I ../../../../../../usr/include/eigen3/ pour compiler !

#include <thread> // for parallel
#include <mutex> // for parallel
#include <functional>

#include <math.h>// pow


#include <typeinfo>


#include "kPLM_algo.h"


int main(int argc, char** argv){
	const size_t d = 2;

try
{

	kPLM<d> algo(argv[1],50,20,1000,10,20,2,0.01,0.01,0.01,100);

	std::cout<< "The parameters of the algorithm are:"<< std::endl;
	std::tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,double,double,double,double,bool,size_t> params = algo.parameters_access();
	std::cout<< "method = " << std::get<0>(params) << std::endl;
	std::cout<< "k = " << std::get<1>(params) << std::endl;
	std::cout<< "q = " << std::get<2>(params) << std::endl;
	std::cout<< "n_signal = " << std::get<3>(params) << std::endl;
	std::cout<< "n_starts = " << std::get<4>(params) << std::endl;
	std::cout<< "epochs = " << std::get<5>(params) << std::endl;
	std::cout<< "d_intrinsic = " << std::get<6>(params) << std::endl;
	std::cout<< "sigma2_min = " << std::get<7>(params) << std::endl;
	std::cout<< "sigma2_max = " << std::get<8>(params) << std::endl;
	std::cout<< "lambda_min = " << std::get<9>(params) << std::endl;
	std::cout<< "lambda_max = " << std::get<10>(params) << std::endl;
	std::cout<< "normalized_det = " << std::get<11>(params) << std::endl;

	std::cout<< "The number of points for the algorithm is:"<< std::get<12>(params) << std::endl;

	std::cout<<std::endl;

	std::cout<< "The first five points of the algorithm are:" << std::endl;
	std::vector<Matrix<double,1,d>> points_values = algo.points_values_access();
	for(size_t i = 0; i<5; ++i)
	{
		std::cout<< points_values[i][0] << ", " <<  points_values[i][1]  << std::endl;
	}
	std::cout<<std::endl;
	
	std::cout<<"Algorithm with non increasing intrinsic dimension : "<< std::endl;

	//auto ans = algo.run_multiple_kPLM(6, 0);
	auto ans = algo.run_multiple_kPLM_several_times(6,0);

	for(auto it = ans.begin(); it!= ans.end(); ++it)
	{
		std::cout<< std::get<3>(*it)<<std::endl;
	}


	std::cout<<"On lance l'algorithme !"<< std::endl;
/*	algo.run_kPLM();
	algo.print_optimal_centroids();
	algo.print_optimal_clusters();
	algo.print_optimal_costs();

	algo.write_centroids();
	algo.write_clusters();
	algo.write_costs();
	algo.write_matrices_inv();
/*
*/

}

catch(std::invalid_argument const&e)
{
	cerr << e.what() << endl;
}
	return(0);

}

