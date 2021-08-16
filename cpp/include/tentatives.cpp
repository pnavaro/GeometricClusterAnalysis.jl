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

using namespace std;
using namespace Eigen;


template <size_t d> 
class Point;

template <size_t d> 
class Centroid;

template <size_t d> 
class Sigma_inverted;

template <size_t d>
class kPLM;

#include<Point.h>
#include<Sigma_inverted.h>
#include<Centroid.h>
#include<kPLM.h>


int main(int argc, char** argv){
	const size_t d = 2;

try
{

	kPLM<d> algo1(argv[1]);
	kPLM<d> algo2(argv[1],3,4,50,10,20,2,0.01,0.01,0.01,100);

// 10 et 20 au lieu de 1 et 1.

/*
	std::cout<< "The parameters of the first algorithm are:"<< std::endl;
	std::tuple<size_t,size_t,size_t,size_t,size_t,size_t,double,double,double,double,bool,size_t> params1 = algo1.parameters_access();
	std::cout<< "k = " << std::get<0>(params1) << std::endl;
	std::cout<< "q = " << std::get<1>(params1) << std::endl;
	std::cout<< "n_signal = " << std::get<2>(params1) << std::endl;
	std::cout<< "n_starts = " << std::get<3>(params1) << std::endl;
	std::cout<< "epochs = " << std::get<4>(params1) << std::endl;
	std::cout<< "d_intrinsic = " << std::get<5>(params1) << std::endl;
	std::cout<< "sigma2_min = " << std::get<6>(params1) << std::endl;
	std::cout<< "sigma2_max = " << std::get<7>(params1) << std::endl;
	std::cout<< "lambda_min = " << std::get<8>(params1) << std::endl;
	std::cout<< "lambda_max = " << std::get<9>(params1) << std::endl;
	std::cout<< "normalized_det = " << std::get<10>(params1) << std::endl;

	std::cout<< "The number of points for the first algorithm is:"<< std::get<11>(params1) << std::endl;

	std::cout<<std::endl;
	std::cout<<std::endl;
*/

	std::cout<< "The parameters of the second algorithm are:"<< std::endl;
	std::tuple<size_t,size_t,size_t,size_t,size_t,size_t,double,double,double,double,bool,size_t> params2 = algo2.parameters_access();
	std::cout<< "k = " << std::get<0>(params2) << std::endl;
	std::cout<< "q = " << std::get<1>(params2) << std::endl;
	std::cout<< "n_signal = " << std::get<2>(params2) << std::endl;
	std::cout<< "n_starts = " << std::get<3>(params2) << std::endl;
	std::cout<< "epochs = " << std::get<4>(params2) << std::endl;
	std::cout<< "d_intrinsic = " << std::get<5>(params2) << std::endl;
	std::cout<< "sigma2_min = " << std::get<6>(params2) << std::endl;
	std::cout<< "sigma2_max = " << std::get<7>(params2) << std::endl;
	std::cout<< "lambda_min = " << std::get<8>(params2) << std::endl;
	std::cout<< "lambda_max = " << std::get<9>(params2) << std::endl;
	std::cout<< "normalized_det = " << std::get<10>(params2) << std::endl;

	std::cout<< "The number of points for the second algorithm is:"<< std::get<11>(params2) << std::endl;

	std::cout<<std::endl;

	std::cout<< "The first five points of the second algorithm are:" << std::endl;
	std::vector<Matrix<double,1,d>> points_values = algo2.points_values_access();
	for(size_t i = 0; i<5; ++i)
	{
		std::cout<< points_values[i][0] << ", " <<  points_values[i][1]  << std::endl;
	}

	std::cout<<std::endl;

	std::cout<<"On lance l'algorithme !"<< std::endl;
	algo2.run_kPLM();
	algo2.print_optimal_centroids();
	algo2.print_optimal_clusters();
	algo2.print_optimal_costs();


	//std::cout<<"Now we write the labels of the points for algo2,"<< std::endl;
	algo2.write_clusters();
	//std::cout<<"and the costs of the points,"<< std::endl;
	algo2.write_costs();
	//std::cout<<"and the optimal centroids,"<< std::endl;
	algo2.write_centroids();
	//std::cout<<"and their associated optimal matrices."<< std::endl;
	algo2.write_matrices_inv();
/*
	std::cout<<"The optimal cost for algo2 is: "<<algo2.get_optimal_cost()<< std::endl;

	std::cout<<std::endl;

	std::vector<Centroid<d>> _optimal_centroids_ = algo2.get_optimal_centroids();
	std::vector<size_t> _optimal_labels_ = algo2.get_optimal_clusters();
	std::vector<double> _optimal_costs_ = algo2.get_optimal_costs();

	std::cout<<"The labels of the points are :";
	for(typename std::vector<size_t>::iterator it = _optimal_labels_.begin(); it!=_optimal_labels_.end(); ++it){std::cout<<*it<<", ";}
	std::cout<<std::endl;

	std::cout<<"The costs of the points are :";
	for(typename std::vector<double>::iterator it = _optimal_costs_.begin(); it!=_optimal_costs_.end(); ++it){std::cout<<*it<<", ";}
	std::cout<<std::endl;

*/
}


/*	

	vector<Point<d>> points = readcsv<d>(argv[1]);

	size_t n_starts = 10 ;
	size_t epochs = 20 ;
	
	if (argc < 5){std::cout<<"You should enter parameters : "<<std::endl<<"dataset file (.csv),"<<std::endl<<"number of centers k,"<<std::endl<<"number of nearest neighbors q,"<<std::endl<<"number of points considered as signal n_signal."<<std::endl;}
	else{
	size_t k = atoi(argv[2]); // atos ? atosizet ?
	size_t q = atoi(argv[3]);
	size_t n_signal = atoi(argv[4]);





// STEP to parralelize


  // Parallel version
  // number of threads
 	const size_t nthreads = std::thread::hardware_concurrency();
	size_t nstarts_int = n_starts/nthreads;
	double nstarts_double = n_starts/(double)nthreads;
	size_t nstarts = nstarts_int + ((nstarts_double!=(double)nstarts_int)?1:0);
	std::vector<std::tuple<vector<Centroid<d>>,vector<size_t>,vector<double>,double>> ans;

   // Pre loop
    std::vector<std::thread> threads(nthreads);
    std::mutex critical;
    for(size_t t = 0;t<nthreads;t++)
    {
      threads[t] = std::thread(std::bind(
	[&](const vector<Point<d>> & points_, size_t epochs_, size_t k_, size_t q_, size_t n_signal_, size_t n_starts_)
        {
		std::tuple<vector<Centroid<d>>,vector<size_t>,vector<double>,double> this_ans = kPLMTrimClustering<d>(points_,epochs_,k_,q_,n_signal_,n_starts_);
              // (optional) make output critical
              std::lock_guard<std::mutex> lock(critical);
		ans.push_back(this_ans);
		//std::cout<<std::get<3>(this_ans)<<std::endl;
	},points,epochs,k,q,n_signal,nstarts));
    }
    std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
    // Post loop



// END STEP to parralelize



	vector<Centroid<d>> centroids = std::get<0>(ans[0]);
	vector<size_t> labels = std::get<1>(ans[0]);
	vector<double> costs = std::get<2>(ans[0]);
	double cost = std::get<3>(ans[0]);
	for(int i=1; i<nthreads; ++i)
	{
		if(std::get<3>(ans[i])<cost){
			centroids = std::get<0>(ans[i]); 
			labels = std::get<1>(ans[i]);
			costs = std::get<2>(ans[i]);
			cost = std::get<3>(ans[i]);
		}
	}


	if (argc >= 6){ write_clusters(labels,argv[5]);}
	else{write_clusters(labels);}
	if (argc >= 7){ write_costs(costs,argv[6]);}
	else{write_costs(costs);}
	if (argc >= 8){	write_centroids(centroids,argv[7]);}
	else{write_centroids(centroids);}
	if (argc >= 9){	write_matrices_inv(centroids,argv[8]);}
	else{write_matrices_inv(centroids);}

	
	} 

/*
*/

catch(std::invalid_argument const&e)
{
	//cerr << "ERREUR :" << e.what() << endl;
	cerr << e.what() << endl;
}
	return(0);

}



/*
Code R :


a = read.csv("clusters.csv")
a = a[,1]
plot(P,col = a)

b = read.csv("centroids.csv")
points(b[b[,6]==1,1:2],col = (1:(nrow(b)))[b[,6]==1],pch = 20)
points(b[b[,6]==1,1:2],col = "black",pch = 2)

costs = read.csv("costs.csv",sep = " ")
color = costs[,1]
rbPal <- colorRampPalette(c('red','blue'))
Col <- rbPal(10)[as.numeric(cut(color,breaks = 10))]
plot(P,col = Col)


*/


/*

try
{
    kPLM<d> algo(....);
    // do stuff with algo
}
catch(std::invalid_argument const&e)
{
    std::cout << "Construction of algo failed" << std::endl;
	// Ou encore: 
	cerr << "ERREUR :" << e.what() << endl;
}

*/

