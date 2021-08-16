#include <iostream>  // for file-reading
#include <numeric> // for iota function

#include "../include/kPLM/Point.h"
#include "../include/kPLM/Sigma_inverted.h"

int main(int argc, char** argv){

	const size_t d = 6; // d has to be the dimension of the points writen in file argv[1] !
	const size_t n_points = 550; // sample size !

	std::vector<Point<d>> points = readcsv<d>(argv[1]);
	std::vector<size_t> indices(points.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::cout<<"The number of points is :"<< points.size()<<std::endl;

	auto mv = calcul_mean_cov(points,indices);
	std::cout<<"The mean of the sample of points is :"<<std::endl; // mean as returned by the R function mean.
	std::cout<< std::get<0>(mv) << std::endl;
	std::cout<<"The covariance matrix of the sample of points is :"<<std::endl;
	std::cout<< std::get<1>(mv) << std::endl;


	std::cout<< std::endl;
	std::cout<< std::endl;


	SelfAdjointEigenSolver<Matrix<double,d,d>> eigensolver(std::get<1>(mv));
	Matrix<double,d,d> Eigenvectors = eigensolver.eigenvectors();
	Matrix<double,1,d> Eigenvalues = eigensolver.eigenvalues();
	Matrix<double,1,d> Eigenvalues_inverted;
	for(size_t dd = 0; dd<d; ++dd){ Eigenvalues_inverted(dd) = 1/Eigenvalues(dd); }


	Sigma_inverted<d> S(Eigenvalues_inverted, Eigenvectors);


	auto m = std::get<0>(mv); 

// Method 1 :
	std::cout<<"Mahalanobis squared norm - method 1 :"<<std::endl;
	std::vector<double> ans1(0);
	for(std::vector<Point<d>>::iterator it = points.begin(); it!=points.end(); ++it)
	{
		Matrix<double,1,d> aux = it->get_X() - m;
		ans1.push_back(S.mahalanobis_distance(aux));
	}

	for(std::vector<double>::iterator it = ans1.begin(); it!=ans1.end(); ++it)
	{
		std::cout <<*it<< ", " ;
	}
		
	std::cout<<std::endl;
	std::cout<<std::endl;


// Method 2 :
	Matrix<double,Dynamic,d> block_points(points.size(), d) ;
	size_t index = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!=points.end(); ++it)
	{
		block_points.row(index) << it->get_X();
		++index;	
	}
	auto maha_distances = S.mahalanobis_distance_block(block_points, m);
	std::cout << maha_distances.transpose() << std::endl;

	
	double difference = 0;
	for(size_t i= 0; i<ans1.size(); ++i)
	{
		difference += (maha_distances[i] - ans1[i])*(maha_distances[i] - ans1[i]);
	}

	std::cout<<std::endl;
	std::cout<< "The difference between the two methods equals "<<difference<<std::endl;
	


	return(0);
}
