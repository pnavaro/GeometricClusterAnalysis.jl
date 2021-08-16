#include <iostream>  // for file-reading
#include <numeric> // for iota function

#include "../include/kPLM/Point.h"

int main(int argc, char** argv){

	const size_t d = 6;

	std::vector<Point<d>> points = readcsv<d>(argv[1]);
	std::vector<size_t> indices(points.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::cout<< points[0].X << std::endl;

	auto mv = calcul_mean_cov(points,indices);
	std::cout<<"The mean of the sample of points is :"<<std::endl; // mean as returned by the R function mean.
	std::cout<< std::get<0>(mv) << std::endl;
	std::cout<<"The covariance matrix of the sample of points is :"<<std::endl;
	std::cout<< std::get<1>(mv) << std::endl;
	std::cout<<"The non-biased covariance matrix of the sample of points is :"<<std::endl; // covariance as returned by the R function cov.
	std::cout<< points.size()/((double)(points.size() - 1))*std::get<1>(mv) << std::endl;

	std::cout<<std::endl;
	std::cout<<std::endl;

	Matrix<double,Dynamic,d> block = block_points(points);
	auto mv2 = calcul_mean_cov<d>(block,indices);
	std::cout<<"The mean of the sample of points is :"<<std::endl; // mean as returned by the R function mean.
	std::cout<< std::get<0>(mv2) << std::endl;
	std::cout<<"The covariance matrix of the sample of points is :"<<std::endl;
	std::cout<< std::get<1>(mv2) << std::endl;
	std::cout<<"The non-biased covariance matrix of the sample of points is :"<<std::endl; // covariance as returned by the R function cov.
	std::cout<< points.size()/((double)(points.size() - 1))*std::get<1>(mv2) << std::endl;

	return(0);
}
