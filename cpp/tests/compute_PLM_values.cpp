#include <iostream>  // for file-reading

#include "../include/kPLM/Centroid.h"


int main(int argc, char** argv){

	const size_t d = 6;


// Get sample of point, compute its covariance matrix.

	std::vector<Point<d>> points = readcsv<d>(argv[1]);

	Matrix<double,Dynamic,d> block = block_points(points);

	std::vector<size_t> indices(points.size());
	std::iota(indices.begin(), indices.end(), 0);

	auto mv = calcul_mean_cov(points,indices);
	Matrix<double,d,d> cov =  std::get<1>(mv);
	Matrix<double,1,d> X =  std::get<0>(mv);

	SelfAdjointEigenSolver<Matrix<double,d,d>> eigensolver(cov);
	Matrix<double,1,d> Eigenvalues_inverted;
	for(size_t dd = 0; dd<d; ++dd){ Eigenvalues_inverted(dd) = 1/eigensolver.eigenvalues()(dd); }
	Matrix<double,d,d> Eigenvectors = eigensolver.eigenvectors();

	Point<d> p(X);

	Centroid<d> centroid(p,Eigenvalues_inverted, Eigenvectors);

	std::vector<double> kPLM_values;
	double val;

	centroid.update_Voronoi_points_indices(indices);

	size_t q = 10;
	size_t method = 0;

	centroid.Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method = method);
	centroid.Centroid<d>::update_m_v(block,q);

	kPLM_values.resize(0);
	val = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!= points.end(); ++it)
	{
		val = centroid.distance_kPLM(*it);
		kPLM_values.push_back(val);
		std::cout<< val<< " ";
	}
	std::cout<<std::endl;
	std::cout<<std::endl;


	std::cout<<"We compute the kPLM again, with blocks of points"<<std::endl;

	Array<double,Dynamic,1> kPLM_values2 = centroid.distance_kPLM_block(block);
	std::cout<<"The kPLM values are:"<<std::endl;
	std::cout<<kPLM_values2.transpose() <<std::endl;

	std::cout<<"The block of points is:"<<std::endl;
	std::cout<< block<<std::endl;

}
