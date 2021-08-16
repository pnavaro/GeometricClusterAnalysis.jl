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


// Different centroids from nothing, or X and/or cov :

	Point<d> p(X);

	Centroid<d> centroid5(p,Eigenvalues_inverted, Eigenvectors);
	std::cout<< "The initial eigenvalues are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_initial().transpose()<< std::endl;
	std::cout<< "The eigenvalues to use are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_to_use().transpose()<< std::endl;

	std::cout<<std::endl;
	std::cout<<"We update everything"<<std::endl;
	std::cout<<std::endl;

	std::vector<double> kPLM_values;
	double val;


	std::cout<<std::endl;
	std::cout<<std::endl;

// We update everything with method 0 :

	centroid5.update_Voronoi_points_indices(indices);

	size_t q = 10;
	size_t method = 0;

	centroid5.Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method = method);
	centroid5.Centroid<d>::update_m_v(block,q);
	std::cout<< "The initial eigenvalues are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_initial().transpose()<< std::endl;
	std::cout<< "The eigenvalues to use are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_to_use().transpose()<< std::endl;

	kPLM_values.resize(0);
	val = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!= points.end(); ++it)
	{
		val = centroid5.distance_kPLM(*it);
		kPLM_values.push_back(val);
		std::cout<< val<< " ";
	}
	std::cout<<std::endl;
	std::cout<<std::endl;

// We update everything with method 1 - from new matrix:

	centroid5.update_Voronoi_points_indices(indices);

	method = 1;

	size_t d_intrinsic = 4;
	double sigma2_min = 0.01;
	double sigma2_max=0.2;
	double lambda_min = 0.2;
	double lambda_max = 3.5;

	centroid5.Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method = method,d_intrinsic = d_intrinsic,sigma2_min = sigma2_min, sigma2_max = sigma2_max, lambda_min = lambda_min, lambda_max = lambda_max);
	centroid5.Centroid<d>::update_m_v(block,q);
	std::cout<< "The initial eigenvalues are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_initial().transpose()<< std::endl;
	std::cout<< "The eigenvalues to use are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_to_use().transpose()<< std::endl;

	kPLM_values.resize(0);
	val = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!= points.end(); ++it)
	{
		val = centroid5.distance_kPLM(*it);
		kPLM_values.push_back(val);
		std::cout<< val<< " ";
	}
	std::cout<<std::endl;
	std::cout<<std::endl;

// We update everything with method 2 - from new matrix:

	centroid5.update_Voronoi_points_indices(indices);

	method = 2;

	centroid5.Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method = method,d_intrinsic = d_intrinsic,sigma2_min = sigma2_min, sigma2_max = sigma2_max, lambda_min = lambda_min, lambda_max = lambda_max);
	centroid5.Centroid<d>::update_m_v(block,q);
	std::cout<< "The initial eigenvalues are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_initial().transpose()<< std::endl;
	std::cout<< "The eigenvalues to use are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_to_use().transpose()<< std::endl;

	kPLM_values.resize(0);
	val = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!= points.end(); ++it)
	{
		val = centroid5.distance_kPLM(*it);
		kPLM_values.push_back(val);
		std::cout<< val<< " ";
	}
	std::cout<<std::endl;
	std::cout<<std::endl;

// We update everything with method 3 - from new matrix:

	centroid5.update_Voronoi_points_indices(indices);

	method = 3;

	centroid5.Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method = method);
	centroid5.Centroid<d>::update_m_v(block,q);
	std::cout<< "The initial eigenvalues are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_initial().transpose()<< std::endl;
	std::cout<< "The eigenvalues to use are :" << std::endl;
	std::cout<< centroid5.get_eigenvalues_to_use().transpose()<< std::endl;

	kPLM_values.resize(0);
	val = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!= points.end(); ++it)
	{
		val = centroid5.distance_kPLM(*it);
		kPLM_values.push_back(val);
		std::cout<< val<< " ";
	}
	std::cout<<std::endl;



	// Other parameters : size_t d_intrinsic = 0,double sigma2_min = 1,double sigma2_max=1,double lambda_min = 1,double lambda_max = __DBL_MAX__




}
