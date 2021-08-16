#include <iostream>  // for file-reading
#include <numeric> // for iota function

#include "../include/kPLM/Point.h"
#include "../include/kPLM/Sigma_inverted.h"

int main(int argc, char** argv){

	const size_t d = 6; // d has to be the dimension of the points writen in file argv[1] !

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


// Eigenvalues
	std::cout<<"The eigenvalues are :"<<std::endl; 
	std::cout<< S.get_eigenvalues_initial().transpose() <<std::endl;
	std::cout<<"They are sorted in non-decreasing order"<<std::endl; 

// Inverse of the eigenvalues
	std::cout<<"The inverse of the eigenvalues are :"<<std::endl; 
	std::cout<< S.get_eigenvalues_inverted_initial().transpose() <<std::endl;
	std::cout<<"They are sorted in non-increasing order"<<std::endl; 

// Recover matrix
	std::cout<< "We recover the initial covariance matrix :"<< std::endl;
	std::cout<< S.return_matrix_initial()<<std::endl;

// Inverse of the matrix
	std::cout<< "Its inverse matrix is :"<< std::endl;
	std::cout<< S.return_inverse_matrix_initial()<<std::endl;

	std::cout<< std::endl;
	std::cout<< std::endl;


// Modification of the eigenvalues : 
	size_t d_intrinsic = 6; // 2 eigenvalues are different, the 4 smallest eigenvalues are all equal.
	double sigma2_min = 0.01;
	double sigma2_max = 0.1;
	double lambda_min = 2;
	double lambda_max = 4;
	std::cout<< "We modify the eigenvalues with trunc_eigenvalues function, with parameters : d_intrinsic = "<<d_intrinsic<<" , sigma2_min = "<<sigma2_min<<", sigma2_max = "<<sigma2_max<<", lambda_min = "<<lambda_min<<", lambda_max = "<<lambda_max<<"."<<std::endl;
	S.trunc_eigenvalues(d_intrinsic,sigma2_min,sigma2_max,lambda_min,lambda_max);
	std::cout<<"The eigenvalues are now:"<<std::endl; 
	std::cout<< S.get_eigenvalues_to_use().transpose() <<std::endl;
	std::cout<<"The inverse of the eigenvalues are :"<<std::endl; 
	std::cout<< S.get_eigenvalues_inverted_to_use().transpose() <<std::endl;
	std::cout<< "We compute the covariance matrix after modification of the eigenvalues :"<< std::endl;
	std::cout<< S.return_matrix_to_use()<<std::endl;
	std::cout<< "Its inverse matrix is :"<< std::endl;
	std::cout<< S.return_inverse_matrix_to_use()<<std::endl;

	std::cout<< std::endl;
	std::cout<< std::endl;


// Modification of the eigenvalues - sigma constant : 
	std::cout<< "We modify the eigenvalues with trunc_eigenvalues function - the constant sigma2 is "<<sigma2_max<<std::endl;
	S.trunc_eigenvalues_sigma2_const(d_intrinsic,sigma2_max,lambda_min,lambda_max);
	std::cout<<"The eigenvalues are now:"<<std::endl; 
	std::cout<< S.get_eigenvalues_to_use().transpose() <<std::endl;
	std::cout<<"The inverse of the eigenvalues are :"<<std::endl; 
	std::cout<< S.get_eigenvalues_inverted_to_use().transpose() <<std::endl;
	std::cout<< "We compute the covariance matrix after modification of the eigenvalues :"<< std::endl;
	std::cout<< S.return_matrix_to_use()<<std::endl;
	std::cout<< "Its inverse matrix is :"<< std::endl;
	std::cout<< S.return_inverse_matrix_to_use()<<std::endl;

	std::cout<< std::endl;
	std::cout<< std::endl;


// Modification of the eigenvalues - determinant equals 1 : 
	std::cout<< "We modify the eigenvalues with trunc_eigenvalues function"<<std::endl;
	S.eigenvalues_det_to_1();
	std::cout<<"The eigenvalues are now:"<<std::endl; 
	std::cout<< S.get_eigenvalues_to_use().transpose() <<std::endl;
	std::cout<<"The inverse of the eigenvalues are :"<<std::endl; 
	std::cout<< S.get_eigenvalues_inverted_to_use().transpose() <<std::endl;
	std::cout<< "We compute the covariance matrix after modification of the eigenvalues :"<< std::endl;
	std::cout<< S.return_matrix_to_use()<<std::endl;
	std::cout<< "Its inverse matrix is :"<< std::endl;
	std::cout<< S.return_inverse_matrix_to_use()<<std::endl;


	std::cout<< std::endl;
	std::cout<< std::endl;


//Check that the eigenvalues_inverted_initial are not modified :
	std::cout<<"The initial eigenvalues are still :"<<std::endl; 
	std::cout<< S.get_eigenvalues_initial().transpose() <<std::endl;

	std::cout<< std::endl;
	std::cout<< std::endl;

// Finally, we compute the Mahalanobis squared norm of the mean :
	std::cout<<"The Mahalanobis squared norm of the mean of the sample, for the matrix constrained with determinant 1 is :"<<std::endl;
	std::cout << S.mahalanobis_distance(std::get<0>(mv)) << std::endl;


	return(0);
}
