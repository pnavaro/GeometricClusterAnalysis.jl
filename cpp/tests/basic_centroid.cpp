#include <iostream>  // for file-reading

#include "../include/kPLM/Centroid.h"



/// CAUTION : IN PRACTICE, kPLM may be used after a call of update_m_v ; and not after a call of update_active_X_m_Sigma_inverted !!!!!!

/// Indeed, if we call kPLM after update_active_X_m_Sigma_inverted, we do not compute the true cost kPLM at points (since we use the old covariance matrix for m and most of all for v, whereas we use the new covariance matrix for the Mahalanobis norm between the point and m) !



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

	Centroid<d> centroid1;
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid1.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid1.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid1.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid1.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid1.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid1.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid1.get_v()<< std::endl;
	std::vector<size_t> Vor = centroid1.get_Voronoi_points_indices();
	std::cout<< "The indices of the points in the Voronoi cell are :";
	for(std::vector<size_t>::iterator it = Vor.begin(); it!=Vor.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid1.get_active()<< std::endl;

	std::cout<<std::endl;
	std::cout<<std::endl;

	double tab[d]{1,2,5};
	Centroid<d> centroid2(tab);
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid2.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid2.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid2.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid2.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid2.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid2.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid2.get_v()<< std::endl;
	std::vector<size_t> Vor2 = centroid2.get_Voronoi_points_indices();
	std::cout<< "The indices of the points in the Voronoi cell are :";
	for(std::vector<size_t>::iterator it = Vor2.begin(); it!=Vor2.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid2.get_active()<< std::endl;

	std::cout<<std::endl;
	std::cout<<std::endl;

	Point<d> p(X);
	Centroid<d> centroid3(p);
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid3.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid3.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid3.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid3.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid3.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid3.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid3.get_v()<< std::endl;
	std::vector<size_t> Vor3 = centroid3.get_Voronoi_points_indices();
	std::cout<< "The indices of the points in the Voronoi cell are :";
	for(std::vector<size_t>::iterator it = Vor3.begin(); it!=Vor3.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid3.get_active()<< std::endl;

	std::cout<<std::endl;
	std::cout<<std::endl;

	Sigma_inverted<d> Sigma(Eigenvalues_inverted, Eigenvectors);
	Centroid<d> centroid4(p, Sigma);
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid4.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid4.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid4.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid4.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid4.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid4.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid4.get_v()<< std::endl;
	std::vector<size_t> Vor4 = centroid4.get_Voronoi_points_indices();
	std::cout<< "The indices of the points in the Voronoi cell are :";
	for(std::vector<size_t>::iterator it = Vor4.begin(); it!=Vor4.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid4.get_active()<< std::endl;

	std::cout<<std::endl;
	std::cout<<std::endl;

	Centroid<d> centroid5(p,Eigenvalues_inverted, Eigenvectors);
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid5.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid5.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid5.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid5.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid5.get_v()<< std::endl;
	std::vector<size_t> Vor5 = centroid5.get_Voronoi_points_indices();
	std::cout<< "The indices of the points in the Voronoi cell are :";
	for(std::vector<size_t>::iterator it = Vor5.begin(); it!=Vor5.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid5.get_active()<< std::endl;


	std::cout<<std::endl;
	std::cout<<"We update m and v"<<std::endl;
	std::cout<<std::endl;

// We use the last centroid, and update m and v :

	size_t q = 1;
	centroid5.Centroid<d>::update_m_v(block,q);


	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid5.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid5.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid5.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid5.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid5.get_v()<< std::endl;
	std::cout<< "The indices of the points in the Voronoi cell are :";
	auto Vor5_1 = centroid5.get_Voronoi_points_indices();
	for(std::vector<size_t>::iterator it = Vor5_1.begin(); it!=Vor5_1.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid5.get_active()<< std::endl;


	std::cout<<std::endl;
	std::cout<<"We update everything"<<std::endl;
	std::cout<<std::endl;

// We update the indices of points in the Voronoi cell (we take every points) :


	centroid5.update_Voronoi_points_indices(indices);


// Now we update everything :

	size_t method = 0;

	centroid5.Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method = 0);
	// Other parameters : size_t d_intrinsic = 0,double sigma2_min = 1,double sigma2_max=1,double lambda_min = 1,double lambda_max = __DBL_MAX__

	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid5.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid5.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid5.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid5.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid5.get_v()<< std::endl;
	std::cout<< "The indices of the points in the Voronoi cell are :";
	auto Vor5_2 = centroid5.get_Voronoi_points_indices();
	for(std::vector<size_t>::iterator it = Vor5_2.begin(); it!=Vor5_2.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid5.get_active()<< std::endl;


	std::cout<<std::endl;
	std::cout<<"We make centroid1 to be centroid5"<<std::endl;
	std::cout<<std::endl;

// We make centroid1 become centroid5 :

	centroid1.equals(centroid5);
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid1.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid1.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid1.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid1.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid1.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid1.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid1.get_v()<< std::endl;
	std::cout<< "The indices of the points in the Voronoi cell are :";
	auto Vor5_3 = centroid1.get_Voronoi_points_indices();
	for(std::vector<size_t>::iterator it = Vor5_3.begin(); it!=Vor5_3.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid1.get_active()<< std::endl;


	std::cout<<std::endl;
	std::cout<<"We compute all kPLM values"<<std::endl;
	std::cout<<std::endl;

// We compute all the kPLM distance, at every point in the sample :

	std::vector<double> kPLM_values(0);
	double val = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!= points.end(); ++it)
	{
		val = centroid5.distance_kPLM(*it);
		kPLM_values.push_back(val);
		std::cout<< val<< " ";
	}
	std::cout<<std::endl;

	std::cout<<std::endl;
	std::cout<<"We do the same think with parameter q = 20 (from the updated covariance matrix)"<<std::endl;
	std::cout<<std::endl;

// We now have a larger parameter q :

	q = 20;
	centroid5.Centroid<d>::update_m_v(block,q);
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid5.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid5.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid5.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid5.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid5.get_v()<< std::endl;
	std::cout<< "The indices of the points in the Voronoi cell are :";
	auto Vor5_4 = centroid5.get_Voronoi_points_indices();
	for(std::vector<size_t>::iterator it = Vor5_4.begin(); it!=Vor5_4.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid5.get_active()<< std::endl;


	std::cout<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<std::endl;

// Now we update everything :

	method = 0;

	centroid5.Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method = 0);
	// Other parameters : size_t d_intrinsic = 0,double sigma2_min = 1,double sigma2_max=1,double lambda_min = 1,double lambda_max = __DBL_MAX__

	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid5.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid5.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid5.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid5.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid5.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid5.get_v()<< std::endl;
	std::cout<< "The indices of the points in the Voronoi cell are :";
	auto Vor5_5 = centroid5.get_Voronoi_points_indices();
	for(std::vector<size_t>::iterator it = Vor5_5.begin(); it!=Vor5_5.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid5.get_active()<< std::endl;


	std::cout<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<std::endl;

// We make centroid1 become centroid5 :

	centroid1.equals(centroid5);
	std::cout<< "The inverse of the matrix to use is :" << std::endl;
	std::cout<< centroid1.get_inverse_matrix_to_use()<< std::endl;
	std::cout<< "The matrix to use is :" << std::endl;
	std::cout<< centroid1.get_matrix_to_use()<< std::endl;
	std::cout<< "The inverse of the initial matrix is :" << std::endl;
	std::cout<< centroid1.get_inverse_matrix_initial()<< std::endl;
	std::cout<< "The initial matrix is :" << std::endl;
	std::cout<< centroid1.get_matrix_initial()<< std::endl;
	std::cout<< "The center is :" << centroid1.get_X()<< std::endl;
	std::cout<< "Its mean is :" << centroid1.get_m()<< std::endl;
	std::cout<< "Its weight is :" << centroid1.get_v()<< std::endl;
	std::cout<< "The indices of the points in the Voronoi cell are :";
	auto Vor5_6 = centroid5.get_Voronoi_points_indices();
	for(std::vector<size_t>::iterator it = Vor5_6.begin(); it!=Vor5_6.end(); ++it){ std::cout << *it<< ", ";}
	std::cout<<std::endl;
	std::cout<< "This point is active (if 1) :" << centroid1.get_active()<< std::endl;


// We compute all the kPLM distance, at every point in the sample :

	kPLM_values.resize(0);
	val = 0;
	for(std::vector<Point<d>>::iterator it = points.begin(); it!= points.end(); ++it)
	{
		val = centroid5.distance_kPLM(*it);
		kPLM_values.push_back(val);
		std::cout<< val<< " ";
	}
	std::cout<<std::endl;

/*
*/
}
