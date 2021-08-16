/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Claire Brécheteau
 *
 *    Copyright (C) 2021 Université Rennes 2
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef SIGMA_INVERTED_H_
#define SIGMA_INVERTED_H_

#include <Eigen/Dense> 

using namespace Eigen;

template <size_t d>  
class Centroid;

template <size_t d>
class kPLM;

template <size_t d> 
class Sigma_inverted {

private:
	Matrix<double,d,1> Eigenvalues_inverted_initial; // Contains 1/Eigenvalues with Eigenvalues sorted in non-decreasing order
	Matrix<double,d,1> Eigenvalues_inverted_to_use;
	Matrix<double,d,d> Eigenvectors; // Orthogonal matrix

public: 
	Sigma_inverted();
	Sigma_inverted(const Matrix<double,d,1> & Eigenvalues_inverted);
	Sigma_inverted(const Matrix<double,d,d> & Eigenvectors);
	Sigma_inverted(const Matrix<double,d,1> & Eigenvalues_inverted, const Matrix<double,d,d> & Eigenvectors);
	Sigma_inverted(const Sigma_inverted<d> & Sigma);


	void trunc_eigenvalues(size_t d_intrinsic,double sigma2_min,double sigma2_max,double lambda_min,double lambda_max = __DBL_MAX__); // The d_intrinsic first eigenvalues truncated if larger than lambda_max or smaller than lambda_min ; The d - d_intrinsic eigenvalues are replaced with their mean, truncated if larger than sigma2_max or smaller than sigma2_min;
	void trunc_eigenvalues_sigma2_const(size_t d_intrinsic,double sigma2,double lambda_min,double lambda_max = __DBL_MAX__); // sigma2 is fixed so no need to compute its value...
	void eigenvalues_det_to_1(); // the eigenvalues are divided by their product to the 1/d.


	double mahalanobis_distance(const Matrix<double,1,d> &vect) const; // Squared Mahalanobis norm of vect for the matrix Sigma_inverted : vect.transpose()*Sigma_inverted*vect

	Matrix<double,Dynamic,1> mahalanobis_distance_block(const Matrix<double,Dynamic,d> & block_points, const Matrix<double,1,d> &vect = Matrix<double,1,d>::Zero());
//    	template<size_t n_points>
//	double mahalanobis_distance(const std::vector<Matrix<double,n_points,d> & blocks_points, const Matrix<double,1,d> &vect = Matrix<double,1,d>::Zero());


	Matrix<double,d,d> return_inverse_matrix_to_use() const;
	Matrix<double,d,d> return_matrix_to_use() const;	
	Matrix<double,d,d> return_inverse_matrix_initial() const;
	Matrix<double,d,d> return_matrix_initial() const;

	Matrix<double,d,1> get_eigenvalues_initial();
	Matrix<double,d,1> get_eigenvalues_inverted_initial();
	Matrix<double,d,1> get_eigenvalues_to_use();
	Matrix<double,d,1> get_eigenvalues_inverted_to_use();

	friend class Centroid<d>;
	friend class kPLM<d>;
};


// Functions definitions

template<size_t d>
double Sigma_inverted<d>::mahalanobis_distance(const Matrix<double,1,d> &vect) const{ // can be used with vect1-vect2
	Matrix<double,1,d> temp;
	temp.noalias() = (vect)*this->Eigenvectors;
	double distance = 0;
	for(size_t dd = 0; dd<d ; ++dd){distance+=this->Eigenvalues_inverted_to_use[dd]*temp[dd]*temp[dd];}
	return(distance);
	//return((vect*this->Eigenvectors).cwiseAbs2() * (this->Eigenvalues_inverted_to_use));
};
// C'est pas forcément le plus rapide.



template<size_t d>
Matrix<double,Dynamic,1> Sigma_inverted<d>::mahalanobis_distance_block(const Matrix<double,Dynamic,d> & block_points, const Matrix<double,1,d> &vect){
	Array<double,Dynamic,d> aux = (block_points.rowwise() - vect)*this->Eigenvectors;
	return((aux*aux).matrix()*(this->Eigenvalues_inverted_to_use));
	//return((( block_points.rowwise() - vect)*this->Eigenvectors).cwiseAbs2() * (this->Eigenvalues_inverted_to_use));
};
// C'est pas forcément le plus rapide.
    
/*	
template<size_t d,size_t n_points>
double Sigma_inverted<d>::mahalanobis_distance<n_points>(const std::vector<Matrix<double,n_points,d> & blocks_points, const Matrix<double,1,d> &vect = Matrix<double,1,d>::Zero()){


};
*/




// Constructors :

template<size_t d>
Sigma_inverted<d>::Sigma_inverted() :
	Eigenvectors(Matrix<double,d,d>::Identity()),
	Eigenvalues_inverted_initial(Matrix<double,d,1>::Constant(1)),
	Eigenvalues_inverted_to_use(Matrix<double,d,1>::Constant(1))
	{}


template<size_t d>
Sigma_inverted<d>::Sigma_inverted(const Matrix<double,d,1> & Eigenvalues_inverted) :
	Eigenvectors(Matrix<double,d,d>::Identity()),
	Eigenvalues_inverted_initial(Eigenvalues_inverted),
	Eigenvalues_inverted_to_use(Eigenvalues_inverted)
	{}

template<size_t d>
Sigma_inverted<d>::Sigma_inverted(const Matrix<double,d,d> & Eigenvectors) :
	Eigenvectors(Eigenvectors),
	Eigenvalues_inverted_initial(Matrix<double,d,1>::Constant(1)),
	Eigenvalues_inverted_to_use(Matrix<double,d,1>::Constant(1))
	{}

template<size_t d>
Sigma_inverted<d>::Sigma_inverted(const Matrix<double,d,1> & Eigenvalues_inverted, const Matrix<double,d,d> & Eigenvectors) :
	Eigenvectors(Eigenvectors),
	Eigenvalues_inverted_initial(Eigenvalues_inverted),
	Eigenvalues_inverted_to_use(Eigenvalues_inverted)
	{}

template<size_t d>
Sigma_inverted<d>::Sigma_inverted(const Sigma_inverted<d> & Sigma) :
	Eigenvectors(Sigma.Eigenvectors),
	Eigenvalues_inverted_initial(Sigma.Eigenvalues_inverted_initial),
	Eigenvalues_inverted_to_use(Sigma.Eigenvalues_inverted_to_use)
	{}

// Modify eigenvalues functions definition :

template <size_t d> 
void Sigma_inverted<d>::trunc_eigenvalues(size_t d_intrinsic,double sigma2_min,double sigma2_max,double lambda_min,double lambda_max){
	double inv_lambda_min = 1/lambda_min;
	double inv_lambda_max = 1/lambda_max;
	for(size_t dd = d - d_intrinsic; dd<d ; ++dd)
	{
		this-> Eigenvalues_inverted_to_use[dd] = (this-> Eigenvalues_inverted_initial[dd] - inv_lambda_max)*(this-> Eigenvalues_inverted_initial[dd] > inv_lambda_max) + inv_lambda_max + (inv_lambda_min - this-> Eigenvalues_inverted_initial[dd])*(this-> Eigenvalues_inverted_initial[dd] > inv_lambda_min);
	}
	double S = 0;
	{
		for(size_t dd = 0; dd<d - d_intrinsic ; ++dd)
		{
			S += 1/(this-> Eigenvalues_inverted_initial[dd]);
		}
		S /= (d-d_intrinsic);
		double inv_sigma2 = 1/((S - sigma2_min)*(S>sigma2_min) + (sigma2_max - S)*(S>sigma2_max) + sigma2_min);
		for(size_t dd = 0; dd<d - d_intrinsic ; ++dd)
		{
			this-> Eigenvalues_inverted_to_use[dd] = inv_sigma2;
		}		
	}
}

template <size_t d> 
void Sigma_inverted<d>::trunc_eigenvalues_sigma2_const(size_t d_intrinsic,double sigma2, double lambda_min,double lambda_max){
	double inv_lambda_min = 1/lambda_min;
	double inv_lambda_max = 1/lambda_max;
	for(size_t dd = d - d_intrinsic; dd< d ; ++dd)
	{
		this-> Eigenvalues_inverted_to_use[dd] = (this-> Eigenvalues_inverted_initial[dd] - inv_lambda_max)*(this-> Eigenvalues_inverted_initial[dd] > inv_lambda_max) + inv_lambda_max + (inv_lambda_min - this-> Eigenvalues_inverted_initial[dd])*(this-> Eigenvalues_inverted_initial[dd] > inv_lambda_min);
	}
	double inv_sigma2 = 1/sigma2;
	for(size_t dd = 0; dd<d - d_intrinsic ; ++dd)
	{
		this-> Eigenvalues_inverted_to_use[dd] = inv_sigma2;
	}
}

template <size_t d> 
void Sigma_inverted<d>::eigenvalues_det_to_1(){
	double product_eigenvalues = 1;
	for(size_t dd = 0; dd<d; ++dd)
	{
		product_eigenvalues*=this-> Eigenvalues_inverted_initial[dd];
	}
	this-> Eigenvalues_inverted_to_use = (this-> Eigenvalues_inverted_initial)/pow(product_eigenvalues ,1/double(d));	
}


// Return matrix functions definition :

template <size_t d> 
Matrix<double,d,d> Sigma_inverted<d>::return_inverse_matrix_to_use() const{
	Matrix<double,d,d> diag = Matrix<double,d,d>::Zero();
	for(size_t dd = 0 ;  dd < d ; ++dd){
		diag(dd,dd) = this->Eigenvalues_inverted_to_use[dd];
	}
	return(this->Eigenvectors * diag * this->Eigenvectors.transpose());
}

template <size_t d> 
Matrix<double,d,d> Sigma_inverted<d>::return_matrix_to_use() const{
	Matrix<double,d,d> diag = Matrix<double,d,d>::Zero();
	for(size_t dd = 0 ;  dd < d ; ++dd){
		diag(dd,dd) = 1/(this->Eigenvalues_inverted_to_use[dd]);
	}
	return(this->Eigenvectors * diag * this->Eigenvectors.transpose());
}

template <size_t d> 
Matrix<double,d,d> Sigma_inverted<d>::return_inverse_matrix_initial() const{
	Matrix<double,d,d> diag = Matrix<double,d,d>::Zero();
	for(size_t dd = 0 ;  dd < d ; ++dd){
		diag(dd,dd) = this->Eigenvalues_inverted_initial[dd];
	}
	return(this->Eigenvectors * diag * this->Eigenvectors.transpose());
}

template <size_t d> 
Matrix<double,d,d> Sigma_inverted<d>::return_matrix_initial() const{
	Matrix<double,d,d> diag = Matrix<double,d,d>::Zero();
	for(size_t dd = 0 ;  dd < d ; ++dd){
		diag(dd,dd) = 1/(this->Eigenvalues_inverted_initial[dd]);
	}
	return(this->Eigenvectors * diag * this->Eigenvectors.transpose());
}

// Get eigenvalues functions definition :

template <size_t d> 
Matrix<double,d,1> Sigma_inverted<d>::get_eigenvalues_initial(){
	Matrix<double,d,1> eigen;
	for(size_t dd = 0 ;  dd < d ; ++dd){
		eigen(dd) = 1/(this->Eigenvalues_inverted_initial[dd]);
	}
	return(eigen);
}

template <size_t d> 
Matrix<double,d,1> Sigma_inverted<d>::get_eigenvalues_inverted_initial(){
	return(this->Eigenvalues_inverted_initial);
}

template <size_t d> 
Matrix<double,d,1> Sigma_inverted<d>::get_eigenvalues_to_use(){
	Matrix<double,d,1> eigen;
	for(size_t dd = 0 ;  dd < d ; ++dd){
		eigen(dd) = 1/(this->Eigenvalues_inverted_to_use[dd]);
	}
	return(eigen);
}

template <size_t d> 
Matrix<double,d,1> Sigma_inverted<d>::get_eigenvalues_inverted_to_use(){
	return(this->Eigenvalues_inverted_to_use);
}






#endif  // SIGMA_INVERTED_H_

