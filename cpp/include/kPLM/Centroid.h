/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Claire Brécheteau
 *
 *    Copyright (C) 2021 Université Rennes 2
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */


//Caution : if update_active_X_m_Sigma_inverted is called, then update_m_v must be called just before distance_kPLM ! If not, the weight v does not correspond to the matrix Sigma !


#ifndef CENTROID_H_
#define CENTROID_H_

#include "Point.h"
#include "Sigma_inverted.h"

#include <numeric> // for iota function

using namespace Eigen;

template <size_t d>
class kPLM;

// class definition

template <size_t d> 
class Centroid {

private:
	Matrix<double,1,d> X;     // coordinates (the mean of the points of indices in Voronoi_points_indices)
	std::vector<size_t> Voronoi_points_indices;
	Sigma_inverted<d> Sigma_inv;
	// We consider the matrix Sigma^-1 = Eigenvectors Diag(Eigenvalues_inverted) Eigenvectors^T
	Matrix<double,1,d> m;	// mean of the nearest neighbors of X
	double v;	// variance of the nearest neighbors for the matrix Sigma_inv + log(det(Sigma))
	bool active;	// true if the associated Voronoi cell is non-empty ; true is default value



public:
	Centroid();
	Centroid(const double (&tab)[d]);
	Centroid(const Point<d> &p);
	Centroid(const Point<d> &p, const Sigma_inverted<d> &Sigma);
	Centroid(const Point<d> &p,const Matrix<double,d,1> & Eigenvalues_inverted, const Matrix<double,d,d> & Eigenvectors);
	// All constructors call the constructor of Sigma_inverted<d>; The matrix Sigma_inv is the Identity matrix when no parameter is given for the matrix.

	double distance_kPLM(const Point<d> &p) const; // squared Mahalanobis to the mean + variance + log(det)
	Array<double,Dynamic,1> distance_kPLM_block (const Matrix<double,Dynamic,d> & block_points); // squared Mahalanobis to the mean + variance + log(det)

	void update_m_v(const Matrix<double,Dynamic,d> & block,size_t q);
	void update_active_X_m_Sigma_inverted(const Matrix<double,Dynamic,d> & block,size_t q,size_t method = 0,size_t d_intrinsic = 0,double sigma2_min = 1,double sigma2_max=1,double lambda_min = 1,double lambda_max = __DBL_MAX__);
	void activate();
	void desactivate();
	void equals (const Centroid<d>& c);

	void update_Voronoi_points_indices(const std::vector<size_t> & indices);

	Matrix<double,d,d> get_inverse_matrix_to_use() const;
	Matrix<double,d,d> get_matrix_to_use() const;	
	Matrix<double,d,d> get_inverse_matrix_initial() const;
	Matrix<double,d,d> get_matrix_initial() const;
	Matrix<double,1,d> get_X() const;
	std::vector<size_t> get_Voronoi_points_indices() const;
	Matrix<double,1,d> get_m() const;
	double get_v() const;
	bool get_active() const;
	Matrix<double,d,1> get_eigenvalues_initial() const;
	Matrix<double,d,1> get_eigenvalues_to_use() const;

	// friend class Point<d>;
	friend class kPLM<d>;
	// friend class Sigma_inverted<d>;
};

// functions definitions


// Ajouter un constructeur avec un centre et une matrice de covariance éventuellement

template <size_t d> 
Centroid<d>::Centroid() :
	v(__DBL_MAX__),
	Voronoi_points_indices(0),
	active(true),
	m(Matrix<double,1,d>::Zero()),
	X(Matrix<double,1,d>::Zero())
	{
	}

template <size_t d> 
Centroid<d>::Centroid(const double (&tab)[d]) :
	v(__DBL_MAX__),
	Voronoi_points_indices(0),
	active(true),
	m(Matrix<double,1,d>::Zero())
	{
		for(size_t dd = 0 ; dd < d ; ++dd){
			X[dd] = tab[dd];
		}
	}

template <size_t d> 
Centroid<d>::Centroid(const Point<d> &p) :
	v(__DBL_MAX__),
	Voronoi_points_indices(0),
	active(true),
	m(Matrix<double,1,d>::Zero())
	{
		for(size_t dd = 0 ; dd < d ; ++dd){
			X[dd] = p.X[dd];
		}
	}

template <size_t d> 
Centroid<d>::Centroid(const Point<d> &p, const Sigma_inverted<d> &Sigma) :
	v(__DBL_MAX__),
	Voronoi_points_indices(0),
	active(true),
	Sigma_inv(Sigma),
	m(Matrix<double,1,d>::Zero())
	{
		for(size_t dd = 0 ; dd < d ; ++dd){
			X[dd] = p.X[dd];
		}
	}


template <size_t d> 
Centroid<d>::Centroid(const Point<d> &p,const Matrix<double,d,1> & Eigenvalues_inverted, const Matrix<double,d,d> & Eigenvectors) :
	v(__DBL_MAX__),
	Voronoi_points_indices(0),
	active(true),
	Sigma_inv(Eigenvalues_inverted,Eigenvectors),
	m(Matrix<double,1,d>::Zero())
	{
		for(size_t dd = 0 ; dd < d ; ++dd){
			X[dd] = p.X[dd];
		}
	}

//////// FONCTION A MODIFIER (prochaine version)

template <size_t d> 
double Centroid<d>::distance_kPLM(const Point<d> &p) const { // Distance for weighted Voronoi cells for the kPLM
	return(this->Sigma_inv.mahalanobis_distance(this->m - p.X)+this->v);
} 

template <size_t d>
Array<double,Dynamic,1> Centroid<d>::distance_kPLM_block(const Matrix<double,Dynamic,d> & block_points) {
	Matrix<double,Dynamic,1> aux = (this->Sigma_inv).mahalanobis_distance_block(block_points,this->m);
	return(aux.array()+this->v);
}

template <size_t d> 
void Centroid<d>::update_m_v(const Matrix<double,Dynamic,d> & block,size_t q){

	Array<double,Dynamic,1> Dist_to_center = (this->Sigma_inv).mahalanobis_distance_block(block,this->X);
	//for (typename vector<Point<d>>::const_iterator it = points.begin(); it != points.end(); ++it) {
	//	Dist_to_center.push_back(this->Sigma_inv.mahalanobis_distance(it->X - this->X));
	//}

	// Indices of the q closest points to this->X, in points, for the Mahalanobis norm.
	vector<size_t> indices_ellipsoid(block.rows());
	iota(indices_ellipsoid.begin(), indices_ellipsoid.end(), 0);
	stable_sort(indices_ellipsoid.begin(), indices_ellipsoid.end(), [&Dist_to_center](size_t i1, size_t i2) {return Dist_to_center[i1] < Dist_to_center[i2];});

	indices_ellipsoid.resize(q);

	// Mean of these points.
	Matrix<double,1,d> mean(Matrix<double,1,d>::Zero());
	for (typename std::vector<size_t>::const_iterator it = indices_ellipsoid.begin() ; it != indices_ellipsoid.end() ; ++it) {
	    mean += block.row(*it) ;
	}
	mean/= q;
	this-> m =  mean;

	// Associated weight.

	//Array<double,Dynamic,d> Dist_for_weight_v = (this->Sigma_inv).mahalanobis_distance_block(block,this->m);

	double weight_v = 0;
	for (typename vector<size_t>::const_iterator it = indices_ellipsoid.begin(); it != indices_ellipsoid.end(); ++it) {
		weight_v+=this->Sigma_inv.mahalanobis_distance(block.row(*it) - mean);
	}
	weight_v/=q;

	for(size_t dd = 0 ;  dd < d ; ++dd){
		weight_v -= log(this->Sigma_inv.Eigenvalues_inverted_to_use[dd]);
	}
	this-> v = weight_v;
}


template <size_t d> 
void Centroid<d>::activate (){
	this->active = true;
}

template <size_t d> 
void Centroid<d>::desactivate (){
	this->active = false;
}

// This function copies all attributes of the centroid, excepts for the indices of the points in the Voronoi cell.
template <size_t d> 
void Centroid<d>::equals (const Centroid<d>& c){
	(this->X).noalias() = c.X;
	(this->m).noalias() = c.m;
	this->v = c.v;
	this->active = c.active;
	(this->Sigma_inv.Eigenvalues_inverted_initial).noalias() = c.Sigma_inv.Eigenvalues_inverted_initial;
	(this->Sigma_inv.Eigenvalues_inverted_to_use).noalias() =  c.Sigma_inv.Eigenvalues_inverted_to_use;
	(this->Sigma_inv.Eigenvectors).noalias() = c.Sigma_inv.Eigenvectors ;
    }

template <size_t d> 
void Centroid<d>::update_active_X_m_Sigma_inverted(const Matrix<double,Dynamic,d> & block,size_t q,size_t method,size_t d_intrinsic,double sigma2_min,double sigma2_max,double lambda_min,double lambda_max)
{

	if((this->Voronoi_points_indices).size() == 0)
	{
		this->active = false;
	}
	else
	{
		
		// Compute the mean (new X) and covariance matrix of the points in the Voronoi cell.
		std::pair<Matrix<double,1,d>, Matrix<double,d,d>> mean_cov_Voronoi = calcul_mean_cov<d>(block, this->Voronoi_points_indices);
		this-> X = mean_cov_Voronoi.first;

		//std::vector<double> Dist_to_center;
		//for (typename vector<Point<d>>::const_iterator it = points.begin(); it != points.end(); ++it)
		//{
		//	Dist_to_center.push_back(this->Sigma_inv.mahalanobis_distance(it->X - mean_cov_Voronoi.first));
		//}

		//Array<double,Dynamic,d> Dist_to_center = (this->Sigma_inv).mahalanobis_distance_block(block,this->X);
		auto Dist_to_center = (this->Sigma_inv).mahalanobis_distance_block(block,this->X);

		// Indices of the q closest points to this->X, in points, for the Mahalanobis norm.
		vector<size_t> indices_ellipsoid(block.rows());
		iota(indices_ellipsoid.begin(), indices_ellipsoid.end(), 0);
		stable_sort(indices_ellipsoid.begin(), indices_ellipsoid.end(), [&Dist_to_center](size_t i1, size_t i2) {return Dist_to_center[i1] < Dist_to_center[i2];});

		indices_ellipsoid.resize(q);

		// Compute the mean (new m) and covariance matrix of the points in the ellipsoid centered at X with old covariance matrix Sigma_inv.
		std::pair<Matrix<double,1,d>, Matrix<double,d,d>> mean_cov_ellipsoid = calcul_mean_cov<d>(block, indices_ellipsoid);
		this-> m =  mean_cov_ellipsoid.first;

		// Update the global covariance matrix (Sigma_inv).
		Matrix<double,d,d> A =  mean_cov_Voronoi.second + mean_cov_ellipsoid.second + (mean_cov_Voronoi.first - mean_cov_ellipsoid.first).transpose()*(mean_cov_Voronoi.first - mean_cov_ellipsoid.first) ;
		SelfAdjointEigenSolver<Matrix<double,d,d>> eigensolver(A);
		this->Sigma_inv.Eigenvectors = eigensolver.eigenvectors();
		for(size_t dd = 0 ;  dd < d ; ++dd)
		{
			this->Sigma_inv.Eigenvalues_inverted_initial[dd] = 1/eigensolver.eigenvalues()[dd]; // inverse of the eigenvalues to get the inverse of the covariance matrix.
		}
		if(method == 0) // No modification of eigenvalues
		{
			this->Sigma_inv.Eigenvalues_inverted_to_use = this->Sigma_inv.Eigenvalues_inverted_initial;
		}
		else
		{
			if(method == 1)
			{
				this->Sigma_inv.trunc_eigenvalues(d_intrinsic,sigma2_min,sigma2_max,lambda_min,lambda_max);
			}
			else
			{
				if(method == 2)
				{
					this->Sigma_inv.trunc_eigenvalues_sigma2_const(d_intrinsic,sigma2_max,lambda_min,lambda_max);
				}
				else
				{
					this->Sigma_inv.eigenvalues_det_to_1();
				}
			}
		}
/*
*/
	}

}

template <size_t d> 
void Centroid<d>::update_Voronoi_points_indices(const std::vector<size_t> & indices){
	this->Voronoi_points_indices = indices;
}


// Function to access the arguments :

template <size_t d> 
Matrix<double,d,d> Centroid<d>::get_inverse_matrix_to_use() const
{
	return(this->Sigma_inv.return_inverse_matrix_to_use());
}

template <size_t d>
Matrix<double,d,d> Centroid<d>::get_matrix_to_use() const
{
	return(this->Sigma_inv.return_matrix_to_use());
}

template <size_t d> 
Matrix<double,d,d> Centroid<d>::get_inverse_matrix_initial() const
{
	return(this->Sigma_inv.return_inverse_matrix_initial());
}

template <size_t d> 
Matrix<double,d,d> Centroid<d>::get_matrix_initial() const
{
	return(this->Sigma_inv.return_matrix_initial());
}

template <size_t d> 
Matrix<double,1,d> Centroid<d>::get_X() const
{
	return(this->X);
}

template <size_t d> 
std::vector<size_t> Centroid<d>::get_Voronoi_points_indices() const
{
	return(this->Voronoi_points_indices);
}

template <size_t d> 
Matrix<double,1,d> Centroid<d>::get_m() const
{
	return(this->m);
}

template <size_t d> 
double Centroid<d>::get_v() const
{
	return(this->v);
}

template <size_t d> 
bool Centroid<d>::get_active() const
{
	return(this->active);
}

template <size_t d> 
Matrix<double,d,1> Centroid<d>::get_eigenvalues_initial() const
{
	return(this->Sigma_inv.get_eigenvalues_initial());
}

template <size_t d> 
Matrix<double,d,1> Centroid<d>::get_eigenvalues_to_use() const
{
	return(this->Sigma_inv.get_eigenvalues_to_use());
}



#endif  // CENTROID_H_
