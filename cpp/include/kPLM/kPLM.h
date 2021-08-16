/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Claire Brécheteau
 *
 *    Copyright (C) 2021 Université Rennes 2
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */



// Modifications à apporter :

// Vérifier que mahalanobis par blocs fonctionne pour des très gros nuages de points; 
// Changer eigenvalues en d 1 au lieu de 1 d, pour ne pas avoir à transposer dans la fonction mahalanobis par bloc
// Ajouter une kPLM par blocs.
// Vérifier que kPLM fonctionne.
// Ajouter kPLM 


#ifndef KPLM_H_
#define KPLM_H_

#include "Point.h"
#include "Centroid.h"


// class definition

template <size_t d> 
class kPLM {

private:

	size_t method;

	std::vector<Point<d>> points;
	Matrix<double,Dynamic,d> block; // Initialisation : block.resize(points.size(),d) ; block.noalias() = block_points(this->points);
	std::vector<Centroid<d>> initial_centroids;

	size_t k; // Initial Number of centroids
	size_t q; // Number of nearest neighbours
	size_t n_signal; // Number of points considered as signal
	size_t n_starts;
	size_t epochs;

	size_t d_intrinsic; // should be between 0 and d (the d-d_intrinsic smallest eigenvalues are equal)
	double sigma2_min;
	double sigma2_max;
	double lambda_min;
	double lambda_max;

	bool normalized_det; // 1 if the determinant of matrices should be 1, 0 if no modification needs to be done.

	std::vector<Centroid<d>> optimal_centroids;
	std::vector<size_t> optimal_labels;
	std::vector<double> optimal_costs;
	double optimal_cost;
	bool optimisation_done; // 1 if the algorithm already ran; 0 if not.

public:

	kPLM(char* points_file);
	kPLM(char* points_file, size_t k, size_t q, size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max);
	kPLM(const std::vector<Point<d>> & points);
	kPLM(const std::vector<Point<d>> & points, size_t k, size_t q, size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max);


	// Une fonction pour supprimer les centroids avec un poids trop gros. (Ca va modifier optimal_centroids, optimal_labels, optimal_costs et optimal_cost).
	
//void remove_centroids(double threshold); // centroids with weight inferior to threshold are removed - Labels of points are updated (optimal_centroids; optimal_labels; optimal_costs;optimal_cost modified) --- Use after a write_centroids (the weights are in this file)

	// Fonctions pour mettre à jour les paramètres.
	void update_parameters(size_t method, size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max);
	void update_parameters_principal(size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs);
	void update_parameters_secondary(size_t method, size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max);

	// Fonction pour avoir accès aux paramètres.
	std::tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,double,double,double,double,bool,size_t> parameters_access();

	// Fonction pour avoir accès à des copies des valeurs des points.
	std::vector<Matrix<double,1,d>> points_values_access();

	// Des fonctions pour créer un fichier avec les différents retours (centroides avec les matrices de covariance, labels, coûts par point, coût total).

	void write_clusters(char const *address = "clusters.csv");
	void write_costs(char const *address = "costs.csv");
	void write_centroids(char const *address = "centroids.csv");
	void write_matrices_inv(char const *address = "matrices_inv.csv");

	double get_optimal_cost();

	// Des fonctions pour retourner des copies des attributs optimal_centroids; optimal_labels; optimal_costs; optimal_cost.
	std::vector<Centroid<d>> get_optimal_centroids();
	std::vector<size_t> get_optimal_clusters();
	std::vector<double> get_optimal_costs();

	void print_optimal_centroids();
	void print_optimal_clusters();
	void print_optimal_costs();

	// Une fonction pour initialiser les centroids.
	void initialise_centroids(bool using_optimal_centroids = false, bool transform_centroids = false); // If first TRUE, use optimal_centroids as initial centroids, if not, use identity matrices ; If second true, transform eigenvalues of the centroids according to the function, if not, just use the centroids.
	void initialise_centroids(const std::vector<Centroid<d>> & centroids);
//void update_initial_centroids(std::vector<Centroid<d>> initial_centroids);

	// Une fonction simple pour calculer optimal_centroids; optimal_labels; optimal_costs; optimal_cost.
	void run_kPLM(bool using_optimal_centroids=false, bool transform_centroids=false);

	// Une fonction pour calculer les optimaux pour différentes valeurs de d_intrinsic, en commençant avec d_intrinsic = d_max, puis diminuant, jusqu'à tomber sur d_min. (d_max = d et d_min = 0 par exemple).
//void run_kPLM(size_t d_max, size_t d_min);
	std::vector<std::tuple<	std::vector<Centroid<d>>, std::vector<size_t>,  std::vector<double>, double>> run_multiple_kPLM(size_t d_max = ULONG_MAX, size_t d_min = ULONG_MAX);
	std::vector<std::tuple<	std::vector<Centroid<d>>, std::vector<size_t>,  std::vector<double>, double>> run_multiple_kPLM_several_times(size_t d_max, size_t d_min);

	// Une fonction d'un point pour donner sa valeur de kPLM (en fonction de optimal_centroids).
//double compute_kPLM_value(Point<d> p);

	// Une fonction pour ajouter du clustering hiérarchique... ???
	// TO WRITE LATER...

};

// auxiliary functions definition

template<size_t d>
kPLM<d>::kPLM(char* points_file) :
	method(0),
	k(1),
	q(1),
	n_starts(10),
	epochs(20),
	d_intrinsic(0), // all eigenvalues are equal
	sigma2_min(1),
	sigma2_max(1),
	lambda_min(1),
	lambda_max(1),
	normalized_det(0), // no modification of the matrices determinant needs to be done.
	optimal_cost(__DBL_MAX__),
	optimisation_done(0) // algorithm did not run yet.
	{
		this->points = readcsv<d>(points_file);
		this->block.resize(points.size(),d) ;
		block.noalias() = block_points(this->points);
		this-> n_signal = (this->points).size();
	}


template<size_t d>
kPLM<d>::kPLM(char* points_file, size_t k, size_t q, size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max) :
	method(0),
	k(k),
	q(q),
	n_signal(n_signal),
	n_starts(n_starts),
	epochs(epochs),
	d_intrinsic(d_intrinsic),
	sigma2_min(sigma2_min),
	sigma2_max(sigma2_max),
	lambda_min(lambda_min),
	lambda_max(lambda_max),
	normalized_det(0), // no modification of the matrices determinant needs to be done.
	optimal_cost(__DBL_MAX__),
	optimisation_done(0) // algorithm did not run yet.
	{
		this->points = readcsv<d>(points_file);
		this->block.resize(points.size(),d) ;
		block.noalias() = block_points(this->points);
		if (n_signal < 0 || n_signal > (this->points).size()) throw std::invalid_argument("ERROR: n_signal should be non-negative and smaller than the sample size. \n");
		if (k < 1 || k > (this->points).size()) throw std::invalid_argument("ERROR: k should be positive and not larger than the sample size. \n");
		if (q < 0 || q > (this->points).size()) throw std::invalid_argument("ERROR: q should be non-negative and not larger than the sample size. \n");
		if (d_intrinsic < 0 || d_intrinsic > d) throw std::invalid_argument("ERROR: d_intrinsic should not be larger than the dimension d of data points. \n");
		if (sigma2_min < 0 || sigma2_min > sigma2_max) throw std::invalid_argument("ERROR: sigma2_min should be positive and not larger than sigma2_max. \n");
		if (lambda_min < sigma2_max || lambda_min > lambda_max) throw std::invalid_argument("ERROR: lambda_min should not be smaller than sigma2_max and should not be larger than lambda_max. \n");
	}




template<size_t d>
kPLM<d>::kPLM(const std::vector<Point<d>> & points) :
	method(0),
	k(1),
	q(1),
	n_starts(10),
	epochs(20),
	d_intrinsic(0), // all eigenvalues are equal
	sigma2_min(1),
	sigma2_max(1),
	lambda_min(1),
	lambda_max(1),
	normalized_det(0), // no modification of the matrices determinant needs to be done.
	optimal_cost(__DBL_MAX__),
	optimisation_done(0) // algorithm did not run yet.
	{
		this->points = points;
		this->block.resize(points.size(),d) ;
		block.noalias() = block_points(this->points);
		this-> n_signal = (this->points).size();
	}


template<size_t d>
kPLM<d>::kPLM(const std::vector<Point<d>> & points, size_t k, size_t q, size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max) :
	method(0),
	k(k),
	q(q),
	n_signal(n_signal),
	n_starts(n_starts),
	epochs(epochs),
	d_intrinsic(d_intrinsic),
	sigma2_min(sigma2_min),
	sigma2_max(sigma2_max),
	lambda_min(lambda_min),
	lambda_max(lambda_max),
	normalized_det(0), // no modification of the matrices determinant needs to be done.
	optimal_cost(__DBL_MAX__),
	optimisation_done(0) // algorithm did not run yet.
	{
		this->points = points;
		this->block.resize(points.size(),d) ;
		block.noalias() = block_points(this->points);
		if (n_signal < 0 || n_signal > (this->points).size()) throw std::invalid_argument("ERROR: n_signal should be non-negative and smaller than the sample size. \n");
		if (k < 1 || k > (this->points).size()) throw std::invalid_argument("ERROR: k should be positive and not larger than the sample size. \n");
		if (q < 0 || q > (this->points).size()) throw std::invalid_argument("ERROR: q should be non-negative and not larger than the sample size. \n");
		if (d_intrinsic < 0 || d_intrinsic > d) throw std::invalid_argument("ERROR: d_intrinsic should not be larger than the dimension d of data points. \n");
		if (sigma2_min < 0 || sigma2_min > sigma2_max) throw std::invalid_argument("ERROR: sigma2_min should be positive and not larger than sigma2_max. \n");
		if (lambda_min < sigma2_max || lambda_min > lambda_max) throw std::invalid_argument("ERROR: lambda_min should not be smaller than sigma2_max and should not be larger than lambda_max. \n");
	}

	
	

//template<size_t d>
//void kPLM<d>::remove_centroids(double threshold);

//template<size_t d>
//void kPLM<d>::update_initial_centroids(); // Identity matrices

//template<size_t d>
//void kPLM<d>::update_initial_centroids(std::vector<Centroid<d>> initial_centroids);

template<size_t d>
void kPLM<d>::update_parameters(size_t method, size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max){
	this->method = method;
	this->k = k;
	this->q = q;
	this->n_signal = n_signal;
	this->n_starts = n_starts;
	this->epochs = epochs;
	this->d_intrinsic = d_intrinsic;
	this->sigma2_min = sigma2_min;
	this->sigma2_max = sigma2_max;
	this->lambda_min = lambda_min;
	this->lambda_max = lambda_max;
	this->optimisation_done = 0;
}

template<size_t d>
void kPLM<d>::update_parameters_principal(size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs){
	this->k = k;
	this->q = q;
	this->n_signal = n_signal;
	this->n_starts = n_starts;
	this->epochs = epochs;
	this->optimisation_done = 0;
}

template<size_t d>
void kPLM<d>::update_parameters_secondary(size_t method, size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max){
	this->method = method;
	this->d_intrinsic = d_intrinsic;
	this->sigma2_min = sigma2_min;
	this->sigma2_max = sigma2_max;
	this->lambda_min = lambda_min;
	this->lambda_max = lambda_max;
	this->optimisation_done = 0;
}

template<size_t d>
std::tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,double,double,double,double,bool,size_t> kPLM<d>::parameters_access(){
	return(make_tuple(this->method, this->k, this->q, this->n_signal, this->n_starts, this->epochs, this->d_intrinsic, this->sigma2_min, this->sigma2_max, this->lambda_min, this->lambda_max, this->normalized_det,this->points.size()));
}

template<size_t d>
std::vector<Matrix<double,1,d>> kPLM<d>::points_values_access(){
	std::vector<Matrix<double,1,d>> points_values(0);
	for(typename std::vector<Point<d>>::iterator it = this->points.begin(); it!= this->points.end(); ++it)
	{
		points_values.push_back(it->X);
	}
	return(points_values);
}

template<size_t d>
void kPLM<d>::write_clusters(char const *address){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant labels are available. \n");
	ofstream myfile(address);
	myfile << "c" << endl;
	for (typename vector<size_t>::const_iterator it = optimal_labels.begin(); it != optimal_labels.end(); ++it) {
		myfile << *it << endl;
	}
	myfile.close();	
}

template<size_t d>
void kPLM<d>::write_costs(char const *address){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant costs for points are available. \n");
	ofstream myfile(address);
	myfile << "c" << endl;
	for (typename vector<double>::const_iterator it = optimal_costs.begin(); it !=optimal_costs.end(); ++it) {
		myfile << *it << endl;
	}
	myfile.close();	
}

template<size_t d>
void kPLM<d>::write_centroids(char const *address){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant optimal centroids are available. \n");
	ofstream myfile(address);
	for(size_t dd = 0 ; dd<d ; ++dd) myfile << "x" << dd << "," ;
	for(size_t dd = 0 ; dd<d ; ++dd) myfile << "mx" << dd << "," ;
	myfile << "v, active" << endl;
	for (typename vector<Centroid<d>>::const_iterator it = optimal_centroids.begin() ; it != optimal_centroids.end(); ++it) {
		for(size_t dd = 0 ; dd<d ; ++dd) myfile << it->X[dd] << "," ;
		for(size_t dd = 0 ; dd<d ; ++dd) myfile << it->m[dd] << "," ;
		myfile << it->v << "," << it->get_active() << endl;
	}
	myfile.close();
}

template<size_t d>
void kPLM<d>::write_matrices_inv(char const *address){ // Each line corresponds to the inverse of a covariance matrix. The first d elements correspond to the first line of the matrix, the elements from index d+1 to 2d, to the second line of the matrix etc.
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant optimal centroids are available. \n");
	ofstream myfile(address);
	for(size_t dd = 0 ; dd<d-1 ; ++dd){
		for(size_t ee=0; ee<d; ++ee){
			myfile << "x" << dd <<"-"<<ee<< "," ;
		}
	}
		for(size_t ee=0; ee<d-1; ++ee){
			myfile << "x" << d-1 <<"-"<<ee<< "," ;
		}
	myfile<< "x" << d-1 <<"-"<< d-1 <<  endl;
	Matrix<double,d,d> Mat ;
	Matrix<double,d,d> Diag(Matrix<double,d,d>::Zero());
	for (typename vector<Centroid<d>>::const_iterator it = optimal_centroids.begin() ; it != optimal_centroids.end(); ++it) {
		for(size_t dd=0; dd<d; ++dd){Diag(dd,dd) = it->Sigma_inv.Eigenvalues_inverted_to_use[dd] ;}
		Mat.noalias() = it->Sigma_inv.Eigenvectors* Diag * (it->Sigma_inv.Eigenvectors).transpose();

		for(size_t dd = 0 ; dd<d-1 ; ++dd){
			for(size_t ee=0; ee<d; ++ee){
				myfile << Mat(dd,ee) << "," ;
			}
		}
			for(size_t ee=0; ee<d-1; ++ee){
				myfile << Mat(d-1,ee) << "," ;
			}
		myfile << Mat(d-1,d-1) << endl;
	}
	myfile.close();
}

template<size_t d>
double kPLM<d>::get_optimal_cost(){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant optimal cost is available. \n");
	return(optimal_cost);
}

template<size_t d>
std::vector<Centroid<d>> kPLM<d>::get_optimal_centroids(){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant optimal centroids are available. \n");
	return(optimal_centroids);
}

template<size_t d>
std::vector<size_t> kPLM<d>::get_optimal_clusters(){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant labels are available. \n");
	return(optimal_labels);
}

template<size_t d>
std::vector<double> kPLM<d>::get_optimal_costs(){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant costs for points are available. \n");
	return(optimal_costs);	
}

template<size_t d>
void kPLM<d>::print_optimal_centroids(){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant optimal centroids are available. \n");
	for(size_t dd = 0 ; dd<d ; ++dd) std::cout << "x" << dd << "," ;
	for(size_t dd = 0 ; dd<d ; ++dd) std::cout << "mx" << dd << "," ;
	std::cout << "v, active" << endl;
	for (typename vector<Centroid<d>>::const_iterator it = optimal_centroids.begin() ; it != optimal_centroids.end(); ++it) {
		for(size_t dd = 0 ; dd<d ; ++dd) std::cout << it->X[dd] << "," ;
		for(size_t dd = 0 ; dd<d ; ++dd) std::cout << it->m[dd] << "," ;
		std::cout << it->v << "," << it->get_active() << endl;
	}
	for(size_t dd = 0 ; dd<d-1 ; ++dd){
		for(size_t ee=0; ee<d; ++ee){
			std::cout << "x" << dd <<"-"<<ee<< "," ;
		}
	}
		for(size_t ee=0; ee<d-1; ++ee){
			std::cout << "x" << d-1 <<"-"<<ee<< "," ;
		}
	std::cout<< "Matrices" <<  endl;
	Matrix<double,d,d> Mat ;
	Matrix<double,d,d> Diag(Matrix<double,d,d>::Zero());
	for (typename vector<Centroid<d>>::const_iterator it = optimal_centroids.begin() ; it != optimal_centroids.end(); ++it) {
		for(size_t dd=0; dd<d; ++dd){Diag(dd,dd) = it->Sigma_inv.Eigenvalues_inverted_to_use[dd] ;}
		Mat.noalias() = it->Sigma_inv.Eigenvectors* Diag * (it->Sigma_inv.Eigenvectors).transpose();

		for(size_t dd = 0 ; dd<d-1 ; ++dd){
			for(size_t ee=0; ee<d; ++ee){
				std::cout << Mat(dd,ee) << "," ;
			}
		}
			for(size_t ee=0; ee<d-1; ++ee){
				std::cout << Mat(d-1,ee) << "," ;
			}
		std::cout << Mat(d-1,d-1) << endl;
	}
}

template<size_t d>
void kPLM<d>::print_optimal_clusters(){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant labels are available. \n");
	for (typename vector<size_t>::const_iterator it = optimal_labels.begin(); it != optimal_labels.end(); ++it) {
		std::cout << *it <<", " ;
	}
	std::cout<<std::endl;
}

template<size_t d>
void kPLM<d>::print_optimal_costs(){
	if(optimisation_done == 0)throw std::invalid_argument("WARNING: The optimisation is not yet done, so no relevant costs for points are available. \n");
	for (typename vector<double>::const_iterator it = optimal_costs.begin(); it !=optimal_costs.end(); ++it) {
		std::cout << *it <<", " ;
	}
	std::cout<<std::endl;
}


// Attention à remettre à jour certains paramètres des centroids....


template<size_t d>
void kPLM<d>::initialise_centroids(bool using_optimal_centroids, bool transform_centroids){
	if(k < 1)throw std::invalid_argument("WARNING: There are no centroids since k<1. \n");
	if(points.size() < 1)throw std::invalid_argument("WARNING:  There are no points, so no centroids to initialise.\n");
	initial_centroids.resize(0);
	if(using_optimal_centroids)
	{
		for (size_t kk = 0; kk < k; ++kk)
		{
			Centroid<d> next_centroid = optimal_centroids[kk];
			next_centroid.activate();
			initial_centroids.push_back(next_centroid);
		}	
	}
	else
	{
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<size_t> distrib(0, points.size()-1);
		srand(time(0));
		for (size_t kk = 0; kk < k; ++kk) {
			Centroid<d> next_centroid(points.at(distrib(gen)));
			next_centroid.activate();
			initial_centroids.push_back(next_centroid);
		}
	}
	if(transform_centroids)
	{
		if(method == 1)
		{
			for(typename std::vector<Centroid<d>>::iterator it = initial_centroids.begin(); it != initial_centroids.end(); ++it)
			{
				it->Sigma_inv.trunc_eigenvalues(d_intrinsic,sigma2_min,sigma2_max,lambda_min,lambda_max);
			}
		}
		else
		{
			if(method == 2)
			{
				for(typename std::vector<Centroid<d>>::iterator it = initial_centroids.begin(); it != initial_centroids.end(); ++it)
				{
					it->Sigma_inv.trunc_eigenvalues_sigma2_const(d_intrinsic,sigma2_max,lambda_min,lambda_max);
				}				
			}
			else
			{
				for(typename std::vector<Centroid<d>>::iterator it = initial_centroids.begin(); it != initial_centroids.end(); ++it)
				{
					it->Sigma_inv.eigenvalues_det_to_1();
				}	
			}
		}
	}
}


template<size_t d>
void kPLM<d>::initialise_centroids(const std::vector<Centroid<d>> & centroids)
{
	if(k < 1)throw std::invalid_argument("WARNING: There are no centroids since k<1. \n");
	if(points.size() < 1)throw std::invalid_argument("WARNING:  There are no points, so no centroids to initialise.\n");
	if(centroids.size() != k)throw std::invalid_argument("WARNING:  The number of initial centroids should be equal to parameter k.\n");
	initial_centroids.resize(0);
	for (size_t kk = 0; kk < k; ++kk)
	{
		Centroid<d> next_centroid = centroids[kk];
		next_centroid.activate();
		initial_centroids.push_back(next_centroid);
	}
}

	// Une fonction simple pour calculer optimal_centroids; optimal_labels; optimal_costs; optimal_cost.
template<size_t d>
void kPLM<d>::run_kPLM(bool using_optimal_centroids, bool transform_centroids)
{	

	//Matrix<double,Dynamic,d> block = block_points(this->points);

	vector<Centroid<d>> opt_centroids(k);
	double opt_cost = __DBL_MAX__;
	optimisation_done = 0;

	for (size_t nst = 0 ; nst < n_starts ; ++nst)
	{

		/*		
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout<<"Start number : "<<nst<<std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
		/*
		*/

		initialise_centroids(using_optimal_centroids, transform_centroids);
		vector<Centroid<d>> centroids = initial_centroids;
		vector<Centroid<d>> old_centroids = initial_centroids;

		// print centroids

		/*
		for (typename vector<Centroid<d>>::const_iterator it = centroids.begin() ; it != centroids.end(); ++it)
		{
			for(size_t dd = 0 ; dd<d ; ++dd) std::cout << it->X[dd] << "," ;
			std::cout << std::endl;
			for(size_t dd = 0 ; dd<d ; ++dd) std::cout << it->m[dd] << "," ;
			std::cout << std::endl;
			std::cout << it->v << "," << it->get_active() << std::endl;
			std::cout << it->Centroid<d>::get_matrix_to_use() << std::endl;
			std::cout << std::endl;
		}
		/*
		*/

		double cost_step = __DBL_MAX__;
		double previous_cost_step = 0;

		size_t nstep = 0;


		while (nstep < epochs && cost_step!=previous_cost_step){

		/*
			std::cout << std::endl;
			std::cout<<"Step number : "<<nstep<<std::endl;
			std::cout << std::endl;
		/*
		*/

			++nstep;
			previous_cost_step = cost_step;

			for(typename std::vector<Centroid<d>>::iterator it = centroids.begin(); it!= centroids.end(); ++it)
			{
				it->Centroid<d>::update_m_v(block,q);
			}

			// Initialisation of minDist
			for (typename std::vector<Point<d>>::iterator it = points.begin(); it != points.end(); ++it)
			{
				it->minDist = __DBL_MAX__;
			}


			// Associate the proper clusterId to each point

			size_t centroid_index = 0;

			for (typename std::vector<Centroid<d>>::iterator c = centroids.begin(); c != centroids.end(); ++c)
			{
				if(c->get_active() == true)
				{
					Array<double,Dynamic,1> kPLM_values = c->Centroid<d>::distance_kPLM_block(block);		
					size_t point_index = 0;

					for (typename std::vector<Point<d>>::iterator it = points.begin(); it != points.end(); ++it)
					{

						if (kPLM_values[point_index] < it->minDist) {
						    it->minDist = kPLM_values[point_index];
						    it->cluster = centroid_index;

						}
						++point_index;
					}
				}
				++centroid_index;
			}
		
			// Trimming step

			std::vector<size_t> idx(points.size());
		 	iota(idx.begin(), idx.end(), 0);
			//stable_sort(idx.begin(), idx.end(), [&points](size_t i1, size_t i2) {return points[i1].minDist < points[i2].minDist;});
			stable_sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {return points[i1].minDist < points[i2].minDist;});

			cost_step = 0;
			for (size_t i = 0 ; i < n_signal ; ++i){
				cost_step += points[idx[i]].minDist;
			}

			for (size_t i = n_signal; i < points.size() ; ++i){
				points[idx[i]].cluster = k; // The points which cluster id equals to k are removed.
			}


			// Compute the new centroids	
			for(size_t kk = 0 ; kk<k; ++kk){old_centroids[kk].equals(centroids[kk]);}

			// Append data to centroids

			for (typename std::vector<Centroid<d>>::iterator it = centroids.begin(); it != centroids.end(); ++it)
			{	
				(it-> Voronoi_points_indices).resize(0);
			}

			for (typename std::vector<Point<d>>::const_iterator it = points.begin(); it != points.end(); ++it)
			{
				size_t point_index = it - begin(points);
				size_t centroid_index = it->cluster;
				if(centroid_index < k)
				{
					centroids[centroid_index].Voronoi_points_indices.push_back(point_index);
				}
			}

			// update centroids
			for (typename std::vector<Centroid<d>>::iterator c = centroids.begin(); c != centroids.end(); ++c){
				size_t centroid_index = c - begin(centroids);
				if(c->Voronoi_points_indices.empty()){
					old_centroids[centroid_index].desactivate();
					centroids[centroid_index].desactivate();
				}
				else{
					c->Centroid<d>::update_active_X_m_Sigma_inverted(block,q,method,d_intrinsic,sigma2_min,sigma2_max,lambda_min,lambda_max);
				}
			}

			
		}

		if(cost_step < opt_cost){
			opt_cost = cost_step;
			for(size_t kk=0; kk<k; ++kk){opt_centroids[kk].equals(old_centroids[kk]);} //opt_centroids = old_centroids;
			for(typename std::vector<Point<d>>::iterator it = points.begin(); it != points.end(); ++it) {
				it->opt_cluster = it->cluster;
				it->opt_minDist = it->minDist;
			}
		}	

	}

	std::vector<size_t> opt_labels(0);
	for (typename std::vector<Point<d>>::iterator it = points.begin(); it != points.end(); ++it) {
		opt_labels.push_back((it->opt_cluster)==k?0:(it->opt_cluster)+1);
	}

	std::vector<double> opt_costs(0);
	for (typename std::vector<Point<d>>::iterator it = points.begin(); it != points.end(); ++it) {
		opt_costs.push_back(it->opt_minDist);
	}

	optimal_centroids = opt_centroids;
	optimal_labels = opt_labels;
	optimal_costs = opt_costs;
	optimal_cost = opt_cost;
	optimisation_done = 1;
}


template<size_t d>
std::vector<std::tuple<	std::vector<Centroid<d>>, std::vector<size_t>,  std::vector<double>, double>> kPLM<d>::run_multiple_kPLM(size_t d_max, size_t d_min)
{
	// Runs only ONCE for different values of d_intrinsic.

	if(d_max == ULONG_MAX){d_max = d;}
	if(d_min == ULONG_MAX){d_min = 0;}

	size_t keep_n_starts = n_starts;
	size_t keep_d_intrinsic = d_intrinsic;
	n_starts = 1;

	run_kPLM(false,false);
	
	std::vector<std::tuple<	std::vector<Centroid<d>>, std::vector<size_t>,  std::vector<double>, double>> optima(0);

	for(size_t d_intrin = d_max+1; d_intrin !=d_min; --d_intrin)
	{

		d_intrinsic = d_intrin - 1;

		std::cout<< d_intrinsic << std::endl;
		run_kPLM(true,true);

		optima.push_back(std::make_tuple(optimal_centroids,optimal_labels,optimal_costs,optimal_cost));
	}
	
	n_starts = keep_n_starts;
	d_intrinsic = keep_d_intrinsic;

	return(optima);
// Une fois avec d_intrinsic = d pour avoir des centroids de départ.
// On tronque les matrices avec dintrinsic.
// On passe au dintrinsic ; puis on diminue dintrinsic jusqu'à ce qui est souhaité. On conserve les résultats des algos !
}

template<size_t d>
std::vector<std::tuple<	std::vector<Centroid<d>>, std::vector<size_t>,  std::vector<double>, double>> kPLM<d>::run_multiple_kPLM_several_times(size_t d_max, size_t d_min)
{

	if(d_max == ULONG_MAX){d_max = d;}
	if(d_min == ULONG_MAX){d_min = 0;}
	
	auto opt = run_multiple_kPLM(d_max,d_min);

	for (size_t nst = 1 ; nst < n_starts ; ++nst)
	{
		auto ans = run_multiple_kPLM(d_max,d_min);
		for(size_t d_intrin = d_max+1; d_intrin !=d_min; --d_intrin)
		{
			d_intrinsic = d_intrin - 1;
			if(std::get<3>(ans[d_intrinsic]) < std::get<3>(opt[d_intrinsic]))
			{
				opt[d_intrinsic] = ans[d_intrinsic];
			}
		}
	}
	return(opt);
}

/*

friend function...
fast_run_multiple_kPLM_several_times(size_t k, size_t q, size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max);
std::vector<Point<d>> & points
kPLM( points, size_t k, size_t q, size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max)
*/


#endif  // KPLM_H_
