// From version v1


// Dans cette version, création d'une classe spéciale pour une méthode de clustering, avec comme attribu les paramètres, le nuage de points, l'ensemble des centres (que l'on peut initialiser, mettre à jour)...



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



/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

								CLASS DEFINITIONS 

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */




/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							Class Point (ok)


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */



// friend template functions for class Point

template <size_t d> 
std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov(const std::vector<Point<d>> & points, const std::vector<size_t> & indices);

// class definition


template <size_t d> 
class Point {

private:
	Matrix<double,1,d> X;     // coordinates
	size_t cluster;     // no default cluster
	double minDist;  // default infinite ; distance to the nearest cluster (for the cost function)
	size_t opt_cluster; // cluster id for the best attempt
	double opt_minDist; // distance to the nearest cluster for the best attempt

	double distance(const Centroid<d> &c) const; //squared Euclidean distance
	double distance_kPLM(const Centroid<d> &c) const; // squared Mahalanobis to the mean + variance + log(det)

public:
	Point();
	Point(const double (&tab)[d],size_t index);
	Point(const Matrix<double,1,d> & tab,size_t index);

	friend class Centroid<d>;
	friend class kPLM<d>;

	friend std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov<d>(const std::vector<Point<d>> & points, const std::vector<size_t> & indices);	
};


/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							Class Centroid


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */


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
	//vector<size_t> near_neigh; 	// indices of some part of the nearest-neighbors of the Centroid with coordinates Xold (not necessarily ordered);
	//Matrix<double,1,d> Xold;
	//double dist_to_old_centroid_threshold;

	void update_m_v(size_t q,const vector<Point<d>> & points);
	void update_X_m_v_Sigma_inverted(const std::vector<Point<d>> & points,size_t q);
	//void initialize_near_neigh_Xold(size_t q,const vector<Point<d>> & points); // Compute the nearest neighbors indices ; keeps the ones with distance smaller then 3 times the distance to its q-th nearest neighbor.
	//double dist_to_old_centroid();
	void equals (const Centroid<d>& c);

public:
	Centroid();
	Centroid(const double (&tab)[d]);
	Centroid(const Point<d> &p);
	// All constructors make a call to the constructor for Sigma_inverted<d>; The matrix Sigma_inv is the Identity matrix.


	
	friend class Point<d>;
	friend class kPLM<d>;
	friend class Sigma_inverted<d>;
};



/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							Class Sigma_inverted


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */




template <size_t d> 
class Sigma_inverted {

private:
	double Eigenvalues_inverted_initial[d]; // Contains 1/Eigenvalues with Eigenvalues sorted in non-decreasing order
	double Eigenvalues_inverted_to_use[d];
	Matrix<double,d,d> Eigenvectors; // Orthogonal matrix
	double trunc_eigenvalues(size_t d_intrinsic,double sigma2_min,double sigma2_max,double lambda_min,double lambda_max = __DBL_MAX__); // The d_intrinsic first eigenvalues truncated if larger than lambda_max or smaller than lambda_min ; The d - d_intrinsic eigenvalues are replaced with their mean, truncated if larger than sigma2_max or smaller than sigma2_min;
	double trunc_eigenvalues_sigma2_const(size_t d_intrinsic,double sigma2,double lambda_min,double lambda_max = __DBL_MAX__); // sigma2 is fixed so no need to compute its value...
	double eigenvalues_det_to_1(); // the eigenvalues are divided by their product to the 1/d.

public: 
	Sigma_inverted();
	Sigma_inverted(double (&Eigenvalues_inverted)[d]);
	Sigma_inverted(Matrix<double,d,d> Eigenvectors);
	Sigma_inverted(double (&Eigenvalues_inverted)[d], Matrix<double,d,d> Eigenvectors);

	double mahalanobis_distance(const Matrix<double,1,d> &vect) const;

	friend class Centroid<d>;
	friend class kPLM<d>;
	//friend void write_matrices_inv<d>(const vector<Centroid<d>> & centroids,char const *address);
};


/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

								CLASS - FUNCTIONS DEFINITIONS 

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							Class Point -- functions definitions


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */



template<size_t d>
Point<d>::Point() :
	X(Matrix<double,1,d>::Zero()),
        cluster(0),
	opt_cluster(0),
	opt_minDist(__DBL_MAX__),
        minDist(__DBL_MAX__)
	{
	}

template<size_t d>
Point<d>::Point(const double (&tab)[d],size_t index) : 
        cluster(0),
	opt_cluster(0),
	opt_minDist(__DBL_MAX__),
        minDist(__DBL_MAX__)
	{
	for(size_t dd = 0 ; dd < d ; ++dd) X[dd] = tab[dd];
	}

template<size_t d>
Point<d>::Point(const Matrix<double,1,d> & tab,size_t index) : 
	X(tab),
        cluster(0),
	opt_cluster(0),
	opt_minDist(__DBL_MAX__),
        minDist(__DBL_MAX__)
	{
	}

template<size_t d>
double Point<d>::distance(const Centroid<d> &c) const { // Square of the Euclidean distance
	return((c.X-X).squaredNorm());
} 

template<size_t d>
double Point<d>::distance_kPLM(const Centroid<d> &c) const { // Distance for weighted Voronoi cells for the kPLM
	return(c.Sigma_inv.mahalanobis_distance(c.m - this->X)+c.v);
} 





/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							Class Centroid -- functions definitions


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

// Ajouter un constructeur avec un centre et une matrice de covariance éventuellement

template<size_t d>
Centroid<d>::Centroid() :
	v(__DBL_MAX__),
	Voronoi_points_indices(0),
	active(true),
	m(Matrix<double,1,d>::Zero()),
	X(Matrix<double,1,d>::Zero())
	{
	}

template<size_t d>
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

template<size_t d>
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


//////// FONCTION A MODIFIER (prochaine version)



template<size_t d>
void Centroid<d>::update_m_v(size_t q,const vector<Point<d>> & points){

		std::vector<double> Dist_to_center;
		for (typename vector<Point<d>>::const_iterator it = points.begin(); it != points.end(); ++it) {
			Dist_to_center.push_back(this->Sigma_inv.mahalanobis_distance(it->X - this->X));
		}
		vector<size_t> indices_ellipsoid(points.size());
		iota(indices_ellipsoid.begin(), indices_ellipsoid.end(), 0);
		stable_sort(indices_ellipsoid.begin(), indices_ellipsoid.end(), [&Dist_to_center](size_t i1, size_t i2) {return Dist_to_center[i1] < Dist_to_center[i2];});

		indices_ellipsoid.resize(q);

		Matrix<double,1,d> mean(Matrix<double,1,d>::Zero());
		for (typename std::vector<size_t>::const_iterator it = indices_ellipsoid.begin() ; it != indices_ellipsoid.end() ; ++it) {
		    mean += points[*it].X ;
		}
		mean/= q;
		this-> m =  mean;

		double weight_v = 0;
		for (typename vector<size_t>::const_iterator it = indices_ellipsoid.begin(); it != indices_ellipsoid.end(); ++it) {
			weight_v+=this->Sigma_inv.mahalanobis_distance(points[*it].X - mean);
		}
		weight_v/=q;
		for(size_t dd = 0 ;  dd < d ; ++dd){
			weight_v -= log(this->Sigma_inv.Eigenvalues_inverted_to_use[dd]);
		}
		this-> v = weight_v;

}



template<size_t d>
void Centroid<d>::equals (const Centroid<d>& c){
	(this->X).noalias() = c.X;
	(this->m).noalias() = c.m;
	this->v = c.v;
	this->active = c.active;
	for(size_t dd=0; dd<d; ++dd){this->Sigma_inv.Eigenvalues_inverted_initial[dd] = c.Sigma_inv.Eigenvalues_inverted_initial[dd];}
	for(size_t dd=0; dd<d; ++dd){this->Sigma_inv.Eigenvalues_inverted_to_use[dd] = c.Sigma_inv.Eigenvalues_inverted_to_use[dd];}
	this->Sigma_inv.Eigenvectors = c.Sigma_inv.Eigenvectors ;
    }


template <size_t d> 
void Centroid<d>::update_X_m_v_Sigma_inverted(const std::vector<Point<d>> & points,size_t q){

		std::pair<Matrix<double,1,d>, Matrix<double,d,d>> mean_cov_Voronoi = calcul_mean_cov(points, this->Voronoi_points_indices);
		this-> X = mean_cov_Voronoi.first;

		std::vector<double> Dist_to_center;
		for (typename vector<Point<d>>::const_iterator it = points.begin(); it != points.end(); ++it) {
			Dist_to_center.push_back(this->Sigma_inv.mahalanobis_distance(it->X - mean_cov_Voronoi.first));
		}
		vector<size_t> indices_ellipsoid(points.size());
		iota(indices_ellipsoid.begin(), indices_ellipsoid.end(), 0);
		stable_sort(indices_ellipsoid.begin(), indices_ellipsoid.end(), [&Dist_to_center](size_t i1, size_t i2) {return Dist_to_center[i1] < Dist_to_center[i2];});

		indices_ellipsoid.resize(q);

		std::pair<Matrix<double,1,d>, Matrix<double,d,d>> mean_cov_ellipsoid = calcul_mean_cov(points, indices_ellipsoid);
		this-> m =  mean_cov_ellipsoid.first;

		double weight_v = 0;
		for (typename vector<size_t>::const_iterator it = indices_ellipsoid.begin(); it != indices_ellipsoid.end(); ++it) {
			weight_v+=this->Sigma_inv.mahalanobis_distance(points[*it].X - mean_cov_ellipsoid.first);
		}
		weight_v/=q;
		for(size_t dd = 0 ;  dd < d ; ++dd){
			weight_v -= log(this->Sigma_inv.Eigenvalues_inverted_to_use[dd]);
		}
		this-> v = weight_v;

		Matrix<double,d,d> A =  mean_cov_Voronoi.second + mean_cov_ellipsoid.second + (mean_cov_Voronoi.first - mean_cov_ellipsoid.first).transpose()*(mean_cov_Voronoi.first - mean_cov_ellipsoid.first) ;
		SelfAdjointEigenSolver<Matrix<double,d,d>> eigensolver(A);
		this->Sigma_inv.Eigenvectors = eigensolver.eigenvectors();
		for(size_t dd = 0 ;  dd < d ; ++dd){
			this->Sigma_inv.Eigenvalues_inverted_initial[dd] = 1/eigensolver.eigenvalues()[dd]; // inverse of the eigenvalues to get the inverse of the covariance matrix.
			this->Sigma_inv.Eigenvalues_inverted_to_use[dd] = this->Sigma_inv.Eigenvalues_inverted_initial[dd];
		}
}

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

			Class Sigma_inverted -- Functions definitions

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */




template<size_t d>
double Sigma_inverted<d>::mahalanobis_distance(const Matrix<double,1,d> &vect) const{ // can be used with vect1-vect2
	Matrix<double,1,d> temp;
	temp.noalias() = (vect)*this->Eigenvectors;
	double distance = 0;
	for(size_t dd = 0; dd<d ; ++dd){distance+=this->Eigenvalues_inverted_to_use[dd]*temp[dd]*temp[dd];}
	return(distance);
};

template<size_t d>
Sigma_inverted<d>::Sigma_inverted() :
	Eigenvectors(Matrix<double,d,d>::Identity())
	{
	for(size_t dd = 0 ;  dd < d ; ++dd){
		Eigenvalues_inverted_initial[dd] = 1;
		Eigenvalues_inverted_to_use[dd] = 1;
	}
	}

template<size_t d>
Sigma_inverted<d>::Sigma_inverted(double (&Eigenvalues_inverted)[d]) :
	Eigenvectors(Matrix<double,d,d>::Identity())
	{
	for(size_t dd = 0 ;  dd < d ; ++dd){
		Eigenvalues_inverted_initial[dd] = Eigenvalues_inverted[dd];
		Eigenvalues_inverted_to_use[dd] = Eigenvalues_inverted[dd];
	}
	}

template<size_t d>
Sigma_inverted<d>::Sigma_inverted(Matrix<double,d,d> Eigenvectors) :
	Eigenvectors(Eigenvectors)
	{
	for(size_t dd = 0 ;  dd < d ; ++dd){
		Eigenvalues_inverted_initial[dd] = 1;
		Eigenvalues_inverted_to_use[dd] = 1;
	}
	}

template<size_t d>
Sigma_inverted<d>::Sigma_inverted(double (&Eigenvalues_inverted)[d], Matrix<double,d,d> Eigenvectors) :
	Eigenvectors(Eigenvectors)
	{
	for(size_t dd = 0 ;  dd < d ; ++dd){
		Eigenvalues_inverted_initial[dd] = Eigenvalues_inverted[dd];
		Eigenvalues_inverted_to_use[dd] = Eigenvalues_inverted[dd];
	}
	}

template <size_t d> 
double Sigma_inverted<d>::trunc_eigenvalues(size_t d_intrinsic,double sigma2_min,double sigma2_max,double lambda_min,double lambda_max){
	for(size_t dd = 0; dd<d_intrinsic ; ++dd)
	{
		this-> Eigenvalues_inverted_to_use[dd] = (this-> Eigenvalues_inverted[dd] - lambda_min)*(this-> Eigenvalues_inverted[dd] >= lambda_min) + lambda_min;
	}
	double S = 0;
	if(d_intrinsic<d)
	{
		for(size_t dd = d_intrinsic; dd<d ; ++dd)
		{
			S += this-> Eigenvalues_inverted[dd];
		}
		S /= (d-d_intrinsic);
		double sigma2 = (S - sigma2_min - sigma2_max)*(sigma2_min<S)*(S<sigma2_max) + (-sigma2_max)*(S<=sigma2_min) + (-sigma2_min)*(S>=sigma2_max) + sigma2_min + sigma2_max;
		for(size_t dd = d_intrinsic; dd<d ; ++dd)
		{
			this-> Eigenvalues_inverted_to_use[dd] = sigma2;
		}		
	}
}

template <size_t d> 
double Sigma_inverted<d>::trunc_eigenvalues_sigma2_const(size_t d_intrinsic,double sigma2, double lambda_min,double lambda_max){
	for(size_t dd = 0; dd<d_intrinsic ; ++dd)
	{
		this-> Eigenvalues_inverted_to_use[dd] = (this-> Eigenvalues_inverted[dd] - lambda_min)*(this-> Eigenvalues_inverted[dd] >= lambda_min) + lambda_min;
	}
	for(size_t dd = d_intrinsic; dd<d ; ++dd)
	{
		this-> Eigenvalues_inverted_to_use[dd] = sigma2;
	}
}

template <size_t d> 
double Sigma_inverted<d>::eigenvalues_det_to_1(){
	double product_eigenvalues = 1;
	for(size_t dd = 0; dd<d; ++dd)
	{
		product_eigenvalues*=this-> Eigenvalues_inverted[dd];
	}
	for(size_t dd = 0; dd<d ; ++dd)
	{
		this-> Eigenvalues_inverted_to_use[dd] = (this-> Eigenvalues_inverted[dd])/pow(product_eigenvalues ,1/double(d));;
	}	
}




/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


								GLOBAL FUNCTIONS


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/






/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


								Read and Write files functions


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

// Read file

template <size_t d>
vector<Point<d>> readcsv(char* adresse) {
    vector<Point<d>> points;
    string line;
    ifstream file(adresse);
    if (!file.is_open()) {
	throw std::invalid_argument("failed to open the pointset file \n");
    }

    else{
    size_t index = 0;

    while (getline(file, line)) {
        stringstream lineStream(line);
        string bit;
        double X[d];
	for(size_t i = 0 ; i<d-1 ; ++i){
		getline(lineStream, bit, ',');
        	X[i] = stod(bit);
	}
        getline(lineStream, bit, '\n');
        X[d-1] = stod(bit);

        points.push_back(Point<d>(X,index));
	index++;
    }
    }
    return points;
}


/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


								Auxiliary functions


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/



// For function update_X_m_v_Sigma_inverted

template <size_t d> 
std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov(const std::vector<Point<d>> & points, const std::vector<size_t> & indices){
	Matrix<double,1,d> mean(Matrix<double,1,d>::Zero());
	for (typename std::vector<size_t>::const_iterator it = indices.begin() ; it != indices.end() ; ++it) {
	    mean += points[*it].X ;
	}
	mean/= indices.size();

	Matrix<double,d,d> cov(Matrix<double,d,d>::Zero());
	for (typename std::vector<size_t>::const_iterator it = indices.begin() ; it != indices.end() ; ++it) {
	    cov += (points[*it].X - mean).transpose() * (points[*it].X - mean) ;
	}
	cov/= indices.size();

	return(std::make_pair(mean, cov));
}

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							Class kPLM


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */



template <size_t d> 
class kPLM {

private:

	std::vector<Point<d>> points;
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

	// Une fonction pour supprimer les centroids avec un poids trop gros. (Ca va modifier optimal_centroids, optimal_labels, optimal_costs et optimal_cost).
	
//void remove_centroids(double threshold); // centroids with weight inferior to threshold are removed - Labels of points are updated (optimal_centroids; optimal_labels; optimal_costs;optimal_cost modified) --- Use after a write_centroids (the weights are in this file)

	// Fonctions pour mettre à jour les paramètres.
	void update_parameters(size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max);
	void update_parameters_principal(size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs);
	void update_parameters_secondary(size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max);

	// Fonction pour avoir accès aux paramètres.
	std::tuple<size_t,size_t,size_t,size_t,size_t,size_t,double,double,double,double,bool,size_t> parameters_access();

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
	void initialise_centroids();
//void update_initial_centroids(std::vector<Centroid<d>> initial_centroids);

	// Une fonction simple pour calculer optimal_centroids; optimal_labels; optimal_costs; optimal_cost.
	void run_kPLM();

	// Une fonction pour calculer les optimaux pour différentes valeurs de d_intrinsic, en commençant avec d_intrinsic = d_max, puis diminuant, jusqu'à tomber sur d_min. (d_max = d et d_min = 0 par exemple).
//void run_kPLM(size_t d_max, size_t d_min);

	// Une fonction d'un point pour donner sa valeur de kPLM (en fonction de optimal_centroids).
//double compute_kPLM_value(Point<d> p);

	// Une fonction pour ajouter du clustering hiérarchique... ???
	// TO WRITE LATER...

};


















/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							Class kPLM -- functions definitions


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */



template<size_t d>
kPLM<d>::kPLM(char* points_file) :
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
		this-> n_signal = (this->points).size();
	}


template<size_t d>
kPLM<d>::kPLM(char* points_file, size_t k, size_t q, size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max) :
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
void kPLM<d>::update_parameters(size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max){
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
void kPLM<d>::update_parameters_secondary(size_t k , size_t q , size_t n_signal, size_t n_starts, size_t epochs, size_t d_intrinsic, double sigma2_min, double sigma2_max, double lambda_min, double lambda_max){
	this->d_intrinsic = d_intrinsic;
	this->sigma2_min = sigma2_min;
	this->sigma2_max = sigma2_max;
	this->lambda_min = lambda_min;
	this->lambda_max = lambda_max;
	this->optimisation_done = 0;
}

template<size_t d>
std::tuple<size_t,size_t,size_t,size_t,size_t,size_t,double,double,double,double,bool,size_t> kPLM<d>::parameters_access(){
	return(make_tuple(this->k, this->q, this->n_signal, this->n_starts, this->epochs, this->d_intrinsic, this->sigma2_min, this->sigma2_max, this->lambda_min, this->lambda_max, this->normalized_det,this->points.size()));
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
		myfile << it->v << "," << it->active << endl;
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
		std::cout << it->v << "," << it->active << endl;
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

template<size_t d>
void kPLM<d>::initialise_centroids(){
	if(k < 1)throw std::invalid_argument("WARNING: There are no centroids since k<1. \n");
	if(points.size() < 1)throw std::invalid_argument("WARNING:  There are no points, so no centroids to initialise.\n");
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<size_t> distrib(0, points.size()-1);
	srand(time(0));
	for (size_t kk = 0; kk < k; ++kk) {
		Centroid<d> C(points.at(distrib(gen)));
		initial_centroids.push_back(C);
	}
}


	// Une fonction simple pour calculer optimal_centroids; optimal_labels; optimal_costs; optimal_cost.
template<size_t d>
void kPLM<d>::run_kPLM()
{	

	vector<Centroid<d>> opt_centroids(k);
	double opt_cost = __DBL_MAX__;

	for (size_t nst = 0 ; nst < n_starts ; ++nst)
	{
		initialise_centroids();
		vector<Centroid<d>> centroids = initial_centroids;
		vector<Centroid<d>> old_centroids = initial_centroids;
		

// Test

	for (typename vector<Centroid<d>>::const_iterator it = centroids.begin() ; it != centroids.end(); ++it) {

		for(size_t dd = 0 ; dd<d ; ++dd) std::cout << it->X[dd] << "," ;
		for(size_t dd = 0 ; dd<d ; ++dd) std::cout << it->m[dd] << "," ;
		std::cout << it->v << "," << it->active << endl;

	Matrix<double,d,d> Mat ;
	Matrix<double,d,d> Diag(Matrix<double,d,d>::Zero());

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



		for(typename std::vector<Centroid<d>>::iterator it = centroids.begin(); it!= centroids.end(); ++it)
		{
			it->update_m_v(q,points);
		}
	

		double cost_step = __DBL_MAX__;
		double previous_cost_step = 0;

		size_t nstep = 0;


		while (nstep < epochs && cost_step!=previous_cost_step){
			++nstep;
			previous_cost_step = cost_step;

			// reinitialisation of minDist
			for (typename vector<Point<d>>::iterator it = begin(points); it != end(points); ++it) {
				it->minDist = __DBL_MAX__;
			}


			// Associate the proper clusterId to each point

			for (typename vector<Centroid<d>>::iterator c = begin(centroids); c != end(centroids); ++c) {
			    if(c->active == true){
				    size_t clusterId = c - begin(centroids);

				    for (typename vector<Point<d>>::iterator it = begin(points); it != end(points); ++it) {

					double dist = it->Point<d>::distance_kPLM(*c);
					if (dist < it->minDist) {
					    it->minDist = dist;
					    it->cluster = clusterId;

					}

				    }
			    }
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

			for (typename vector<Centroid<d>>::iterator it = begin(centroids); it != end(centroids); ++it)
			{	
				(it-> Voronoi_points_indices).resize(0);
			}

			for (typename vector<Point<d>>::const_iterator it = begin(points); it != end(points); ++it)
			{
				size_t pointId = it - begin(points);
				size_t clusterId = it->cluster;
				if(clusterId<k)
				{
					centroids[clusterId].Voronoi_points_indices.push_back(pointId);
				}
			}

			// update centroids
			for (typename vector<Centroid<d>>::iterator c = begin(centroids); c != end(centroids); ++c){
				size_t clusterId = c - begin(centroids);
				if(c->Voronoi_points_indices.empty()){
					old_centroids[clusterId].active = false;
				}
				else{
					c->Centroid<d>::update_X_m_v_Sigma_inverted(points,q);
				}
			}

			
		}

		if(cost_step < opt_cost){
			opt_cost = cost_step;
			for(size_t kk=0; kk<k; ++kk){opt_centroids[kk].equals(old_centroids[kk]);} //opt_centroids = old_centroids;
			for(typename vector<Point<d>>::iterator it = begin(points); it != end(points); ++it) {
				it->opt_cluster = it->cluster;
				it->opt_minDist = it->minDist;
			}
		}	

	}

	std::vector<size_t> opt_labels;
	for (typename vector<Point<d>>::iterator it = begin(points); it != end(points); ++it) {
		opt_labels.push_back((it->opt_cluster)==k?0:(it->opt_cluster)+1);
	}

	std::vector<double> opt_costs;
	for (typename vector<Point<d>>::iterator it = begin(points); it != end(points); ++it) {
		opt_costs.push_back(it->opt_minDist);
	}

	optimal_centroids = opt_centroids;
	optimal_labels = opt_labels;
	optimal_costs = opt_costs;
	optimal_cost = opt_cost;
	optimisation_done = 1;
}



	// Une fonction pour calculer les optimaux pour différentes valeurs de d_intrinsic, en commençant avec d_intrinsic = d_max, puis diminuant, jusqu'à tomber sur d_min. (d_max = d et d_min = 0 par exemple).
//template<size_t d>
//void kPLM<d>::run_kPLM(size_t d_max, size_t d_min);

	// Une fonction d'un point pour donner sa valeur de kPLM (en fonction de optimal_centroids).
//template<size_t d>
//double kPLM<d>::compute_kPLM_value(Point<d> p);



/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


							main


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */



///// ATTENTION... MEME SUR UN EXEMPLE SIMPLE, CERTAINS POINTS NE SONT PAS DE LA BONNE COULEUR !!!!

///// Ajouter une erreur dans la lecture si on n'a pas accès à un nombre

int main(int argc, char** argv){
	const size_t d = 2;

try
{

	kPLM<d> algo1(argv[1]);
	kPLM<d> algo2(argv[1],3,4,50,1,1,2,0.01,0.01,0.01,100);

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

