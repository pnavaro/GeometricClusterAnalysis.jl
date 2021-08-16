/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Claire Brécheteau
 *
 *    Copyright (C) 2021 Université Rennes 2
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */


#ifndef POINT_H_
#define POINT_H_

#include <Eigen/Dense> 
#include <fstream>   // for file-reading
#include <vector>

using namespace std;
using namespace Eigen;

template <size_t d> 
class Point;

template <size_t d> 
class Centroid;

template <size_t d>
class kPLM;

template <size_t d> 
std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov(const std::vector<Point<d>> & points, const std::vector<size_t> & indices);

template <size_t d> 
std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov(const Matrix<double,Dynamic,d> & block_points, const std::vector<size_t> & indices);



template<size_t d>
Matrix<double,Dynamic,d> block_points(const std::vector<Point<d>> & points);


// class definition


template <size_t d> 
class Point {

//private:
public:
	Matrix<double,1,d> X;     // coordinates
	size_t cluster;     // no default cluster
	double minDist;  // default infinite ; distance to the nearest cluster (for the cost function)
	size_t opt_cluster; // cluster id for the best attempt
	double opt_minDist; // distance to the nearest cluster for the best attempt

	//double distance(const Centroid<d> &c) const; //squared Euclidean distance

public:
	Point();
	Point(const double (&tab)[d]);
	Point(const Matrix<double,1,d> & tab);

	Matrix<double,1,d> get_X();

	friend class Centroid<d>;
	friend class kPLM<d>;

	friend std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov<d>(const std::vector<Point<d>> & points, const std::vector<size_t> & indices);
	friend std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov<d>(const Matrix<double,Dynamic,d> & block_points, const std::vector<size_t> & indices);

	friend Matrix<double,Dynamic,d> block_points<d>(const std::vector<Point<d>> & points);
};


// auxiliary functions definition


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
Point<d>::Point(const double (&tab)[d]) : 
        cluster(0),
	opt_cluster(0),
	opt_minDist(__DBL_MAX__),
        minDist(__DBL_MAX__)
	{
	for(size_t dd = 0 ; dd < d ; ++dd) X[dd] = tab[dd];
	}

template<size_t d>
Point<d>::Point(const Matrix<double,1,d> & tab) : 
	X(tab),
        cluster(0),
	opt_cluster(0),
	opt_minDist(__DBL_MAX__),
        minDist(__DBL_MAX__)
	{
	}

template<size_t d>
Matrix<double,1,d> Point<d>::get_X(){
	return(this->X);
}


// friend functions definition

template <size_t d> 
std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov(const std::vector<Point<d>> & points, const std::vector<size_t> & indices){
	Matrix<double,1,d> mean(Matrix<double,1,d>::Zero());
	for (typename std::vector<size_t>::const_iterator it = indices.begin() ; it != indices.end() ; ++it) {
	    mean.noalias() += points[*it].X ;
	}
	mean/= indices.size();

	Matrix<double,d,d> cov(Matrix<double,d,d>::Zero());
	for (typename std::vector<size_t>::const_iterator it = indices.begin() ; it != indices.end() ; ++it) {
	    cov.noalias() += (points[*it].X - mean).transpose() * (points[*it].X - mean) ;
	}
	cov/= indices.size();

	return(std::make_pair(mean, cov));
}



/// EN COURS !!!!
template <size_t d> 
std::pair<Matrix<double,1,d>, Matrix<double,d,d>> calcul_mean_cov(const Matrix<double,Dynamic,d> & block_points, const std::vector<size_t> & indices)
{
	Matrix<double,1,d> mean(Matrix<double,1,d>::Zero());
	for (typename std::vector<size_t>::const_iterator it = indices.begin() ; it != indices.end() ; ++it) {
	    mean.noalias() += block_points.row(*it) ;
	}
	mean/= indices.size();

	Matrix<double,d,d> cov(Matrix<double,d,d>::Zero());
	for (typename std::vector<size_t>::const_iterator it = indices.begin() ; it != indices.end() ; ++it)
	{
		Matrix<double,Dynamic,d> centered_data = block_points.row(*it) - mean; 
		cov.noalias() += centered_data.transpose() * centered_data ;
	}
	cov/= indices.size();

	return(std::make_pair(mean, cov));
}
// If we took all indices :
//	Matrix<double,1,d> mean = block_points.colwise().mean();
//	Matrix<double,Dynamic,d> centered = block_points.rowwise() - mean;
//	Matrix<double,d,d> cov = (centered.adjoint() * centered) / double(block_points.rows());

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

        points.push_back(Point<d>(X));
    }
    }
    return points;
}


template <size_t d>
Matrix<double,Dynamic,d> block_points(const std::vector<Point<d>> & points){
	Matrix<double,Dynamic,d> block(points.size(), d);
	size_t index = 0;
	for(typename std::vector<Point<d>>::const_iterator it = points.begin(); it!=points.end(); ++it)
	{
		block.row(index) << it->X;
		++index;	
	}	
	return(block);
}


#endif  // POINT_H_
