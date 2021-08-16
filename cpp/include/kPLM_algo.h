
#ifndef kPLM_ALGO_H_
#define kPLM_ALGO_H_


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

using namespace std;
using namespace Eigen;

#include "kPLM/Point.h"
#include "kPLM/Sigma_inverted.h"
#include "kPLM/Centroid.h"
#include "kPLM/kPLM.h"


#endif  // kPLM_ALGO_H_
