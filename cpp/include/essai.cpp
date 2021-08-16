
#include <iostream>  // for file-reading




int main(int argc, char** argv){

	size_t nb = 0;
	size_t k = 3;
	for(size_t opt_cluster=0; opt_cluster<4; ++opt_cluster)
	{
	nb = (opt_cluster)==k?0:(opt_cluster)+1 ;
	std::cout << nb << std::endl;
	}
	return(0);

}
