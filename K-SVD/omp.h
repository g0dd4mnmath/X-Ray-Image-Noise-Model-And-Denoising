#ifndef _OMP_H_
#define _OMP_H_
#include <opencv2/opencv.hpp>
#include "utilities.h"

struct ompPara
{
	int atoms;
	int featureSize;
	int sparsityLevel;
	int debug;
};

std::vector<float> omp(
	std::vector<float> srcPatch,
	std::vector<float> dictionary,
	ompPara oPara
);
#endif // !_OMP_H_
