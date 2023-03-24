#ifndef _UTILITIES_H_
#define _UTILITIES_H_
#include <opencv2/opencv.hpp>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct Variances
{
	std::vector<float> atom;
	float variance;
};
inline bool CompareVariances(
	Variances p1, 
	Variances p2
){
	return (p1.variance < p2.variance);
}

float ComputeVariance(
	const std::vector<float> srcVec
);

std::vector<float> Mat2Vector(
	const cv::Mat src
);

cv::Mat Vector2Mat(
	std::vector<float> srcVec,
	int rows,
	int cols
);

std::vector<float> Img2Patches(
	std::vector<float> srcVec,
	int imgWidth,
	int imgHeight,
	int patchWidth,
	int patchHeight,
	int slidingStep
);

std::vector<float> Patch2Img(
	std::vector<float> dstPatches,
	std::vector<float>& weight,
	int patchWidth,
	int patchHeight,
	int numPatchX,
	int numPatchY,
	int slidingStep
);

float NormL2Vec(
	std::vector<float> const& u
);

std::vector<float> NormalizeMatrixNormL2(
	std::vector<float> srcMatrix,
	int rows,
	int cols
);

std::vector<float> NormalizeMatrix(
	std::vector<float> srcVec,
	int rows,
	int cols,
	float a,
	float b
);

void showDictionary(
	std::vector<float> dictionary, 
	int featureSize, 
	int atoms
);

std::vector<float> matrixMultiply(
	std::vector<float> A,
	std::vector<float> B,
	int rowsA,
	int colsA,
	int colsB
);

void mmul_cublas(
	std::vector<float> A, 
	cublasOperation_t transa, 
	std::vector<float> B, 
	cublasOperation_t transb, 
	std::vector<float>& C, 
	int rowsA, 
	int colsA, 
	int colsB
);

void mvmul_cublas(
	std::vector<float> A, 
	cublasOperation_t transa, 
	std::vector<float> x, 
	std::vector<float>& y, 
	int rowsA, 
	int colsA
);


#endif // !_UTILITIES_H_
