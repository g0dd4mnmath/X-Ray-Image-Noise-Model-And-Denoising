#include "utilities.h"

float ComputeVariance(
	const std::vector<float> srcVec
) {
	float mean = 0;
	float sum = 0;

	for (int i = 0; i < srcVec.size(); i++) {
		sum = sum + srcVec[i];
	}
	mean = sum / srcVec.size();

	float temp = 0;
	for (int i = 0; i < srcVec.size(); i++) {
		temp = temp + pow(srcVec[i] - mean, 2);
	}
	return temp / (srcVec.size());
}