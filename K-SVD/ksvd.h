#ifndef _KSVD_H_
#define _KSVD_H_

#include <opencv2/opencv.hpp>

struct ksvdPara
{
	int interation;
	int atoms;
	int featureSize;
	int debug;
	int sparsityThreshold;
};

std::vector<float> ksvdInitializeDictionary(
	const std::vector<float> trainPatches,
	ksvdPara kPara
);

void ksvdUpdateDictionary(
	std::vector<float> srcPatch,
	std::vector<float>& sparseCode,
	std::vector<float>& dictionary,
	ksvdPara kPara
);

float ksvdComputeReconstructionError(
	std::vector<float> Y,
	std::vector<float> D,
	std::vector<float> X,
	ksvdPara kPara
);

void KSVD(
	std::vector<float>& sparseCode,
	std::vector<float>& dictionary,
	std::vector<float> srcPatches,
	ksvdPara kPara
);



#endif // !_KSVD_H_
