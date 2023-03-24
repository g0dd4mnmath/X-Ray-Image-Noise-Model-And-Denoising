#include "utilities.h"

std::vector<float> Img2Patches(
	std::vector<float> srcVec,
	int imgWidth,
	int imgHeight,
	int patchWidth,
	int patchHeight,
	int slidingStep
) {
	std::vector<float> patches;

	for (int i = 0; i + patchWidth <= imgWidth; i += slidingStep)
		for (int j = 0; j + patchHeight <= imgHeight; j += slidingStep)
		{
			std::vector<float> tempPatch;
			int startPatchIndex = i * imgHeight + j;

			for (int k = startPatchIndex; k < startPatchIndex + patchWidth * patchHeight; k += imgHeight)
				tempPatch.insert(tempPatch.end(), srcVec.begin() + k, srcVec.begin() + k + patchHeight);

			patches.insert(patches.end(), tempPatch.begin(), tempPatch.end());
		}

	return patches;
}