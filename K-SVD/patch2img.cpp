#include "utilities.h"

std::vector<float> Patch2Img(
	std::vector<float> dstPatches,
	std::vector<float>& weight,
	int patchWidth,
	int patchHeight,
	int numPatchX,
	int numPatchY,
	int slidingStep
) {
	int imgWidth = patchWidth + slidingStep * (numPatchX - 1);
	int imgHeight = patchHeight + slidingStep * (numPatchY - 1);
	int patchSize = patchWidth * patchHeight;
	std::vector<float> dstVec(imgWidth * imgHeight, 0);
	std::vector<float> blockWeight(patchWidth * patchHeight, 1);

	int patchIndex = 0;
	for (int i = 0; i + patchIndex <= imgWidth; i += slidingStep)
		for (int j = 0; j + patchHeight <= imgHeight; j += slidingStep)
		{
			int startPatchIndex = i * imgHeight + j;
			std::vector<float> tempPatch(dstPatches.begin() + patchIndex * patchSize, dstPatches.begin() + (patchIndex + 1) * patchSize);

			int colIndex = 0;
			for (int k = startPatchIndex; k < startPatchIndex + patchWidth * imgHeight; k += imgHeight)
			{
				std::transform(tempPatch.begin() + colIndex * patchHeight, tempPatch.begin() + (colIndex + 1) * patchHeight,
					dstVec.begin() + k, dstVec.begin() + k, std::plus<float>());
				std::transform(blockWeight.begin() + colIndex * patchHeight, blockWeight.begin() + (colIndex + 1) * patchHeight,
					weight.begin() + k, weight.begin() + k, std::plus<float>());
				colIndex++;
			}
			patchIndex++;
		}
	return dstVec;
}