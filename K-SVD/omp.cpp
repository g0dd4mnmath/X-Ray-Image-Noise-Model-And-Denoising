#include "omp.h"

std::vector<float> omp(
	std::vector<float> srcPatch,
	std::vector<float> dictionary,
	ompPara oPara
) {
	int srcPatchSize = srcPatch.size() / oPara.featureSize;
	std::vector<float> sparseCodeRes(srcPatchSize * oPara.atoms);

	float weight = 0;
	int chosenAtomIndex;

	for (int patchIndex = 0; patchIndex < srcPatchSize; ++patchIndex)
	{
		std::vector<float> tempPatch(srcPatch.begin() + patchIndex * oPara.featureSize, srcPatch.begin() + (patchIndex + 1) * oPara.featureSize);
		std::vector<float> residualVec(tempPatch.begin(), tempPatch.end());

		std::vector<float> tempSparseCode(oPara.atoms, 0);

		std::vector<int> chosenAtomIndexList;
		std::vector<float> chosenAtomList;

		int iter = 0;
		while (iter < oPara.sparsityLevel)
		{
			float maxProduct = -pow(2, 32);
			std::vector<float> proj(oPara.atoms, 0);
			mvmul_cublas(dictionary, CUBLAS_OP_T, residualVec, proj, oPara.featureSize, oPara.atoms);
			for (int atomId = 0; atomId < oPara.atoms; atomId++) {
				if (std::abs(proj[atomId]) > maxProduct) {
					maxProduct = std::abs(proj[atomId]);
					chosenAtomIndex = atomId;
				}
			}
			weight = proj[chosenAtomIndex];
			std::vector<float> chosenAtom(dictionary.begin() + chosenAtomIndex * oPara.featureSize, dictionary.begin() + (chosenAtomIndex + 1) * oPara.featureSize);
			chosenAtomIndexList.push_back(chosenAtomIndex);
			chosenAtomList.insert(chosenAtomList.end(), chosenAtom.begin(), chosenAtom.end());
			
			cv::Mat chosenAtomMat = Vector2Mat(chosenAtomList, oPara.featureSize, chosenAtomIndexList.size());
			cv::Mat tempPatchMat = Vector2Mat(tempPatch, oPara.featureSize, 1);
			cv::Mat chosenAtomMatInvert;
			cv::invert(chosenAtomMat, chosenAtomMatInvert, cv::DECOMP_SVD);
			cv::Mat weightList;
			weightList = chosenAtomMatInvert * tempPatchMat;

			for (int i = 0; i < chosenAtomIndexList.size(); ++i)
			{
				int tempAtomIndex = chosenAtomIndexList[i];
				tempSparseCode[tempAtomIndex] = weightList.at<float>(i);
			}

			cv::Mat tempMat = chosenAtomMat * weightList;
			std::vector<float> tempVec = Mat2Vector(tempMat.clone());
			std::transform(tempPatch.begin(), tempPatch.end(), tempVec.begin(), residualVec.begin(), std::minus<float>());

			if (NormL2Vec(residualVec) < 0.001)
				break;

			iter++;
		}

		sparseCodeRes.insert(sparseCodeRes.begin() + patchIndex * oPara.atoms, tempSparseCode.begin(), tempSparseCode.end());
	}
	return sparseCodeRes;
}