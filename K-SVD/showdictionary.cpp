#include "utilities.h"

void showDictionary(
	std::vector<float> dictionary,
	int featureSize,
	int atoms
) {
	std::vector<float> normalizedDictionary = NormalizeMatrix(dictionary, featureSize, atoms, 0, 1);

	std::vector<Variances> pairAtomVariance(atoms);
	for (int i = 0; i < atoms; ++i)
	{
		std::vector<float> tempAtom(normalizedDictionary.begin() + i * featureSize, normalizedDictionary.begin() + (i + 1) * featureSize);
		pairAtomVariance[i].atom = tempAtom;
		pairAtomVariance[i].variance = ComputeVariance(tempAtom);
	}

	std::sort(pairAtomVariance.begin(), pairAtomVariance.end(), CompareVariances);

	std::vector<float> tempDictionary;
	for (int i = 0; i < atoms; ++i)
		tempDictionary.insert(tempDictionary.end(), pairAtomVariance[i].atom.begin(), pairAtomVariance[i].atom.end());

	int patchWidth = sqrt(featureSize);
	int patchHeight = sqrt(featureSize);
	int dictionaryWidth = sqrt(atoms) * patchWidth;
	int dictionaryHeight = sqrt(atoms) * patchHeight;

	std::vector<float> weight(dictionaryWidth * dictionaryHeight, 0);
	std::vector<float> dictionaryVec = Patch2Img(tempDictionary, weight, patchWidth, patchHeight, sqrt(atoms), sqrt(atoms), patchHeight);

	cv::Mat dictionaryMat = Vector2Mat(dictionaryVec, dictionaryHeight, dictionaryWidth);

	cv::resize(dictionaryMat, dictionaryMat, cv::Size(512, 512));

	cv::imshow("dictionary", dictionaryMat);
	cv::waitKey(0);
}