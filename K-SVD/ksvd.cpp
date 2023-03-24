#include "ksvd.h"
#include "utilities.h"
#include "omp.h"

std::vector<float> ksvdInitializeDictionary(
	const std::vector<float> trainPatches,
	ksvdPara kPara
) {
	std::vector<float> dictionary;

	for (int i = 0; i < kPara.atoms; ++i)
		dictionary.insert(dictionary.end(), trainPatches.begin() + i * kPara.featureSize, trainPatches.begin() + (i + 1) * kPara.featureSize);

	std::vector<float> normalizedDictionary = NormalizeMatrixNormL2(dictionary, kPara.featureSize, kPara.atoms);

	return normalizedDictionary;
}

void ksvdUpdateDictionary(
	std::vector<float> srcPatch,
	std::vector<float>& sparseCode,
	std::vector<float>& dictionary,
	ksvdPara kPara
) {
	for (int atomIndex = 0; atomIndex < kPara.atoms; ++atomIndex)
	{
		std::vector<int> relevantDataIndices;
		for (int i = 0; i < sparseCode.size() / kPara.atoms; ++i)
		{
			if (sparseCode[i * kPara.atoms + atomIndex] != 0)
				relevantDataIndices.push_back(i);
		}

		if (relevantDataIndices.size() >= 1)
		{
			std::vector<float> selectInput;
			std::vector<float> selectSparseCode;

			for (int i = 0; i < relevantDataIndices.size(); ++i)
			{
				int inputIndex = relevantDataIndices[i];
				selectInput.insert(selectInput.end(), srcPatch.begin() + inputIndex * kPara.featureSize, srcPatch.begin() + (inputIndex + 1) * kPara.featureSize);
				selectSparseCode.insert(selectSparseCode.end(), sparseCode.begin() + inputIndex * kPara.atoms, sparseCode.begin() + (inputIndex + 1) * kPara.atoms);
			}

			int numSelectInput = selectInput.size() / kPara.featureSize;
			for (int i = 0; i < numSelectInput; ++i)
				selectSparseCode[i * kPara.atoms + atomIndex] = 0;

			std::vector<float> errorMatrix(selectInput.size());
			std::vector<float> DX = matrixMultiply(dictionary, selectSparseCode, kPara.featureSize, kPara.atoms, numSelectInput);
			std::transform(selectInput.begin(), selectInput.end(), DX.begin(), errorMatrix.begin(), std::minus<float>());

			cv::Mat E = Vector2Mat(errorMatrix, kPara.featureSize, numSelectInput);

			cv::Mat S, U, Vt;
			cv::SVD::compute(E, S, U, Vt);
			cv::Mat V = Vt.t();

			cv::Mat betterAtom;
			betterAtom = (-1) * U.col(0).clone();
			cv::Mat betterCoef = (-1) * S.at<float>(0, 0) * V.col(0);

			std::vector<float> firstCol = Mat2Vector(betterAtom);

			std::copy(firstCol.begin(), firstCol.end(), dictionary.begin() + atomIndex * kPara.featureSize);

			for (int i = 0; i < relevantDataIndices.size(); i++)
			{
				int inputIndex = relevantDataIndices[i];
				sparseCode[inputIndex * kPara.atoms + atomIndex] = betterCoef.at<float>(i);
			}
		}
	}
}

float ksvdComputeReconstructionError(
	std::vector<float> Y,
	std::vector<float> D,
	std::vector<float> X,
	ksvdPara kPara
) {
	int colsX = X.size() / kPara.atoms;
	std::vector<float> DX = matrixMultiply(D, X, kPara.featureSize, kPara.atoms, colsX);

	std::vector<float> errorMatrix(Y.size());
	std::transform(Y.begin(), Y.end(), DX.begin(), errorMatrix.begin(), std::minus<float>());

	float accError = 0;
	for (int i = 0; i < errorMatrix.size(); ++i)
		accError += pow(errorMatrix[i], 2);

	float resError = sqrt(accError / errorMatrix.size());
	return resError;
}

void KSVD(
	std::vector<float>& sparseCode,
	std::vector<float>& dictionary,
	std::vector<float> srcPatches,
	ksvdPara kPara
) {
	ompPara oPara;
	oPara.atoms = kPara.atoms;
	oPara.featureSize = kPara.featureSize;
	oPara.sparsityLevel = kPara.sparsityThreshold;

	for (int it = 0; it < kPara.interation; ++it)
	{
		sparseCode = omp(srcPatches, dictionary, oPara);

		ksvdUpdateDictionary(srcPatches, sparseCode, dictionary, kPara);

		float resError = ksvdComputeReconstructionError(srcPatches, dictionary, sparseCode, kPara);
	}
}