#include "utilities.h"

std::vector<float> NormalizeMatrixNormL2(
	std::vector<float> srcMatrix,
	int rows,
	int cols
) {
	std::vector<float> dstMatrix;

	for (int i = 0; i < cols; ++i)
	{
		std::vector<float> colVec(srcMatrix.begin() + i * rows, srcMatrix.begin() + (i + 1) * rows);
		float acc = 0;

		for (int j = 0; j < colVec.size(); ++j)
			acc += pow(colVec[j], 2);

		float normL2 = sqrt(acc);

		for (int j = 0; j < colVec.size(); ++j)
			colVec[j] /= normL2;
		
		dstMatrix.insert(dstMatrix.end(), colVec.begin(), colVec.end());
	}

	return dstMatrix;
}