#include "utilities.h"

std::vector<float> NormalizeMatrix(
	std::vector<float> srcVec,
	int rows,
	int cols,
	float a,
	float b
) {
	std::vector<float> dstVec;

	for (int col = 0; col < cols; ++col)
	{
		std::vector<float> oneCol(srcVec.begin() + col * rows, srcVec.begin() + (col + 1) * rows);
		std::vector<float> result(oneCol.size());

		float min = 99999;
		float max = -99999;

		for (int i = 0; i < oneCol.size(); ++i)
		{
			if (oneCol[i] > max)
				max = oneCol[i];
			if (oneCol[i] < min)
				min = oneCol[i];
		}

		for (int i = 0; i < oneCol.size(); ++i)
			result[i] = (oneCol[i] - min) * (b - a) / (max - min) + a;

		dstVec.insert(dstVec.end(), result.begin(), result.end());
	}

	return dstVec;
}