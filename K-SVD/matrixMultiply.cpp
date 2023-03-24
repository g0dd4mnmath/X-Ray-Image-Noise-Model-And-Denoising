#include "utilities.h"

std::vector<float> matrixMultiply(
	std::vector<float> A,
	std::vector<float> B,
	int rowsA,
	int colsA,
	int colsB
) {
	std::vector<float> C(rowsA * colsB, 0);
	int rowsB = colsA;
	int rowsC = rowsA;
	int colsC = colsB;
	for (int j = 0; j < colsC; ++j)
		for (int i = 0; i < rowsC; ++i)
			for (int k = 0; k < colsA; ++k)
				C[j * rowsC + i] = C[j * rowsC + i] + A[i + k * rowsA] * B[j * rowsB + k];
			
	return C;
}