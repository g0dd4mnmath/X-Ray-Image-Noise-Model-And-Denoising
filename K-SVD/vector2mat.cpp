#include "utilities.h"

cv::Mat Vector2Mat(
	std::vector<float> srcVec,
	int rows,
	int cols
) {
	cv::Mat dst(rows, cols, CV_32FC1);

	for (int j = 0; j < cols; ++j)
		for (int i = 0; i < rows; ++i)
			dst.at<float>(i, j) = srcVec[j * rows + i];

	return dst;
}