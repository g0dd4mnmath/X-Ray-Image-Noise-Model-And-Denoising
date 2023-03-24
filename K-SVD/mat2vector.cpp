#include "utilities.h"

std::vector<float> Mat2Vector(
	const cv::Mat src
) {
	cv::Mat transpose;
	cv::transpose(src, transpose);
	std::vector<float> srcVec;
	if (transpose.isContinuous())
	{
		srcVec.assign((float*)transpose.datastart, (float*)transpose.dataend);
	}
	else
	{
		for (int i = 0; i < transpose.rows; ++i)
			srcVec.insert(srcVec.end(), (float*)transpose.ptr<float>(i), (float*)transpose.ptr<float>(i) + transpose.cols);
	}
	return srcVec;
}