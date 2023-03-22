#ifndef _GET_NOISE_LEVEL_H_
#define _GET_NOISE_LEVEL_H_
#include <opencv2/opencv.hpp>

float GetNoiseLevel(
	cv::Mat src,
	int patchSize,
	int iter
);
#endif // !_GET_NOISE_LEVEL_H_
