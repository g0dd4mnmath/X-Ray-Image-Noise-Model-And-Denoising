#include "caculatessim.h"

double SSIM(
	cv::Mat src1,
	cv::Mat src2
) {
	cv::Mat meanRes, stddevRes, src12;
	cv::meanStdDev(src1, meanRes, stddevRes);
	double mean1 = meanRes.at<double>(0, 0);
	double sigma1 = stddevRes.at<double>(0, 0);

	cv::meanStdDev(src2, meanRes, stddevRes);
	double mean2 = meanRes.at<double>(0, 0);
	double sigma2 = stddevRes.at<double>(0, 0);

	src1.convertTo(src1, CV_64FC1);
	src2.convertTo(src2, CV_64FC1);
	cv::multiply(src1, src2, src12);
	double sigma12 = cv::mean(src12)[0] - mean1 * mean2;

	double c1, c2, L, k1, k2;
	k1 = 0.01;
	k2 = 0.03;
	L = pow(2, 16) - 1;
	c1 = pow(k1 * L, 2);
	c2 = pow(k2 * L, 2);
	double numerator = (2 * mean1 * mean2 + c1) * (2 * sigma12 + c2);
	double denominator = (pow(mean1, 2) + pow(mean2, 2) + c1) * (pow(sigma1, 2) + pow(sigma2, 2) + c2);
	
	
	
	return numerator / denominator;
}

double MSSIM(
	cv::Mat src1,
	cv::Mat src2,
	int patchSize
) {
	double mssim = 0.0;
	int index = 0;
	for (int i = 0; i < src1.rows - patchSize + 1; ++i)
		for (int j = 0; j < src1.cols - patchSize + 1; ++j)
		{
			cv::Mat patch1 = src1(cv::Rect(j, i, patchSize, patchSize));
			cv::Mat patch2 = src2(cv::Rect(j, i, patchSize, patchSize));
			mssim += SSIM(patch1, patch2);
			index++;
		}
	return mssim / index;

}