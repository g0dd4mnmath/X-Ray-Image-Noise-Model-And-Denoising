#include "generalizedanscombetransformation.h"

void GeneralizedAnscombeTransformation_Forward(
	cv::Mat src,
	cv::Mat& dst,
	double sigma,
	double alpha,
	double mu
) {
	//dst.convertTo(dst, src.type());
	if (src.type() != CV_64FC1)
		src.convertTo(src, CV_64FC1);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			GeneralizedAnscombeTransformation_Forward(src.at<double>(i, j), sigma, alpha, mu);
		}
	src.convertTo(dst, CV_32FC1);
}

void GeneralizedAnscombeTransformation_Forward(
	double& pixelValue,
	double sigma,
	double alpha,
	double mu
) {
	pixelValue = (pixelValue - mu) / alpha;
	sigma = sigma / alpha;
	pixelValue = 2.0 * sqrt(MAX(0, pixelValue + (3.0 / 8.0) + pow(sigma, 2)));
}

void GeneralizedAnscombeTransformation_ClosedFormInverse(
	cv::Mat src,
	cv::Mat& dst,
	double sigma,
	double alpha,
	double mu
) {
	//dst.convertTo(dst, src.type());
	if (src.type() != CV_64FC1)
		src.convertTo(src, CV_64FC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			GeneralizedAnscombeTransformation_ClosedFormInverse(src.at<double>(i, j), sigma, alpha, mu);
		}
	src.convertTo(dst, CV_32FC1);
}

void GeneralizedAnscombeTransformation_ClosedFormInverse(
	double& pixelValue,
	double sigma,
	double alpha,
	double mu
) {
	sigma = sigma / alpha;
	pixelValue = MAX(0.0, pow(pixelValue, 2) / 4.0 + 1.0 / 4.0 * sqrt(3.0 / 2.0) * pow(pixelValue, -1) - 11.0 / 8.0 * pow(pixelValue, -2) + 5.0 / 8.0 * sqrt(3.0 / 2.0) * pow(pixelValue, -3) - 1.0 / 8.0 - pow(sigma, 2));
	pixelValue = pixelValue * alpha + mu;
}

void AnscombeTransformation_Forward(
	cv::Mat src,
	cv::Mat& dst
) {
	dst.convertTo(dst, src.type());
	if (src.type() != CV_64FC1)
		src.convertTo(src, CV_64FC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			AnscombeTransformation_Forward(src.at<double>(i, j));
		}
	src.convertTo(dst, dst.type());
}

void AnscombeTransformation_Forward(
	double& pixelValue
) {
	pixelValue = sqrt(pixelValue);
}

void AnscombeTransformation_Inverse(
	cv::Mat src,
	cv::Mat& dst
) {
	dst.convertTo(dst, src.type());
	if (src.type() != CV_64FC1)
		src.convertTo(src, CV_64FC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			AnscombeTransformation_Forward(src.at<double>(i, j));
		}
	src.convertTo(dst, dst.type());
}

void AnscombeTransformation_Inverse(
	double& pixelValue
) {
	pixelValue = pow(pixelValue, 2);
}