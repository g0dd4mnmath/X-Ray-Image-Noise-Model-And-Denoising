#include "getnoiselevel.h"

cv::Mat convmtx(
	cv::Mat H,
	int m,
	int n
) {
	cv::Mat result = cv::Mat::zeros(cv::Size(m * n, (m - H.rows + 1) * (n - H.cols + 1)), H.type());
	
	for (int i = 0, p = 0; i < m - H.rows + 1; ++i)
		for (int j = 0; j < n - H.cols + 1; ++j, ++p)
			for (int k = 0; k < H.rows; ++k)
				for (int l = 0; l < H.cols; ++l)
					result.at<float>(p, (i + k) * n + j + l) = H.at<float>(k, l);
				
	return result;
}

cv::Mat im2col(
	cv::Mat src,
	cv::Size patchSize
) {
	if (src.type() == CV_32FC1)
	{
		cv::Mat result = cv::Mat::zeros(cv::Size((src.cols - patchSize.width + 1) * (src.rows - patchSize.height + 1), patchSize.area()), src.type());

		for (int i = 0, countx = 0; i < src.cols - patchSize.width + 1; ++i)
			for (int j = 0; j < src.rows - patchSize.height + 1; ++j, ++countx)
				for (int k = 0, county = 0; k < patchSize.width; ++k)
					for (int l = 0; l < patchSize.height; ++l, ++county)
						result.at<float>(county, countx) = src.at<float>(j + l, i + k);

		return result;
	}
	else if (src.type() == CV_16UC1)
	{
		cv::Mat result = cv::Mat::zeros(cv::Size((src.cols - patchSize.width + 1) * (src.rows - patchSize.height + 1), patchSize.area()), src.type());

		for (int i = 0, countx = 0; i < src.cols - patchSize.width + 1; ++i)
			for (int j = 0; j < src.rows - patchSize.height + 1; ++j, ++countx)
				for (int k = 0, county = 0; k < patchSize.width; ++k)
					for (int l = 0; l < patchSize.height; ++l, ++county)
						result.at<ushort>(county, countx) = src.at<ushort>(j + l, i + k);
		return result;
	}
	return cv::Mat();
}

void sortcols(
	cv::Mat src,
	cv::Mat& dst
) {
	cv::Mat srcT;
	cv::transpose(src, srcT);
	src.release();
	cv::Mat index;
	cv::sortIdx(srcT, index, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
	
	cv::Mat dstT = cv::Mat::zeros(srcT.size(), srcT.type());
	for (int i = 0; i < index.rows; ++i)
	{
		srcT.row(index.at<int>(i, 0)).copyTo(dstT.row(i));
	}
	
	cv::transpose(dstT, dst);
}

float GetNoiseLevel(
	cv::Mat src,
	int patchSize,
	int iter
) {
	src.convertTo(src, CV_32FC1);
	
	cv::Mat dsth = cv::Mat::zeros(src.size(), src.type());
	cv::Mat dstv = cv::Mat::zeros(src.size(), src.type());
	cv::Mat kernelh = cv::Mat::zeros(cv::Size(3, 1), CV_32FC1);
	cv::Mat kernelv = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
	kernelh.at<float>(0, 0) = -0.5;
	kernelh.at<float>(0, 1) = 0.0;
	kernelh.at<float>(0, 2) = 0.5;
	cv::transpose(kernelh, kernelv);

	cv::filter2D(src, dsth, CV_32FC1, kernelh);
	cv::filter2D(src, dstv, CV_32FC1, kernelv);

	dsth = dsth(cv::Rect(cv::Point(1, 0), cv::Point(src.cols - 1, src.rows )));
	dstv = dstv(cv::Rect(cv::Point(0, 1), cv::Point(src.cols, src.rows - 1)));

	cv::pow(dsth, 2, dsth);
	cv::pow(dstv, 2, dstv);
	

	//Éú³É¾í»ý¾ØÕó
	cv::Mat Dh = convmtx(kernelh, patchSize, patchSize);
	cv::Mat Dv = convmtx(kernelv, patchSize, patchSize);
	kernelh.release();
	kernelv.release();
	

	cv::Mat DDh, DDv, DD;
	cv::mulTransposed(Dh, DDh, true);
	cv::mulTransposed(Dv, DDv, true);
	DD = DDh + DDv;
	Dh.release();
	Dv.release();
	DDh.release();
	DDv.release();
	

	cv::Mat w, u, vt;
	cv::SVD::compute(DD, w, u, vt);
	cv::Mat nonZero = w > 0.0001;
	w.release();
	u.release();
	vt.release();

	int rank = cv::countNonZero(nonZero);
	float trace = cv::trace(DD)[0];
	nonZero.release();
	DD.release();
	float tau0 = 54.4109;
	
	
	cv::Mat X = im2col(src, cv::Size(patchSize, patchSize));
	cv::Mat Xh = im2col(dsth, cv::Size(patchSize - 2, patchSize));
	cv::Mat Xv = im2col(dstv, cv::Size(patchSize, patchSize - 2));
	dsth.release();
	dstv.release();

	cv::Mat Xtr;
	cv::vconcat(Xh, Xv, Xtr);
	cv::reduce(Xtr, Xtr, 0, cv::REDUCE_SUM);
	Xh.release();
	Xv.release();
	
	
	cv::Mat XtrX;
	cv::vconcat(Xtr, X, XtrX);
	sortcols(XtrX, XtrX);
	Xtr.release();
	X.release();
	

	int decim = 3;
	int p = (XtrX.cols / (decim + 1));
	

	cv::Mat pMat = cv::Mat::zeros(cv::Size(p, 1), CV_32SC1);
	for (int i = 0; i < pMat.cols; ++i)
		pMat.at<int>(i) = (i + 1) * (decim + 1);
		
	cv::Mat XtrNew = cv::Mat::zeros(pMat.size(), XtrX.type());
	for (int i = 0; i < pMat.cols; ++i)
		XtrNew.at <float>(i) = XtrX.at <float>(0, pMat.at<int>(i) - 1);
	
	
	cv::Mat XNew = cv::Mat::zeros(cv::Size(pMat.cols, XtrX.rows - 1), XtrX.type());
	for (int i = 0; i < pMat.cols; ++i)
		for (int j = 1, k = 0; j < XtrX.rows; ++j, ++k)
			XNew.at<float>(k, i) = XtrX.at<float>(j, pMat.at<int>(i) - 1);

	XtrX.release();
	pMat.release();

	
	float tau = 0.99;
	float sigma;
	cv::Mat cov, eigenValues;
	if (XNew.cols < XNew.rows)
		sigma = 0;
	else
	{
		cv::mulTransposed(XNew, cov, false);
		cov = cov / (float)(XNew.cols - 1);
		cv::eigen(cov, eigenValues);
		//cv::sort(eigenValues, eigenValues, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
		sigma = eigenValues.at<float>(eigenValues.rows - 1);
	}

	for (int i = 0; i < iter - 1; ++i)
	{
		tau = sigma * tau0;
		
		cv::threshold(XtrNew, XtrNew, tau, 1, cv::THRESH_TOZERO_INV);
		std::vector<cv::Point> index;
		cv::findNonZero(XtrNew, index);
		cv::Mat Xtrtemp = cv::Mat::zeros(cv::Size(index.size(), XtrNew.rows), XtrNew.type());
		for (int j = 0; j < index.size(); ++j)
			XtrNew.col(index[j].x).copyTo(Xtrtemp.col(j));
		XtrNew = Xtrtemp;
		Xtrtemp.release();


		cv::Mat Xtemp = cv::Mat::zeros(cv::Size(index.size(), XNew.rows), XNew.type());
		for (int j = 0; j < index.size(); ++j)
			XNew.col(index[j].x).copyTo(Xtemp.col(j));
		XNew = Xtemp;
		Xtemp.release();

		if (XNew.cols < XNew.rows)
			break;

		cv::mulTransposed(XNew, cov, false);
		cov = cov / (XNew.cols - 1);
		cv::eigen(cov, eigenValues);
		cv::sort(eigenValues, eigenValues, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
		sigma = eigenValues.at<float>(0);
		index.clear();
	}

	
	return sqrt(sigma);
}

