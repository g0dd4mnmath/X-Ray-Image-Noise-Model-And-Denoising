#include "readdcm.h"


cv::Mat ReadDcm(std::string filepath)
{
	cv::Mat image;
	DicomImage* img = new DicomImage(filepath.c_str());

	if (img->isMonochrome() && img->getStatus() == EIS_Normal && img != NULL)
	{
		if (img->isMonochrome())
		{
			int Width = img->getWidth();			//获得图像宽度
			int Height = img->getHeight();			//获得图像高度

			Uint16* pixelData = (Uint16*)(img->getOutputData(16));	//获得16位的图像数据指针
			if (pixelData != NULL)
			{
				image = cv::Mat::zeros(Height, Width, CV_16UC1);
				unsigned short* data = nullptr;
				for (int i = 0; i < Height; i++)
				{
					data = image.ptr<unsigned short>(i);
					for (int j = 0; j < Width; j++)
					{
						*data++ = pixelData[i * Width + j];
					}
				}
				delete img;
				return image;
			}
		}
	}
}