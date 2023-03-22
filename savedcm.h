#ifndef _SAVE_DCM_H_
#define _SAVE_DCM_H_
#include <stdlib.h>
#include <iostream>


#include <opencv2/opencv.hpp>
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"

#pragma comment(lib, "Netapi32.lib")
#pragma comment(lib, "ws2_32.lib")

bool SaveDcm(cv::Mat src, std::string filepath);

#endif // !_SAVE_DCM_H_
