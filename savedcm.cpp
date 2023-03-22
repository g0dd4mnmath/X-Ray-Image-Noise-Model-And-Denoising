#include "savedcm.h"

bool SaveDcm(cv::Mat src, std::string filepath)
{
	char uid[100];
	DcmFileFormat fformat;
	DcmDataset* dataset = fformat.getDataset();

	DcmTag tag;
	dataset->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
	dataset->putAndInsertString(DCM_SOPInstanceUID, dcmGenerateUniqueIdentifier(uid, SITE_INSTANCE_UID_ROOT));
	dataset->putAndInsertString(DCM_PatientName, "Wang^Justin");

	Uint16 samples_per_pixel, planar_configuration, rows, columns, pixel_representation;
	std::string photometric_interpretation;
	
	rows = src.rows;
	columns = src.cols;
	if (src.type() != CV_16UC1)
	{
		std::cout << "only support 16-bit image!" << "\n";
		return false;
	}
	pixel_representation = 0;
	if (src.channels() == 1)
	{
		samples_per_pixel = src.channels();
		photometric_interpretation = "MONOCHROME2";
	}
		
	else
	{
		std::cout << "only support gray image!" << "\n";
		return false;
	}

	dataset->putAndInsertUint16(DCM_SamplesPerPixel, samples_per_pixel);
	dataset->putAndInsertString(DCM_PhotometricInterpretation, photometric_interpretation.c_str(), static_cast<uint32_t>(photometric_interpretation.length()));

	dataset->putAndInsertUint16(DCM_Rows, rows);
	dataset->putAndInsertUint16(DCM_Columns, columns);
	dataset->putAndInsertUint16(DCM_BitsAllocated, src.elemSize1() * 8);

	dataset->putAndInsertUint16(DCM_BitsStored, src.elemSize1() * 8);
	dataset->putAndInsertUint16(DCM_HighBit, src.elemSize1() * 8 - 1);
	dataset->putAndInsertUint16(DCM_PixelRepresentation, pixel_representation);
	dataset->putAndInsertString(DCM_LossyImageCompression, "00");
	
	dataset->putAndInsertUint8Array(DCM_PixelData, src.data, src.size().height * src.size().width*2);

	OFCondition status = fformat.saveFile(filepath.c_str(), EXS_LittleEndianExplicit);
	if (status.bad())
		return false;
	
	return true;
}