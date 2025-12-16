#include "dng_host.h"
#include "dng_image_writer.h"
#include "dng_info.h"
#include "dng_negative.h"
#include "dng_simple_image.h"
#include "dng_file_stream.h"
#include "dng_memory_stream.h"
#include "dng_xmp.h"
#include "dng_xmp_sdk.h"
#include "dng_camera_profile.h"
#include "dng_color_space.h"
#include "dng_date_time.h"
#include "dng_image.h"
#include "dng_preview.h"

#include <iostream>
#include <vector>
#include <time.h>
#include <cstring>

using namespace std;

struct DNGMetadata {
	// doesn't change for a given camera
	string cameraMake;
	string cameraModel;
	uint32_t bayerPattern;                 // 0=GRBG, 1=RGGB, 2=GBRG, 3=BGGR
	uint32 illuminant1;
	std::array<float, 9> colorMatrix1;
	std::array<float, 9> forwardMatrix1{};
	
	uint32 illuminant2;
	std::array<float, 9> colorMatrix2;
	std::array<float, 9> forwardMatrix2{};
	
	// only changes if there are special modes that change the image frame
	dng_rect activeArea;
	int cropXOrigin;
	int cropYOrigin;
	int cropWidth;
	int cropHeight;
	dng_rect fullBounds;

	// can change per ISO
	std::array<double, 8> noiseProfile;

	// can change
	float baselineExposure;

	// Per-image data
	std::optional<double> exposureTime;   // in seconds (e.g., 0.008 for 1/125s)
	std::optional<double> fNumber;        // e.g., 2.8
	std::optional<double> isoSpeed;       // e.g., 400
	std::optional<double> focalLength;    // in mm, e.g., 50.0
	uint32_t whiteLevel;
	std::array<uint16_t, 4> blackLevelPerChannel;  // R, G, G, B
	dng_vector_3 cameraNeutral;
	
	// Constructor with defaults
	DNGMetadata()
		  : cameraMake("Unknown"),
			cameraModel("Unknown"),
			whiteLevel(0),
			bayerPattern(0)
		{}
};

class DNGWriter {
private:
	dng_host host;
	
	void SetCurrentDateTime(dng_exif* exif) {
		dng_date_time_info dt;
		dng_date_time now;
		time_t rawtime;
		struct tm * timeinfo;
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		now.fYear = timeinfo->tm_year + 1900;
		now.fMonth = timeinfo->tm_mon + 1;
		now.fDay = timeinfo->tm_mday;
		now.fHour = timeinfo->tm_hour;
		now.fMinute = timeinfo->tm_min;
		now.fSecond = timeinfo->tm_sec;
		dt.SetDateTime(now);
		exif->fDateTimeOriginal = dt;
		exif->fDateTimeDigitized = dt;
	}
	
public:
	void WriteDNG(const char* outputPath,
				  const uint16_t* bayerData,
				  const DNGMetadata& metadata,
				  bool compressed = false) {
		
		try {
			// Create the negative (holds the image data and metadata)
			AutoPtr<dng_negative> negative(host.Make_dng_negative());
			
			// Set basic image dimensions
			negative->SetDefaultCropSize(dng_urational(metadata.cropWidth, 1),
										dng_urational(metadata.cropHeight, 1));
			negative->SetDefaultCropOrigin(metadata.cropXOrigin, metadata.cropYOrigin);
			negative->SetActiveArea(metadata.activeArea);
			
			// Set the color filter array pattern (Bayer)
			negative->SetColorKeys(colorKeyRed, colorKeyGreen, colorKeyBlue);
			negative->SetColorChannels(3);
			negative->SetBayerMosaic(metadata.bayerPattern);
			
			// Set baseline exposure and noise
			negative->SetBaselineExposure(metadata.baselineExposure);
//			negative->SetBaselineNoise(metadata.baselineNoise);
//			negative->SetBaselineSharpness(metadata.baselineSharpness);
			
			// Set white balance (as shot - neutral values)
			negative->SetCameraNeutral(metadata.cameraNeutral);
			
			// Set white level (same for all channels)
			negative->SetWhiteLevel(metadata.whiteLevel, 0);
			
			// Set black level per channel
			// For Bayer CFA, we need to set black levels for the 2x2 pattern
			// The order depends on the Bayer pattern
			// Map by bayerPattern: 0=GRBG, 1=RGGB, 2=GBRG, 3=BGGR
			const auto& bl = metadata.blackLevelPerChannel;
			uint16_t q0, q1, q2, q3;

			switch (metadata.bayerPattern) {
				case 0: // GRBG: [G R / B G] -> order: TL, TR, BL, BR
					q0 = bl[1]; q1 = bl[0]; q2 = bl[3]; q3 = bl[1];
					break;
				case 1: // RGGB: [R G / G B]
					q0 = bl[0]; q1 = bl[1]; q2 = bl[1]; q3 = bl[3];
					break;
				case 2: // GBRG: [G B / R G]
					q0 = bl[1]; q1 = bl[3]; q2 = bl[0]; q3 = bl[1];
					break;
				case 3: // BGGR: [B G / G R]
					q0 = bl[3]; q1 = bl[1]; q2 = bl[1]; q3 = bl[0];
					break;
				default:
					q0 = q1 = q2 = q3 = 0;
					break;
			}

			negative->SetQuadBlacks(q0, q1, q2, q3, 0);  // plane 0

			// Set camera metadata
			negative->SetModelName(metadata.cameraModel.c_str());
			negative->SetLocalName(metadata.cameraMake.c_str());
			
			// Set EXIF data
			dng_exif* exif = negative->GetExif();
			exif->fModel.Set_ASCII(metadata.cameraModel.c_str());
			exif->fMake.Set_ASCII(metadata.cameraMake.c_str());
			
			// Set optional EXIF fields if provided
			if (metadata.exposureTime.has_value()) {
				exif->fExposureTime = dng_urational(
					static_cast<uint32_t>(metadata.exposureTime.value() * 1000000),
					1000000);
			}
			
			if (metadata.fNumber.has_value()) {
				exif->fFNumber = dng_urational(
					static_cast<uint32_t>(metadata.fNumber.value() * 100),
					100);
			}
			
			if (metadata.isoSpeed.has_value()) {
				exif->fISOSpeedRatings[0] = static_cast<uint32_t>(metadata.isoSpeed.value());
				exif->fISOSpeedRatings[1] = 0;
			}
			
			if (metadata.focalLength.has_value()) {
				exif->fFocalLength = dng_urational(
					static_cast<uint32_t>(metadata.focalLength.value() * 100),
					100);
			}
			
			// Set timestamp
			SetCurrentDateTime(exif);

			// Set camera profile if color matrix is provided
			AutoPtr<dng_camera_profile> profile(new dng_camera_profile());
			profile->SetName("Camera Standard");
			
			profile->SetCalibrationIlluminant1(metadata.illuminant1);
			profile->SetColorMatrix1(ConvertToDNGColorMatrix(metadata.colorMatrix1));
			
			profile->SetCalibrationIlluminant2(metadata.illuminant2);
			profile->SetColorMatrix2(ConvertToDNGColorMatrix(metadata.colorMatrix2));
			
			if (metadata.forwardMatrix1[0] != 0) {
				profile->SetForwardMatrix1(ConvertToDNGColorMatrix(metadata.forwardMatrix1));
				profile->SetForwardMatrix2(ConvertToDNGColorMatrix(metadata.forwardMatrix2));
			}
			negative->AddProfile(profile);
			
			dng_noise_profile noiseProfile = this->ConvertToDNGNoiseProfile(metadata.noiseProfile);
			if (!noiseProfile.IsValid() || !noiseProfile.IsValidForNegative(*negative)) {
				printf("bad noise profile");
			}

			negative->SetNoiseProfile(noiseProfile);
			
			// Create a simple image to hold the Bayer data
			dng_rect bounds = metadata.fullBounds;
			AutoPtr<dng_simple_image> image(new dng_simple_image(bounds,
																  1,      // 1 plane for Bayer
																  ttShort, // 16-bit
																  host.Allocator()));
			
			// Copy Bayer data into the image
			dng_pixel_buffer buffer;
			buffer.fArea = bounds;
			buffer.fPlane = 0;
			buffer.fPlanes = 1;
			buffer.fRowStep = bounds.r - bounds.l;
			buffer.fColStep = 1;
			buffer.fPlaneStep = 1;
			buffer.fPixelType = ttShort;
			buffer.fPixelSize = 2;
			buffer.fData = const_cast<uint16_t*>(bayerData); 
			image->Put(buffer);
			
			// Set the image - cast to AutoPtr<dng_image>
			AutoPtr<dng_image> imagePtr;
			imagePtr.Reset(image.Release());
			negative->SetStage1Image(imagePtr);
			
			// Build IPTC metadata
			negative->RebuildIPTC(true);
			
			// Create the DNG writer
			dng_file_stream stream(outputPath, true);
			dng_image_writer writer;
			
			// Write the DNG file
			writer.WriteDNG(host,
						   stream,
						   *negative.Get(),
						   nullptr,  // preview list
						   dngVersion_Current,
						   compressed);
			
			std::cout << "DNG file written successfully: " << outputPath << std::endl;
			
		} catch (const dng_exception& e) {
			std::cerr << "DNG Exception: " << e.ErrorCode() << std::endl;
			throw;
		} catch (...) {
			std::cerr << "Unknown exception occurred" << std::endl;
			throw;
		}
	}
	
	void ValidateDNG(const char* path) {
		try {
			dng_host host;
			dng_file_stream stream(path);

			dng_info info;
			info.Parse(host, stream);   // parses headers and tags

			AutoPtr<dng_negative> negative(host.Make_dng_negative());
			info.PostParse(host);  // builds the negative

			std::cout << "DNG parsed successfully: " << path << std::endl;
		}
		catch (const dng_exception& e) {
			std::cerr << "DNG Exception: " << e.ErrorCode() << std::endl;
		}
		catch (...) {
			std::cerr << "Unknown error opening DNG" << std::endl;
		}
	}

	
private:
	
	dng_noise_profile ConvertToDNGNoiseProfile(std::array<double, 8> noiseCoeffs) {
		std::vector<dng_noise_function> funcs;
		funcs.reserve(4);

		vector<pair<float, float>> noisePairs;
		noisePairs.push_back(pair(noiseCoeffs[0], noiseCoeffs[1]));
		noisePairs.push_back(pair((noiseCoeffs[2] + noiseCoeffs[4]) / 2.0, (noiseCoeffs[3] + noiseCoeffs[5]) / 2.0));
		noisePairs.push_back(pair(noiseCoeffs[6], noiseCoeffs[7]));
		
		for (auto& pr : noisePairs) {
			dng_noise_function f = dng_noise_function(pr.first, pr.second);
			funcs.push_back(f);
		}

		dng_noise_profile p(funcs);
		return p;
	}
	
	dng_matrix ConvertToDNGColorMatrix(std::array<float, 9> inMatrix) {
		dng_matrix cm(3, 3);  // 3 rows, 3 columns
		for (int row = 0; row < 3; ++row) {
			for (int col = 0; col < 3; ++col) {
				cm[row][col] = inMatrix[row * 3 + col];
			}
		}
		return cm;
	}
};
