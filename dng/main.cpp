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
#include <string>
#include <optional>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>

// This project uses nlohmann/json (https://github.com/nlohmann/json)
// Licensed under the MIT License.

#include "DNGWriter.hpp"
#include "json.hpp"

std::string g_inputPath;
std::string g_outputPath;
bool g_same_path = false;

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

/*
 For this project: [camera]/[indoor|outdoor]/[scene-number]_[scene_name]_[focal_length]/[scene-number]_[scene_name]_[focal_length]_[ISO]_[clean|noisy]_[number]
 */

struct RAWFileInfo {
	string rawDataFullPath;
	string rawMetadataFullPath;
	string subdirectory;		// used to create subfolders underneath the output directory
};

typedef std::pair<std::string, std::string> rawMetadataPair;

std::vector<rawMetadataPair> findFiles(std::string inputDirectory)
{
	vector<rawMetadataPair> result;
	
	if (!fs::exists(inputDirectory) || !fs::is_directory(inputDirectory)) {
		std::cerr << "Error: " << inputDirectory << " is not a valid directory\n";
		return result;
	}

	vector<string> rawDataFilePaths;

	for (const auto& entry : fs::directory_iterator(inputDirectory)) {
		if (entry.is_regular_file() && entry.path().extension() == ".rawdata") {
			rawDataFilePaths.push_back(entry.path());
		}
		else if (entry.is_directory()) {
			auto subItems = findFiles(entry.path());
			result.insert(result.end(), subItems.begin(), subItems.end());
		}
	}
	
	for (const auto& rawDataPath : rawDataFilePaths) {
		fs::path rawPath(rawDataPath);
		fs::path metaPath = rawPath;
		metaPath.replace_extension(".json");

		if (fs::exists(metaPath)) {
			rawMetadataPair pr(rawPath.string(), metaPath.string());
			result.push_back(pr);
		} else {
			std::cerr << "Warning: metadata file missing for " << rawPath << "\n";
		}
	}

	return result;
}

void parseArgs(int argc, char* argv[]) {
	std::optional<std::string> inputPath;
	std::optional<std::string> outputPath;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-i" && i + 1 < argc) {
			inputPath = argv[++i];  // consume next argument
		} else if (arg == "-o" && i + 1 < argc) {
			outputPath = argv[++i]; // consume next argument
		} else {
			std::cerr << "Unknown or incomplete argument: " << arg << "\n";
			return 1;
		}
	}

	if (!inputPath) {
		std::cerr << "Error: -i <path> is required\n";
		return 1;
	}

	std::cout << "Input path: " << *inputPath << "\n";
	if (outputPath) {
		std::cout << "Output path: " << *outputPath << "\n";
	} else {
		std::cout << "No output path provided\n";
	}

	if (inputPath.has_value()) {
		g_inputPath = inputPath.value();
		g_outputPath = g_inputPath;
	}
	
	if (outputPath.has_value()) {
		g_outputPath = outputPath.value();
	}
	else {
		g_same_path = true;
	}
}

void from_json(const json& j, DNGMetadata& md) {
	j.at("bayer_pattern").get_to(md.bayerPattern);
	j.at("iso").get_to(md.isoSpeed);
	j.at("shutter_speed").get_to(md.exposureTime);
	
	vector<float> camera_neutral;
	j.at("as_shot").get_to(camera_neutral);
	md.cameraNeutral = dng_vector_3(camera_neutral[0], camera_neutral[1], camera_neutral[2]);
	
	j.at("aperture").get_to(md.fNumber);
	j.at("focal_length").get_to(md.focalLength);
	j.at("white_level").get_to(md.whiteLevel);
	
	vector<unsigned short> black_levels;
	j.at("black_level_per_channel").get_to(black_levels);
	md.blackLevelPerChannel = {black_levels[0], black_levels[1], black_levels[2], black_levels[3]};
	
	vector<double> noise_levels;
	j.at("noise_profile").get_to(noise_levels);
	md.noiseProfile = {noise_levels[0], noise_levels[1], noise_levels[2], noise_levels[3],
		noise_levels[4], noise_levels[5], noise_levels[6], noise_levels[7]};

	j.at("camera_model").get_to(md.cameraModel);
	j.at("camera_make").get_to(md.cameraMake);
	
	j.at("illuminant1").get_to(md.illuminant1);
	
	vector<float> matrix;
	
	j.at("color_matrix1").get_to(matrix);
	md.colorMatrix1 = {matrix[0], matrix[1], matrix[2], matrix[3],
		matrix[4], matrix[5], matrix[6], matrix[7], matrix[8]};

	j.at("illuminant2").get_to(md.illuminant2);

	j.at("color_matrix2").get_to(matrix);
	md.colorMatrix2 = {matrix[0], matrix[1], matrix[2], matrix[3],
		matrix[4], matrix[5], matrix[6], matrix[7], matrix[8]};

	// optional
	try {
		j.at("forward_matrix1").get_to(matrix);
		md.forwardMatrix1 = {matrix[0], matrix[1], matrix[2], matrix[3],
			matrix[4], matrix[5], matrix[6], matrix[7], matrix[8]};
		
		j.at("forward_matrix2").get_to(matrix);
		md.forwardMatrix2 = {matrix[0], matrix[1], matrix[2], matrix[3],
			matrix[4], matrix[5], matrix[6], matrix[7], matrix[8]};
	}
	catch(...) {
		
	}
	
	// dng_rect = t, l, b, r

	vector<unsigned short> activeArea;
	j.at("active_area").get_to(activeArea);
	md.activeArea = dng_rect(activeArea[0], activeArea[1], activeArea[2], activeArea[3]);

	vector<unsigned short> cropOrigin;
	j.at("crop_origin").get_to(cropOrigin);
	md.cropXOrigin = cropOrigin[0];
	md.cropYOrigin = cropOrigin[1];

	vector<unsigned short> cropSize;
	j.at("crop_size").get_to(cropSize);
	md.cropWidth = cropSize[0];
	md.cropHeight = cropSize[1];

	vector<unsigned short> fullBounds;
	j.at("bounds").get_to(fullBounds);
	md.fullBounds = dng_rect(fullBounds[0], fullBounds[1], fullBounds[2], fullBounds[3]);

//	j.at("baseline_noise").get_to(md.baselineNoise);
//	j.at("baseline_sharpness").get_to(md.baselineSharpness);
	j.at("baseline_exposure").get_to(md.baselineExposure);
}

void makeTestDNG() {
	// Example: Create a simple gradient Bayer pattern for testing
	const uint32_t width = 5504;
	const uint32_t height = 3672;
	
	std::vector<uint16_t> bayerData(width * height);
	
	// Generate a simple test pattern
	for (uint32_t row = 0; row < height; row++) {
		for (uint32_t col = 0; col < width; col++) {
			// Simple gradient
			uint16_t value = static_cast<uint16_t>((row * 65535) / height);
			bayerData[row * width + col] = value;
		}
	}
	
	DNGWriter writer;
	DNGMetadata metadata = DNGMetadata();
	metadata.cameraMake = "SONY";
	metadata.cameraModel = "Sony DSC-RX100M4";
	metadata.activeArea = dng_rect(0, 0, height, width);
	metadata.cropXOrigin = 12;
	metadata.cropYOrigin = 12;
	metadata.cropWidth = 5472;
	metadata.cropHeight = 3648;
	metadata.fullBounds = dng_rect(0, 0, height, width);
	
	metadata.bayerPattern = 1;	// Map by bayerPattern: 0=GRBG, 1=RGGB, 2=GBRG, 3=BGGR
	metadata.illuminant1 = 17;
	metadata.colorMatrix1 = {(0.736600),
		(-0.321300),
		(0.038000),
		(-0.360900),
		(1.112700),
		(0.285200),
		(-0.021800),
		(0.069400),
		(0.582100),
	};
	metadata.forwardMatrix1 = {0.7978,
		0.1352,
		0.0313,
		0.288,
		0.7119,
		0.0001,
		0,
		0,
		0.8251,
	};

	metadata.illuminant2 = 21;
	metadata.colorMatrix2 = {(0.659600),
		(-0.207900),
		(-0.056200),
		(-0.478200),
		(1.301600),
		(0.193300),
		(-0.097000),
		(0.158100),
		(0.518100),
	};
	metadata.forwardMatrix2 = {0.7978,
		0.1352,
		0.0313,
		0.288,
		0.7119,
		0.0001,
		0,
		0,
		0.8251,
	};

	metadata.noiseProfile = {
		(8.107881e-05),
		(5.553180e-08),
		(8.291752e-05),
		(5.470989e-08),
		(8.291752e-05),
		(5.470989e-08),
		(8.180209e-05),
		(5.460511e-08),
	};

	metadata.baselineExposure = -0.25;

	metadata.exposureTime = 1.0/125.0;
	metadata.fNumber = 2.8;
	metadata.isoSpeed = 400;
	metadata.focalLength = 50;
	metadata.cameraNeutral = dng_vector_3(0.38209, 1, 0.544681);
	metadata.whiteLevel = 16300;
	metadata.blackLevelPerChannel = {5812, 6473, 6473, 4490};
	writer.WriteDNG("output_basic.dng",
					bayerData.data(),
					metadata);
	writer.ValidateDNG("output_basic.dng");
}

int makeDNG(string rawDataPath, string rawMetadataPath, string outputDirectory)
{
	// 1. open the raw data into memory
	std::ifstream in(rawDataPath, std::ios::binary);
	if (!in) {
		std::cerr << "Failed to open " << rawDataPath << "\n";
		return 1;
	}

	// Get file size
	in.seekg(0, std::ios::end);
	std::streamsize size = in.tellg();
	in.seekg(0, std::ios::beg);

	// Number of 16-bit elements
	size -= 8;		// first 8 bytes are height and width
	std::size_t count = size / sizeof(uint16_t);
	
	// size height, width, but as 32 bit quantities
	std::streamsize heightByteCount = 4;
	uint32_t height = -1;
	uint32_t width = -1;
	in.read(reinterpret_cast<char*>(&height), heightByteCount);
	in.read(reinterpret_cast<char*>(&width), heightByteCount);
	std::vector<uint16_t> bayerData(count);

	// Read raw bytes directly into vector
	in.read(reinterpret_cast<char*>(bayerData.data()), size);

	
	// 2. open the metadata json file and convert into values for the writer
	std::ifstream file(rawMetadataPath);
	json data = json::parse(file);
	DNGMetadata md = data.get<DNGMetadata>();
	
	
	// 3. output full file path
	fs::path rawPath(rawDataPath);
	fs::path outputPath;
	
	// maintain the hierarchy
	if (g_same_path) {
		outputPath = rawPath;
		outputPath.replace_extension(".dng");
	}
	else {
		fs::path outputDirectoryPath(outputDirectory);
		fs::path newFileName = rawPath.stem();
		if (!fs::exists(outputDirectoryPath)) {
			fs::create_directories(outputDirectoryPath);
		}
		newFileName.replace_extension(".dng");
		fs::path outputPath = outputDirectoryPath / newFileName;
	}
	

	// 4. Call the writer.
	DNGWriter writer;
	writer.WriteDNG(outputPath.c_str(),
					bayerData.data(),
					md);
	writer.ValidateDNG(outputPath.c_str());
	return 0;
}

int main(int argc, char* argv[])
{
	auto start = std::chrono::high_resolution_clock::now();

	bool deleteInputFiles = true;
	parseArgs(argc, argv);
	std::vector<rawMetadataPair> pairs = findFiles(g_inputPath);
	
	std::sort(pairs.begin(), pairs.end(),
			  [](const auto& a, const auto& b) {
				  return a.first < b.first;
			  });

	for (const auto& pr : pairs) {
		makeDNG(pr.first, pr.second, g_outputPath);
		if (deleteInputFiles) {
			fs::remove(pr.first);
			fs::remove(pr.second);
		}
	}
    
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Elapsed time: " << elapsed.count() / 1000.0 << " s\n";

	return 0;
}
