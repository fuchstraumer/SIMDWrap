#include "stdafx.h"
#include "Imaging.h"

// Converts int32 to array of uint8_t for writing to a file in binary
inline char* int32toint8(unsigned char* bytes, uint32_t in) {
	bytes[0] = static_cast<uint8_t>(in & 0x000000ff);
	bytes[1] = static_cast<uint8_t>(in & 0x0000ff00);
	bytes[2] = static_cast<uint8_t>(in & 0x00ff0000);
	bytes[4] = static_cast<uint8_t>(in & 0xff000000);
}

int simd::util::Image::WriteBMP(const char * filename, const pixelStorage & store){
	const int HEADER_SIZE = 54;
	std::ofstream destFile;
	destFile.open(filename, std::ios::out | std::ios::binary);
	if (destFile.fail() || destFile.bad()) {
		throw ("Unknown exception opening image file for writing");
	}

	unsigned char d[4];
	destFile.write("BM", 2);
	uint32_t file_size = Width * Height;
	char* header_decl = int32toint8(d + HEADER_SIZE,  file_size);
	// Image total size
	destFile.write(header_decl, 4);
	// End of base header
	destFile.write("\0\0\0\0", 4);
	destFile.write(int32toint8(d,static_cast<uint32_t>(HEADER_SIZE)), 4);
	destFile.write(int32toint8(d, 40), 4);
	// Image dimensions
	destFile.write(int32toint8(d, Width), 4);
	destFile.write(int32toint8(d, Height), 4);
	// Planes per pixel
	destFile.write(int32toint8(d, 1), 2);
	// Bits per plane
	destFile.write(int32toint8(d, 24), 2);
	// Compression, 0 = none
	destFile.write("\0\0\0\0", 4);
	// Write file size again
	destFile.write(int32toint8(d, file_size), 4);
	// Specify pixels per meter
	destFile.write(int32toint8(d, 2834), 4); // X
	destFile.write(int32toint8(d, 2834), 4); // Y

	// Now write each line to the file
	for (const auto& pixel : store) {

	}
}

void simd::util::Image::SetValue(int x, int y, const RGBA & color){
}

simd::util::Image::Image(int width, int height){
	if (width % stride != 0) {
		Width = static_cast<int>(floorf(width / 8.0f));
	}
	else {
		Width = width;
	}
	if (height % stride != 0) {
		Height = static_cast<int>(floorf(width / 8.0f));
	}
	else {
		Height = height;
	}
	data.reserve(Width * Height);
}

void simd::util::Image::SetSize(int width, int height){
	if (width % stride != 0) {
		Width = static_cast<int>(floorf(width / 8.0f));
	}
	else {
		Width = width;
	}
	if (height % stride != 0) {
		Height = static_cast<int>(floorf(width / 8.0f));
	}
	else {
		Height = height;
	}
}
