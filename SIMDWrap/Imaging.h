#pragma once
#ifndef IMAGING_H
#define IMAGINE_H
#include "SIMD.h"
#include <vector>
#include <fstream>
#ifdef SIMD_LEVEL_AVX2

namespace simd {

	namespace util {

		const int RASTER_MAX_DIMENSION = 32767;

	
		
		// Base struct defining a color, without alpha channel
		using RGBv = struct rgb_v {
			float red[8];
			float green[8];
			float blue[8];
		};

		// Base struct defining a color, with alpha channel
		// this is for 8 total RGBA pixels
		using RGBAv = struct rgba_v {
			float red[8];
			float green[8];
			float blue[8];
			float alpha[8];
		};

		using RGBA = struct rgba {
			float red;
			float green;
			float blue;
			float alpha;
		};

		using Pixel = struct pixel {
			RGBA Color;
			int x, y;

			pixel(int _x, int _y, RGBA color) {
				x = _x;
				y = _y;
				Color = color;
			}
		};

		using pixelStorage = std::vector<Pixel>;

		class Image {
		public:
			Image() = default;
			// Build image with given w/h
			// Round width/height down to be a multiple of 8 due to SIMD constraints
			Image(int width, int height);
			// No copying allowed
			Image(const Image& rhs) = delete;

			~Image() = default;
			// Clears image to rgba values given
			void Clear(const float& r, const float& g, const float& b, const float &a);

			int GetHeight() const;

			int GetWidth() const;

			size_t GetMemUsed() const;

			void SetSize(int width, int height);

			int ReadBMP(const char *filename);

			// Write to a bmp at filename using storage "store"
			int WriteBMP(const char *filename, const pixelStorage& store);

			// Sets the color value at the given position
			void SetValue(int x, int y, const RGBA& color);

			// Use SIMD sets to set the color values at the given positions
			void SetValue(const ivec8 &x, const ivec8 &y, const RGBAv& colors);

		private:
			// Using AVX2 instructions, the stride will be 8 pixels at a time
			const int stride = 8;
			// Width/height. These are rounded down to be a whole multiple of 8 
			// upon image init
			unsigned int Width, Height;
			// Storage/data for this image
			pixelStorage data;
		};

	}
}

#endif // SIMD_LEVEL_AVX2

#endif // !IMAGING_H
