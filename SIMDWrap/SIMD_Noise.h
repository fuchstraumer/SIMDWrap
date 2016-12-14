#ifndef SIMD_NOISE_H
#define SIMD_NOISE_H

#include "SIMD_Math.h"
#include "SIMD_Constants.h"
#include <random>
#include <functional>
namespace simd {


// Following are noise functions for SSE3/4.1 intrinsics.

#ifdef SIMD_LEVEL_SSE3
#include "SIMD_SSE.h"
	// "Ease curve" for noise values.
	vec4 quintic_interp(vec4 const& input);
	
	ivec4 hash(ivec4 const &seed, ivec4 const& x, ivec4 const& y, ivec4 const& z);

	vec4 gradientcoord(ivec4 const &seed, ivec4 const &xi, ivec4 const &yi, ivec4 const &zi, vec4 const &x, vec4 const& y, vec4 const &z);
	
	vec4 simplex(ivec4 const &seed, vec4 const &x, vec4 const &y, vec4 const &z);

	float FBM(ivec4 const &seed, vec4 const &xi, vec4 const &yi, vec4 const &zi, float frequency, int octaves, float lacunarity, float gain);

	vec4 RidgedMulti(ivec4 const &seed, vec4 const &xi, vec4 const &yi, vec4 const &zi, float frequency, int octaves, float lacunariy, float gain);

#endif // SIMD_LEVEL_SSE3


#ifdef SIMD_LEVEL_AVX2
#include "SIMD_AVX.h"
	// Following are noise functions for AVX2 instructions.
	// Operate on vec8's

	static __forceinline ivec8 hash(ivec8 const& seed, ivec8 const &x, ivec8 const &y, ivec8 const &z);

	static ivec8 gradient(ivec8 const& seed, ivec8 const &xi, ivec8 const &yi, ivec8 const &zi, ivec8 const &x, ivec8 const &y, ivec8 const &z);

	static __forceinline vec8 simplex(ivec8 const& seed, vec8 const &x, vec8 const &y, vec8 const &z);


	class NoiseMap {
	public:
		NoiseMap() = default;

		virtual void Build() = 0;

		void SetCallback(std::function<vec8(ivec8 const& seed, vec8 const &x, vec8 const &y, vec8 const &z)>& noise_func);

	protected:
		std::function<vec8(ivec8 const& seed, vec8 const &x, vec8 const &y, vec8 const &z)> noiseFunc;
	};

	// Simple 2D planar map

	class PlaneMap : public NoiseMap {
	public:
		PlaneMap(int x_min, int x_max, int y_min, int y_max);

		virtual void Build() override;

		std::pair<int, int> GetXBounds() const;

		std::pair<int, int> GetYBounds() const;

		void SetXBounds(int x_min, int x_max);

		void SetYBounds(int y_min, int y_max);

	private:
		std::pair<int, int> xBounds, yBounds;
	};

	// Map of noise values to a cylinder, bounded by angular values given and the height values given

	class CylinderMap : public NoiseMap {
	public:
		CylinderMap(double lower_angle, double upper_angle, double lower_height, double upper_height);

		virtual void Build() override;

		std::pair<double, double> GetHeightBounds() const;

		std::pair<double, double> GetAngularBounds() const;

		void SetAngularBounds(double lower_angle, double upper_angle);

		void SetHeightBounds(double lower_height, double upper_height);

	private:
		std::pair<double, double> angularBounds, heightBounds;
	};

	// Map of noise values for a sphere, bounded by the four spherical coordinates specified
	class SphereMap : public NoiseMap {
	public:
		SphereMap(double east_long_bound, double west_long_bound, double north_lat_bound, double south_lat_bound);

		virtual void Build() override;

		std::pair<double, double> GetLongitudionalBounds() const;

		std::pair<double, double> GetLattitudionalBounds() const;

		void SetLongitudionalBounds(double east_long, double west_long);

		void SetLattitudionalBounds(double north_lat, double south_lat);
	};

#endif // SIMD_LEVEL_AVX2

}


/*

	SIMD Noise map/storage class useable by any/all types of instructions.

*/

#endif // !SIMD_TERRAIN_H
