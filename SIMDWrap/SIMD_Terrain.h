#ifndef SIMD_TERRAIN_H
#define SIMD_TERRAIN_H
#include "SIMD_SSE.h"
#define f3const vec4(0.3333333f,0.3333333f,0.3333333f,0.3333333f)
#define g3const vec4(0.1666667f,0.1666667f,0.1666667f,0.1666667f)
#define oneVecF vec4(1.0f,1.0f,1.0f,1.0f)
#define oneVecI ivec4(1,1,1,1)
namespace simd {
	
	vec4 simplex(ivec4 const &seed, vec4 const &x, vec4 const &y, vec4 const &z) {
		vec4 f; f = f3const * ((x + y) + z);
		vec4 x0 = vec4::floor(x + f);
		vec4 y0 = vec4::floor(y + f);
		vec4 z0 = vec4::floor(z + f);

		ivec4 i = vec4::convertToInt(x);
		ivec4 j = vec4::convertToInt(y);
		ivec4 k = vec4::convertToInt(z);

		vec4 g = g3const * ivec4::ConvertToFloat((i + j) + k);
		x0 = x - (x0 - g);
		y0 = y - (y0 - g);
		z0 = z - (z0 - g);

		ivec4 x0_ge_y0 = ivec4::CastToInt(x0 >= y0);
		ivec4 y0_ge_z0 = ivec4::CastToInt(y0 >= z0);
		ivec4 x0_ge_z0 = ivec4::CastToInt(x0 >= z0);

		ivec4 i1 = oneVecI & (x0_ge_y0 & x0_ge_z0);
		ivec4 j1 = oneVecI & ivec4::andnot(x0_ge_z0, y0_ge_z0);
		ivec4 k1 = oneVecI & ivec4::andnot(x0_ge_z0, ~y0_ge_z0);

		ivec4 i2 = oneVecI & (x0_ge_y0 | x0_ge_z0);
		ivec4 j2 = oneVecI & (~x0_ge_y0 | y0_ge_z0);
		ivec4 k2 = oneVecI & (x0_ge_y0 & y0_ge_z0);
	}
}


#endif // !SIMD_TERRAIN_H
