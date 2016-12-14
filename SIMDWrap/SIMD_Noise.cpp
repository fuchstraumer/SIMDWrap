#include "stdafx.h"
#include "SIMD_Noise.h"

namespace simd {

	ivec4 hash(ivec4 const &seed, ivec4 const& x, ivec4 const& y, ivec4 const& z) {
		ivec4 hashtmp = seed;
		hashtmp += (x * xPrime);
		hashtmp += (y * yPrime);
		hashtmp += (z * zPrime);

		hashtmp = ((hashtmp * hashtmp) * ivec4(60493)) * hashtmp;
		hashtmp = ivec4:: xor ((hashtmp >> 13), hashtmp);
		return hashtmp;
	}

	vec4 gradientcoord(ivec4 const &seed, ivec4 const &xi, ivec4 const &yi, ivec4 const &zi, vec4 const &x, vec4 const& y, vec4 const &z) {
		ivec4 hashresult = ((hash(seed, xi, yi, zi))) & ivec4(15);
		vec4 u = CastToFloat((hashresult < ivec4(8)));
		u = blendv(y, x, u);

		vec4 v = CastToFloat(hashresult < ivec4(4));
		vec4 h12o14 = CastToFloat((hashresult == ivec4(12)) | (hashresult == ivec4(14)));
		h12o14 = blendv(z, x, h12o14);
		v = blendv(h12o14, y, v);

		vec4 h1 = CastToFloat((hashresult & ivec4(1)) << 31);
		vec4 h2 = CastToFloat((hashresult & ivec4(2)) << 30);

		return (vec4:: xor (u, h1) + vec4:: xor (v, h2));
	}

	vec4 simplex(ivec4 const &seed, vec4 const &x, vec4 const &y, vec4 const &z) {
		vec4 f; f = thirdconst * ((x + y) + z);
		vec4 x0 = floor(x + f);
		vec4 y0 = floor(y + f);
		vec4 z0 = floor(z + f);

		ivec4 i = ConvertToInt(x);
		ivec4 j = ConvertToInt(y);
		ivec4 k = ConvertToInt(z);

		vec4 g = thirdconst * ConvertToFloat((i + j) + k);
		x0 = x - (x0 - g);
		y0 = y - (y0 - g);
		z0 = z - (z0 - g);

		ivec4 x0_ge_y0 = CastToInt(x0 >= y0);
		ivec4 y0_ge_z0 = CastToInt(y0 >= z0);
		ivec4 x0_ge_z0 = CastToInt(x0 >= z0);

		ivec4 i1 = oneVecI & (x0_ge_y0 & x0_ge_z0);
		ivec4 j1 = oneVecI & ivec4::andnot(x0_ge_z0, y0_ge_z0);
		ivec4 k1 = oneVecI & ivec4::andnot(x0_ge_z0, ~y0_ge_z0);

		ivec4 i2 = oneVecI & (x0_ge_y0 | x0_ge_z0);
		ivec4 j2 = oneVecI & (~x0_ge_y0 | y0_ge_z0);
		ivec4 k2 = oneVecI & (x0_ge_y0 & y0_ge_z0);

		vec4 x1 = (x0 - ConvertToFloat(i1)) + thirdconst;
		vec4 y1 = (y0 - ConvertToFloat(j1)) + thirdconst;
		vec4 z1 = (z0 - ConvertToFloat(k1)) + thirdconst;
		vec4 x2 = (x0 - ConvertToFloat(i2)) + halfconst;
		vec4 y2 = (y0 - ConvertToFloat(j2)) + halfconst;
		vec4 z2 = (z0 - ConvertToFloat(k2)) + halfconst;
		vec4 x3 = (x0 - oneVecF) + halfconst;
		vec4 y3 = (y0 - oneVecF) + halfconst;
		vec4 z3 = (z0 - oneVecF) + halfconst;

		vec4 t0 = ((vec4(0.6f) - (x0 * x0)) - (y0 * y0)) - (z0 * z0);
		vec4 t1 = ((vec4(0.6f) - (x1 * x1)) - (y1 * y1)) - (z1 * z1);
		vec4 t2 = ((vec4(0.6f) - (x2 * x2)) - (y2 * y2)) - (z2 * z2);
		vec4 t3 = ((vec4(0.6f) - (x3 * x3)) - (y3 * y3)) - (z3 * z3);

		vec4 n0 = t0 >= vec4(0.0f);
		vec4 n1 = t1 >= vec4(0.0f);
		vec4 n2 = t2 >= vec4(0.0f);
		vec4 n3 = t3 >= vec4(0.0f);

		t0 = t0 * t0;
		t1 = t1 * t1;
		t2 = t2 * t2;
		t3 = t3 * t3;

		n0 = n0 & ((t0 * t0) * gradientcoord(seed, i, j, k, x0, y0, z0));
		n1 = n1 & ((t1 * t1) * gradientcoord(seed, i + i1, j + j1, k + k1, x1, y1, z1));
		n2 = n2 & ((t2 * t2) * gradientcoord(seed, i + i2, j + j2, k + k2, x2, y2, z2));
		n3 = n2 & ((t3 * t3) * gradientcoord(seed, i + oneVecI, j + oneVecI, k + oneVecI, x3, y3, z3));

		return (vec4(32.0f, 32.0f, 32.0f, 32.0f) * (n0 + n1 + n2 + n3));
	}

	static float FBM(ivec4 const &seed, vec4 const &xi, vec4 const &yi, vec4 const &zi, float frequency, int octaves, float lacunarity, float gain) {
		vec4 sum(0.0f);
		vec4 amplitude(1.0f);
		vec4 x, y, z;
		for (int i = 0; i < octaves; ++i) {
			// Multiply initial coords by frequency to scale them to right domain
			x = xi * vec4(frequency);
			y = yi * vec4(frequency);
			z = zi * vec4(frequency);
			// Get simplex value
			vec4 n = simplex(seed, x, y, z);
			// Total simplex value is current * amplitude
			sum += n * amplitude;
			// Scale the frequency by the lacunarity since this is octaved noise
			frequency *= lacunarity;
			// Gain changes over octaves as well.
			amplitude *= gain;
		}
		return sum.Data.m128_f32[0];
	}
	

	ivec8 hash(ivec8 const &seed, ivec8 const& x, ivec8 const& y, ivec8 const& z) {
		ivec8 hashtmp = seed;
		hashtmp += (x * ivec8(1619));
		hashtmp += (y * ivec8(31337));
		hashtmp += (z * ivec8(6971));

		hashtmp = ((hashtmp * hashtmp) * ivec8(60493)) * hashtmp;
		hashtmp = ivec8::xor((hashtmp >> 13), hashtmp);
		return hashtmp;
	}

	vec8 gradientcoord(ivec8 const &seed, ivec8 const &xi, ivec8 const &yi, ivec8 const &zi, vec8 const &x, vec8 const& y, vec8 const &z) {
		ivec8 hashresult = ((hash(seed, xi, yi, zi))) & ivec8(15);
		vec8 u = ivec8::CastToFloat((hashresult < ivec8(8)));
		u = blendv(y, x, u);

		vec8 v = ivec8::CastToFloat(hashresult < ivec8(4));
		vec8 h12o14 = ivec8::CastToFloat((hashresult == ivec8(12)) | (hashresult == ivec8(14)));
		h12o14 = blendv(z, x, h12o14);
		v = blendv(h12o14, y, v);

		vec8 h1 = ivec8::CastToFloat((hashresult & ivec8(1)) << 31);
		vec8 h2 = ivec8::CastToFloat((hashresult & ivec8(2)) << 30);

		return (vec8::xor(u, h1) + vec8::xor(v, h2));
	}

	vec8 simplex(ivec8 const & seed, vec8 const & x, vec8 const & y, vec8 const & z) {
		vec8 G3(.166666667f);
		vec8 n0, n1, n2, n3; // Noise contributions from the four simplex corners
		vec8 noise; // Return value
		vec8 gx0, gy0, gz0, gx1, gy1, gz1, gx2, gy2, gz2, gx3, gy3, gz3;
		vec8 x1, y1, z1, x2, y2, z2, x3, y3, z3;
		vec8 t0, t1, t2, t3, t20, t40, t21, t41, t22, t42, t23, t43;
		vec8 temp0, temp1, temp2, temp3;
		// Skew input space to determine correct simplex cell to use
		vec8 s = (x + y + z) * G3;
		vec8 xs = x + s;
		vec8 ys = y + s;
		vec8 zs = z + s;

		ivec8 ii, i = ConvertToInt(floor(xs));
		ivec8 jj, j = ConvertToInt(floor(ys));
		ivec8 kk, k = ConvertToInt(floor(zs));

		vec8 t = ivec8::ConvertToFloat(i + j + k) * G3;
		// Bring cell back to euclidean space
		vec8 X0 = ivec8::ConvertToFloat(i) - t;
		vec8 Y0 = ivec8::ConvertToFloat(j) - t;
		vec8 Z0 = ivec8::ConvertToFloat(k) - t;
		// X,Y,Z distances from origin of this simplex cell
		vec8 x0 = x - X0;
		vec8 y0 = y - Y0;
		vec8 z0 = z - Z0;

		// Determine which simplex (not cell) we're in
		ivec8 i1, j1, k1, i2, j2, k2;

		ivec8 x0_ge_y0 = ConvertToInt(x0 >= y0);
		ivec8 y0_ge_z0 = ConvertToInt(y0 >= z0);
		ivec8 x0_ge_z0 = ConvertToInt(x0 >= z0);

		i1 = ivec8(1) & (x0_ge_y0 | x0_ge_z0);
		j1 = ivec8(1) & ivec8::and_not(x0_ge_y0, y0_ge_z0);
		k1 = ivec8(1) & ivec8::and_not(x0_ge_z0, ivec8::not(y0_ge_z0));

		i2 = ivec8(1) & (x0_ge_y0 | x0_ge_z0);
		j2 = ivec8(1) & (ivec8::not(x0_ge_y0), y0_ge_z0);
		k2 = ivec8(1) & ivec8::not(x0_ge_z0 & y0_ge_z0);

		x1 = (x0 - ivec8::ConvertToFloat(i1)) + G3;
		y1 = (y0 - ivec8::ConvertToFloat(j1)) + G3;
		z1 = (z0 - ivec8::ConvertToFloat(k1)) + G3;

		x2 = (x0 - ivec8::ConvertToFloat(i2)) + (G3 * G3);
		y2 = (y0 - ivec8::ConvertToFloat(j2)) + (G3 * G3);
		z2 = (z0 - ivec8::ConvertToFloat(k2)) + (G3 * G3);

		x3 = (x0 - vec8(1.0f)) + (G3 * G3 * G3);
		y3 = (y0 - vec8(1.0f)) + (G3 * G3 * G3);
		z3 = (z0 - vec8(1.0f)) + (G3 * G3 * G3);
		
		t0 = ( ( vec8(0.6f) - (x0 * x0) ) - (y0 * y0) ) - (z0 * z0);
		t1 = ((vec8(0.6f) - (x1 * x1)) - (y1 * y1)) - (z1 * z1);
		t2 = ((vec8(0.6f) - (x2 * x2)) - (y2 * y2)) - (z2 * z2);
		t3 = ((vec8(0.6f) - (x3 * x3)) - (y3 * y3)) - (z3 * z3);

		n0 = t0 > vec8(0.0f);
		n1 = t1 > vec8(0.0f);
		n2 = t2 > vec8(0.0f);
		n3 = t3 > vec8(0.0f);

		t0 *= t0;
		t1 *= t1;
		t2 *= t2;
		t3 *= t3;

		vec8 n00 = n0 & ((t0 * t0) * gradientcoord(seed, i, j, k, x0, y0, z0));
		vec8 n01 = n1 & ((t1 * t1) * gradientcoord(seed, i + i1, j + j1, k + k1, x1, y1, z1));
		vec8 n02 = n2 & ((t2 * t2) * gradientcoord(seed, i + i2, j + j2, k + k2, x2, y2, z2));
		vec8 n03 = n3 & ((t3 * t3) * gradientcoord(seed, i + ivec8(1), j + ivec8(1), k + ivec8(1), x3, y3, z3));

		return vec8(32.0f) * (n00 + n01 + n02 + n03);
	}

} // namespace simd