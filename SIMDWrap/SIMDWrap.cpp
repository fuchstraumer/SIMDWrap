// SIMDWrap.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <string>
#include "SIMD.h"


constexpr size_t array_size = 4000;

int main() {

	using namespace simd;
	
	std::array<float, 8 * array_size> farray0;
	std::array<float, 8 * array_size> farray1;
	std::array<float, 8 * array_size> fdest0;

	std::array<vec8, array_size> array0;
	std::array<vec8, array_size> array1;
	std::array<vec8, array_size> dest0;

	auto start_time = std::chrono::high_resolution_clock::now();

	std::iota(array0.begin(), array0.end(), 0.0f);
	std::iota(array1.begin(), array1.end(), -1 * static_cast<int>(array_size));
	std::transform(array0.begin(), array0.end(), array1.begin(), dest0.begin(), std::plus<vec8>());
	
	auto current_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<double, std::milli>(current_time - start_time);
	std::cerr << duration.count() << std::endl;


	dest0.fill(vec8(0.0f));

	start_time = std::chrono::high_resolution_clock::now();

	std::transform(array0.begin(), array0.end(), array1.begin(), dest0.begin(), std::minus<vec8>());

	current_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<double, std::milli>(current_time - start_time);
	std::cerr << duration.count() << std::endl;

	dest0.fill(vec8(0.0f));
	// 0.764586
	// 0.4135
	start_time = std::chrono::high_resolution_clock::now();

	std::iota(farray0.begin(), farray0.end(), 0.0f);
	std::iota(farray1.begin(), farray1.end(), -1 * static_cast<int>(array_size));
	std::transform(farray0.begin(), farray0.end(), farray1.begin(), fdest0.begin(), std::plus<float>());

	current_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<double, std::milli>(current_time - start_time);
	std::cerr << duration.count() << std::endl;
	return 0;
}

