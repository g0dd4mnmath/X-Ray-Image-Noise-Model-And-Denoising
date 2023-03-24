#include "utilities.h"

float NormL2Vec(
	std::vector<float> const& u
) {
	float accum = 0;
	for (int i = 0; i < u.size(); ++i)
		accum += pow(u[i], 2);

	return sqrt(accum);
}