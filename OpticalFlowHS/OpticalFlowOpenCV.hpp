#include "cv.h"
#include "highgui.h"
#include <math.h>
#include <iostream>

class OpticalFlowOpenCV
{
public:
	int runFromImg(char* input1, char* input2, char* output, float lambda, int it);
	int runFromCamera(float lambda, int it);
};