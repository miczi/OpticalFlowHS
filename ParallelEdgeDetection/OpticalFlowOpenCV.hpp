#include "cv.h"
#include "highgui.h"
#include <math.h>
#include <iostream>

class OpticalFlowOpenCV
{
protected:
	int times[1];
	int count;
	int time;
	int timeIdx;
public:
	int run();
	void mi();
	void mr();
	void ms();
	void me();
	void mes();
	void md();
};