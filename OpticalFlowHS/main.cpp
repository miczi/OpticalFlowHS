#include "HSOpticalFlowOpenCL.hpp"
#include "OpticalFlowOpenCV.hpp"

void runCLDisk(int argc, char * argv[]);
void runCLCamera(int argc, char * argv[]);
void runCVDisk(int argc, char * argv[]);
void runCVCamera(int argc, char * argv[]);

int argccld = 10;
char* argvcld[10] = {"null", "-cl", "-hd", "city_1.jpg", "city_2.jpg", "city_cl_out.jpg", "15", "10", "1", "GPU"};
int argcclc = 7;
char* argvclc[7] = {"null", "-cl", "-cam", "50", "500", "8", "CPU"};
int argccvd = 8;
char* argvcvd[8] = {"null", "-cv", "-hd", "city_1.jpg", "city_2.jpg", "city_cv_out.jpg", ".1", "10"};
int argccvc = 5;
char* argvcvc[5] = {"null", "-cv", "-cam", ".05", "500"};

int	main(int argc, char * argv[])
{
	//runCLDisk(argccld, argvcld);
	//runCVDisk(argccvd, argvcvd);	
	runCLCamera(argcclc, argvclc);
	//runCVCamera(argccvc, argvcvc);
	getchar();
	return 0;

	if (argc == 1){
		std::cout<<"Nie zostaly podane zadne parametry do funkcji main.\nKoncze dzialanie!\n";
		return 0;
	}
	if (strcmp(argv[1], "-cl") == 0){
		if (strcmp(argv[2], "-hd") == 0){
			if (argc != 10){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				runCLDisk(argc, argv);
			}
		}
		else if (strcmp(argv[2], "-cam") == 0){
			if (argc != 7){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				runCLCamera(argc, argv);
			}
		}
	}
	else if (strcmp(argv[1], "-cv") == 0)
	{
		if (strcmp(argv[2], "-hd") == 0)
		{
			if (argc != 8)
			{
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				runCVDisk(argc, argv);	
			}
		}
		else if (strcmp(argv[2], "-cam") == 0)
		{
			if (argc != 5)
			{
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				runCVCamera(argc, argv);
			}
		}
	}
	getchar();
	return 0;
}

void runCLDisk(int argc, char * argv[])
{
	char *src = argv[2];
	char *input1 = argv[3];
	char *input2 = argv[4];
	char *output = argv[5];
	int alpha = atoi(argv[6]);
	int it = atoi(argv[7]);
	int gs = atoi(argv[8]);
	char* dType = argv[9];

	std::cout<<"OpenCL dysk!\n";

	HSOpticalFlowOpenCL clOpticalFlow("OpticalFlow", src, input1, input2, output, alpha, it, gs, dType);
	clOpticalFlow.initialize();
	clOpticalFlow.setup();
	clOpticalFlow.run();
	clOpticalFlow.cleanup();
}

void runCLCamera(int argc, char * argv[])
{
	char *src = argv[2];
	int alpha = atoi(argv[3]);
	int it = atoi(argv[4]);
	int gs = atoi(argv[5]);
	char* dType = argv[6];

	std::cout<<"OpenCL kamera!\n";
	HSOpticalFlowOpenCL clOpticalFlow("OpticalFlow", src, alpha, it, gs, dType);
	clOpticalFlow.initialize();
	clOpticalFlow.setup();
	clOpticalFlow.run();
	clOpticalFlow.cleanup();
}

void runCVDisk(int argc, char * argv[])
{
	char *input1 = argv[3];
	char *input2 = argv[4];
	char *output = argv[5];
	float lambda = atof(argv[6]);
	int it = atoi(argv[7]);

	std::cout<<"OpenCV dysk!\n";
	OpticalFlowOpenCV *e = new OpticalFlowOpenCV();
	e->runFromImg(input1, input2, output, lambda, it);
}

void runCVCamera(int argc, char * argv[])
{
	float lambda = atof(argv[3]);
	int it = atoi(argv[4]);

	std::cout<<"OpenCV kamera!\n";
	OpticalFlowOpenCV *e = new OpticalFlowOpenCV();
	e->runFromCamera(lambda, it);
}

