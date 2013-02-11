#include "EdgeDetectionOpenCL.hpp"
#include "OpticalFlowOpenCV.hpp"

int	main(int argc, char * argv[])
{
	/*if (argc == 1){
		std::cout<<"Nie zostaly podane zadne parametry do funkcji main.\nKoncze dzialanie!\n";
		return 0;
	}*/
	/*if (strcmp(argv[1], "-cl") == 0)
	{
		if (argc != 6){
			std::cout<<"Bledna lista argumentow!\n";
			return 0;
		}
		else
		{
			char *src = argv[1];
			int lt = atoi(argv[2]);
			int ht = atoi(argv[3]);
			int gs = atoi(argv[4]);
			char* dType = argv[5];

			std::cout<<"lt: " << lt << "\n";
			std::cout<<"ht: " << ht << "\n";
			std::cout<<"edge OpenCL kamera!\n";
			EdgeDetectionOpenCL clFilters("ParallelEdgeDetection", src, lt, ht, gs, dType);
			clFilters.initialize();
			clFilters.setup();
			clFilters.run();
			clFilters.cleanup();
		}
	}
	else if (strcmp(argv[1], "-cv") == 0)
	{*/
		/*std::cout<<"OpenCV kamera!\n";
		OpticalFlowOpenCV *e = new OpticalFlowOpenCV();
		e->run();*/
	//}
	std::cout<<"OpenCV kamera!\n";
	OpticalFlowOpenCV *e = new OpticalFlowOpenCV();
	e->run();
	getchar();
	return 0;
}