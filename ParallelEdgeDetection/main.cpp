#include "EdgeDetectionOpenCL.hpp"
#include "EdgeDetectionOpenCV.hpp"
#include "OpticalFlowOpenCV.hpp"

int	main(int argc, char * argv[])
{
	if (argc == 1){
		std::cout<<"Nie zostaly podane zadne parametry do funkcji main.\nKoncze dzialanie!\n";
		return 0;
	}
	if (strcmp(argv[1], "-o") == 0){
		if (strcmp(argv[2], "-hd") == 0){
			if (argc != 10){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				char *src = argv[2];
				char *input = argv[3];
				char *output = argv[4];
				int lt = atoi(argv[5]);
				int ht = atoi(argv[6]);
				int it = atoi(argv[7]);
				int gs = atoi(argv[8]);
				char* dType = argv[9];

				std::cout<<"edge OpenCL dysk!\n";
				EdgeDetectionOpenCL clFilters("ParallelEdgeDetection", src, input, output, lt, ht, it, gs, dType);
				clFilters.initialize();
				clFilters.setup();
				clFilters.run();
				clFilters.cleanup();
			}
		}
		else if (strcmp(argv[2], "-cam") == 0){
			if (argc != 7){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				char *src = argv[2];
				int lt = atoi(argv[3]);
				int ht = atoi(argv[4]);
				int gs = atoi(argv[5]);
				char* dType = argv[6];

				std::cout<<"lt: " << lt << "\n";
				std::cout<<"ht: " << ht << "\n";
				std::cout<<"edge OpenCL kamera!\n";
				EdgeDetectionOpenCL clFilters("ParallelEdgeDetection", src,lt, ht, gs, dType);
				clFilters.initialize();
				clFilters.setup();
				clFilters.run();
				clFilters.cleanup();
			}
		}
	}
	else if (strcmp(argv[1], "-ref") == 0){
		if (strcmp(argv[2], "-hd") == 0){
			if (argc != 8){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				char *src = argv[2];
				char *input = argv[3];
				char *output = argv[4];
				int lt = atoi(argv[5]);
				int ht = atoi(argv[6]);
				int it = atoi(argv[7]);

				std::cout<<"edge OpenCV dysk!\n";
				EdgeDetectionOpenCV *e = new EdgeDetectionOpenCV(src, input, output, lt, ht, it);
				e->runDetection();
			}
		}
		else if (strcmp(argv[2], "-cam") == 0){
			if (argc != 5){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				char *src = argv[2];
				int lt = atoi(argv[3]);
				int ht = atoi(argv[4]);

				std::cout<<"edge OpenCV kamera!\n";
				EdgeDetectionOpenCV *e = new EdgeDetectionOpenCV(src, lt, ht);
				e->runDetection();
			}
		}
		else
		{
			std::cout<<"OpenCV kamera!\n";
			OpticalFlowOpenCV *e = new OpticalFlowOpenCV();
			e->run();
		}
	}
	getchar();
	return 0;
}