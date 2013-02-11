#include "HSOpticalFlowOpenCL.hpp"
#include "EdgeDetectionOpenCV.hpp"

int	main(int argc, float * argv[])
{
	/*if (argc == 1){
		std::cout<<"Nie zostaly podane zadne parametry do funkcji main.\nKoncze dzialanie!\n";
		return 0;
	}
	if (strcmp(argv[1], "-o") == 0){
		if (strcmp(argv[2], "-hd") == 0){
			if (argc != 11){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				float *src = argv[2];
				float *input1 = argv[3];
				float *input2 = argv[4];
				float *output = argv[5];
				int lt = atoi(argv[6]);
				int ht = atoi(argv[7]);
				int it = atoi(argv[8]);
				int gs = atoi(argv[9]);
				float* dType = argv[10];*/

				std::cout<<"OpenCL dysk!\n";
				//HSOpticalFlowOpenCL clFilters("ParallelEdgeDetection", "-hd", "test3.png", "test4.png", "blabla.png", 15 ,10, 1, "CPU");
				HSOpticalFlowOpenCL clFilters("ParallelEdgeDetection", "-cam", "test1.png", "test2.png", "blabla.png", 15, 2,1, "CPU");
				
				std::cout<<"initialize\n";
				clFilters.initialize();
				std::cout<<"setup\n";
				clFilters.setup();
				std::cout<<"run\n";
				clFilters.run();
				std::cout<<"cleanup\n";
				clFilters.cleanup();
		/*	}
		}
		else if (strcmp(argv[2], "-cam") == 0){
			if (argc != 7){
				std::cout<<"Bledna lista argumentow!\n";
				return 0;
			}
			else
			{
				float *src = argv[2];
				int lt = atoi(argv[3]);
				int ht = atoi(argv[4]);
				int gs = atoi(argv[5]);
				float* dType = argv[6];

				std::cout<<"OpenCL kamera!\n";
				HSOpticalFlowOpenCL clFilters("ParallelEdgeDetection", src,lt, ht, gs, dType);
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
				float *src = argv[2];
				float *input = argv[3];
				float *output = argv[4];
				int lt = atoi(argv[5]);
				int ht = atoi(argv[6]);
				int it = atoi(argv[7]);

				std::cout<<"OpenCV dysk!\n";
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
				float *src = argv[2];
				int lt = atoi(argv[3]);
				int ht = atoi(argv[4]);

				std::cout<<"OpenCV kamera!\n";
				EdgeDetectionOpenCV *e = new EdgeDetectionOpenCV(src, lt, ht);
				e->runDetection();
			}
		}
	}*/
	getchar();
	return 0;
}