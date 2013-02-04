#ifndef FILTERS_H_
#define FILTERS_H_

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <cmath>
#include <SDKCommon.hpp>
#include <SDKApplication.hpp>
#include <SDKFile.hpp>
#include <SDKBitMap.hpp>
#include <cv.h>
#include <highgui.h>
#include <windows.h>

#define GROUP_SIZE 32

/**
* Edge Detection 
* Class implements OpenCL Edge detection
* Derived from SDKSample base class
*/

class EdgeDetectionOpenCL : public SDKSample
{

	cl_uchar4* pixelData;				/**< Pointer to image data */
	cl_uchar4* inputImageData;          /**< Input bitmap data to device */
	cl_uchar4* outputImageData;         /**< Output from device */
	cl_uchar4* tempImageData;
	cl_uchar4* tempImageDataS;
	cl_uchar4* tempImageDataN;
	cl_int* gradDirection;
	cl_int* treshd;

	cl_context context;                 /**< CL context */
	cl_device_id *devices;              /**< CL device list */
	cl_device_type dType;

	cl_mem inputImageBuffer;            /**< CL memory buffer for input Image*/
	cl_mem tempImageBuffer;				/**< CL memory buffer for gauss filter*/
	cl_mem tempImageBufferS;			/**< CL memory buffer for sobel filter*/
	cl_mem tempImageBufferN;			/**< CL memory buffer for nms*/
	cl_mem gradDirectionBuffer;			/**< CL memory buffer for angle*/
	cl_mem outputImageBuffer;           /**< CL memory buffer for Output Image*/
	cl_mem treshb;

	cl_command_queue commandQueue;      /**< CL command queue */
	cl_program program;                 /**< CL program  */
	cl_kernel gauss_filter;             /**< CL kernel */
	cl_kernel sobel_filter;				/**< CL kernel */
	cl_kernel nms_dt;					/**< CL kernel */
	cl_kernel blob;						/**< CL kernel */

	cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
	cl_uint width;                      /**< Width of image */
	cl_uint height;                     /**< Height of image */
	cl_bool byteRWSupport;
	size_t kernelWorkGroupSize;         /**< Group Size returned by kernel */
	size_t blockSizeX;                  /**< Work-group size in x-direction */
	size_t blockSizeY;                  /**< Work-group size in y-direction */
	char* src;
	char* input;
	char* output;
	int iterations;                     /**< Number of iterations for kernel execution */
	
	double totalTimeM;
	double totalTimeG;
	double totalTimeS;
	double totalTimeN;
	double totalTimeB;

	double totalTime;
	double totalTimeGK;
	double totalTimeSK;
	double totalTimeNK;
	double totalTimeBK;

	IplImage *image;
	IplImage *gray;
	IplImage *edge;
	CvCapture* capture;

public:

	/**
	* Read bitmap image and allocate host memory
	* @return 1 on success and 0 on failure
	*/
	int readInputImage();


	/**
	* Read input frame and allocate host memory
	* @return 1 on success and 0 on failure
	*/
	int readInputFrame();

	/** 
	* Constructor 
	* Initialize member variables
	* @param name name of sample (const char*) 
	* @param src source (hard disc or camera) (const char*)
	* @param input name of input file (char*)
	* @param output name of output file (char*)
	* @param lt low treshold (int)
	* @param ht hight treshold (int)
	* @param it number of iterations (int)
	* @param gs group size (int)
	* @param dType type od device (char*)
	*/
	EdgeDetectionOpenCL(const char* name, char* src, char* input, char* output, int lt, int ht, int it, int gs, char* dType)
		: SDKSample(name),
		inputImageData(NULL),
		outputImageData(NULL),
		byteRWSupport(true)
	{
		pixelSize = sizeof(cl_uchar4);
		pixelData = NULL;
		blockSizeX = gs;
		blockSizeY = 1;
		this->src = src;
		this->input = input;
		this->output = output;

		treshd = (cl_int*)malloc(2 * sizeof(cl_int));

		this->treshd[0] = lt;
		this->treshd[1] = ht;
		this->iterations = it;
		totalTime = 0.0;				
		totalTimeGK = 0.0;
		totalTimeSK = 0.0;
		totalTimeNK = 0.0;
		totalTimeBK = 0.0;

		totalTimeM = 0.0;
		totalTimeG = 0.0;
		totalTimeS = 0.0;
		totalTimeN = 0.0;
		totalTimeB = 0.0;


		if (strcmp(dType, "CPU") == 0)
			this->dType = CL_DEVICE_TYPE_CPU;
		else
			this->dType = CL_DEVICE_TYPE_GPU;
	}

	/** 
	* Constructor 
	* Initialize member variables
	* @param name name of sample (const char*) 
	* @param src source (hard disc or camera) (const char*)
	* @param lt low treshold (int)
	* @param ht hight treshold (int)
	* @param gs group size (int)
	* @param dType type od device (char*)
	*/
	EdgeDetectionOpenCL(const char* name, char* src, int lt, int ht, int gs, char* dType)
		: SDKSample(name),
		inputImageData(NULL),
		outputImageData(NULL),
		byteRWSupport(true)
	{
		pixelSize = sizeof(cl_uchar4);
		pixelData = NULL;
		blockSizeX = gs;
		blockSizeY = 1;
		this->src = src;

		treshd = (cl_int*)malloc(2 * sizeof(cl_int));
		this->treshd[0] = lt;
		this->treshd[1] = ht;
		totalTime = 0.0;
		totalTimeGK = 0.0;
		totalTimeSK = 0.0;
		totalTimeNK = 0.0;
		totalTimeBK = 0.0;

		totalTimeM = 0.0;
		totalTimeG = 0.0;
		totalTimeS = 0.0;
		totalTimeN = 0.0;
		totalTimeB = 0.0;


		if (strcmp(dType, "CPU") == 0)
			this->dType = CL_DEVICE_TYPE_CPU;
		else
			this->dType = CL_DEVICE_TYPE_GPU;
	}

	~EdgeDetectionOpenCL()
	{
	}

	/**
	* Allocate image memory and Load bitmap file
	* @return 1 on success and 0 on failure
	*/
	//int setupSobelFilter();

	/**
	* OpenCL related initialisations. 
	* Set up Context, Device list, Command Queue, Memory buffers
	* Build CL kernel program executable
	* @return 1 on success and 0 on failure
	*/
	int setupCL();

	/**
	* Set values for kernels' arguments, enqueue calls to the kernels
	* on to the command queue, wait till end of kernel execution.
	* Get kernel start and end time if timing is enabled
	* @return 1 on success and 0 on failure
	*/
	int runCLKernels();

	/**
	* Reference CPU implementation of Binomial Option
	* for performance comparison
	*/
	//void sobelFilterCPUReference();

	/**
	* Override from SDKSample. Print sample stats.
	*/
	void printStats();

	/**
	* Override from SDKSample. Initialize 
	* command line parser, add custom options
	*/
	int initialize();

	/**
	* Override from SDKSample, adjust width and height 
	* of execution domain, perform all sample setup
	*/
	int setup();

	/**
	* Override from SDKSample
	* Run OpenCL edge detection
	*/
	int run();

	/**
	* Override from SDKSample
	* Cleanup memory allocations
	*/
	int cleanup();

	/**
	* Override from SDKSample
	* Verify against reference implementation
	*/
	int verifyResults();
};

#endif // FILTERS_H_
