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

class HSOpticalFlowOpenCL : public SDKSample
{

	cl_float4* pixelData;				/**< Pointer to image data */
	cl_float4* inputImageData1;          /**< Input bitmap data to device */
	cl_float4* inputImageData2; 
	cl_float4* outputImageData;         /**< Output from device */
	cl_float4* tempImageData;
	cl_float4* tempImageDataS;
	cl_float4* tempImageDataN;
	cl_int* gradDirection;


	cl_float4* Ex;
	cl_float4* Ey;
	cl_float4* Et;

	cl_float4* u_avg;
	cl_float4* v_avg;
	cl_float4* u;
	cl_float4* v;

	cl_float alpha;


	cl_context context;                 /**< CL context */
	cl_device_id *devices;              /**< CL device list */
	cl_device_type dType;

	cl_mem inputImageBuffer1;            /**< CL memory buffer for input Image1 (first frame)*/
	cl_mem inputImageBuffer2;            /**< CL memory buffer for input Image2 (next frame)*/


	cl_mem ExBuffer;	/**< CL memory buffer for x derivatives*/
	cl_mem EyBuffer;	/**< CL memory buffer for y derivatives*/
	cl_mem EtBuffer;	/**< CL memory buffer for t (frame) derivatives*/

	cl_mem u_avgBuffer;
	cl_mem v_avgBuffer;

	cl_mem uBuffer;
	cl_mem vBuffer;



	cl_command_queue commandQueue;      /**< CL command queue */
	cl_program program;                 /**< CL program  */
	cl_kernel ComputeDerivativesKernel;	
	cl_kernel u_v_avgKernel;
	cl_kernel u_v_updateKernel;

	cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
	cl_uint width;                      /**< Width of image */
	cl_uint height;                     /**< Height of image */
	cl_bool byteRWSupport;
	size_t kernelWorkGroupSize;         /**< Group Size returned by kernel */
	size_t blockSizeX;                  /**< Work-group size in x-direction */
	size_t blockSizeY;                  /**< Work-group size in y-direction */
	char* src;
	char* input1;
	char* input2;
	char* output;
	int iterations;                     /**< Number of iterations for kernel execution */
	
	double totalTimeM;
	double totalTimeG;
	double totalTimeS;
	double totalTimeN;
	double totalTimeB;
	double totalTimeD;
	double totalTimeUVA;
	double totalTimeUV;

	double totalTime;
	double totalTimeGK;
	double totalTimeSK;
	double totalTimeNK;
	double totalTimeBK;
	double totalTimeDK;
	double totalTimeUVAK;
	double totalTimeUVK;
	

	IplImage *image;
	IplImage *gray;
	IplImage *edge;
	CvCapture* capture;

public:

	/**
	* Read bitmap image and allocate host memory
	* @return 1 on success and 0 on failure
	*/
	int readInputImage(cl_float4** inputImageDat);


	/**
	* Read input frame and allocate host memory
	* @return 1 on success and 0 on failure
	*/
	int readInputFrame(cl_float4** inputImageDat);

	/** 
	* Constructor 
	* Initialize member variables
	* @param name name of sample (const float*) 
	* @param src source (hard disc or camera) (const float*)
	* @param input name of input file (float*)
	* @param output name of output file (float*)
	* @param lt low treshold (int)
	* @param ht hight treshold (int)
	* @param it number of iterations (int)
	* @param gs group size (int)
	* @param dType type od device (float*)
	*/
	HSOpticalFlowOpenCL(const char* name, char* src, char* input1, char* input2, char* output, float alp, int it, int gs, char* dType)
		: SDKSample(name),
		byteRWSupport(true)
	{
		pixelSize = sizeof(cl_float4);
		pixelData = NULL;
		blockSizeX = gs;
		blockSizeY = 1;
		this->src = src;
		this->input1 = input1;
		this->input2 = input2;
		this->output = output;

		this->alpha = alp;
		this->iterations = it;
		
		totalTime = 0.0;				
		totalTimeGK = 0.0;
		totalTimeSK = 0.0;
		totalTimeNK = 0.0;
		totalTimeBK = 0.0;
		totalTimeDK = 0.0;
		totalTimeUVAK = 0.0;
		totalTimeUVK = 0.0;

		totalTimeM = 0.0;
		totalTimeG = 0.0;
		totalTimeS = 0.0;
		totalTimeN = 0.0;
		totalTimeB = 0.0;
		totalTimeD = 0.0;
		totalTimeUVA = 0.0;
		totalTimeUV = 0.0;


		if (strcmp(dType, "CPU") == 0)
			this->dType = CL_DEVICE_TYPE_CPU;
		else
			this->dType = CL_DEVICE_TYPE_GPU;
	}

	/** 
	* Constructor 
	* Initialize member variables
	* @param name name of sample (const float*) 
	* @param src source (hard disc or camera) (const float*)
	* @param lt low treshold (int)
	* @param ht hight treshold (int)
	* @param gs group size (int)
	* @param dType type od device (float*)
	*/
	HSOpticalFlowOpenCL(const char* name, char* src, int lt, int ht, int gs, char* dType)
		: SDKSample(name),
		byteRWSupport(true)
	{
		pixelSize = sizeof(cl_float4);
		pixelData = NULL;
		blockSizeX = gs;
		blockSizeY = 1;
		this->src = src;

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

	~HSOpticalFlowOpenCL()
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

	
	int runDerivatives();
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
