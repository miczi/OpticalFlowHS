#include "EdgeDetectionOpenCL.hpp"

int EdgeDetectionOpenCL::readInputImage()
{
	// get the image data
	this->height = image->height;
	this->width = image->width;

	CvScalar s;
	pixelData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

	for(unsigned int i = 0; i < height; i++){
		for(unsigned int j = 0; j < width; j++){
			s = cvGet2D(gray,i,j); 
			pixelData[j + i*width].s[0] = (cl_uchar)s.val[0];
			pixelData[j + i*width].s[1] = (cl_uchar)s.val[1];
			pixelData[j + i*width].s[2] = (cl_uchar)s.val[2];
		}
	}
	/* allocate memory for input & output image data  */
	inputImageData  = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

	/* error check */
	if(inputImageData == NULL)
	{
		sampleCommon->error("Failed to allocate memory! (inputImageData)");
		return SDK_FAILURE;
	}

	/* allocate memory for output image data */
	tempImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
	tempImageDataS = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
	tempImageDataN = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
	outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

	gradDirection = (cl_int*)malloc(width * height * sizeof(cl_int));
	gradDirection = (cl_int*)malloc(width * height * sizeof(cl_int));

	/* error check */
	if(outputImageData == NULL)
	{
		sampleCommon->error("Failed to allocate memory! (outputImageData)");
		return SDK_FAILURE;
	}

	memset(outputImageData, 0, width * height * pixelSize);
	memset(tempImageData, 0, width * height * pixelSize);
	memset(tempImageDataS, 0, width * height * pixelSize);
	memset(tempImageDataN, 0, width * height * pixelSize);
	memset(gradDirection, 0, width * height * sizeof(cl_int));

	/* error check */
	if(pixelData == NULL)
	{
		sampleCommon->error("Failed to read pixel Data!");
		return SDK_FAILURE;
	}

	/* Copy pixel data into inputImageData */
	memcpy(inputImageData, pixelData, width * height * pixelSize);

	return SDK_SUCCESS;
}

int EdgeDetectionOpenCL::readInputFrame()
{
	CvScalar s;
	memset(pixelData, 0, width * height * pixelSize);
	memset(inputImageData, 0, width * height * pixelSize);

	for(unsigned int i = 0; i < height; i++){
		for(unsigned int j = 0; j < width; j++){
			s = cvGet2D(gray,i,j); 
			pixelData[j + i*width].s[0] = (cl_uchar)s.val[0];
			pixelData[j + i*width].s[1] = (cl_uchar)s.val[1];
			pixelData[j + i*width].s[2] = (cl_uchar)s.val[2];
		}
	}

	/* Copy pixel data into inputImageData */
	memcpy(inputImageData, pixelData, width * height * pixelSize);
	/* Initialize data to 0 */
	memset(tempImageData, 0, width * height * pixelSize);
	memset(tempImageDataS, 0, width * height * pixelSize);
	memset(tempImageDataN, 0, width * height * pixelSize);
	memset(outputImageData, 0, width * height * pixelSize);
	memset(gradDirection, 0, width * height * sizeof(cl_int));
	return 0;
}


int EdgeDetectionOpenCL::setupCL()
{
	cl_int status = CL_SUCCESS;
	
	size_t deviceListSize;

	cl_uint numPlatforms;
	cl_platform_id platform = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	if (0 < numPlatforms) 
	{
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		for (unsigned i = 0; i < numPlatforms; ++i) 
		{
			char pbuf[100];
			status = clGetPlatformInfo(platforms[i],
				CL_PLATFORM_VENDOR,
				sizeof(pbuf),
				pbuf,
				NULL);
			platform = platforms[i];
			if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) 
			{
				break;
			}
		}
		delete[] platforms;
	}

	if(NULL == platform)
	{
		sampleCommon->error("OpenCL platform not found!");
		return SDK_FAILURE;
	}

	cl_context_properties cps[3] = 
	{
		CL_CONTEXT_PLATFORM, 
		(cl_context_properties)platform, 
		0
	};

	context = clCreateContextFromType(
		cps,
		dType,
		NULL,
		NULL,
		&status);

	/* First, get the size of device list data */
	status = clGetContextInfo(
		context, 
		CL_CONTEXT_DEVICES, 
		0, 
		NULL, 
		&deviceListSize);

	/* Now allocate memory for device list based on the size we got earlier */
	devices = (cl_device_id*)malloc(deviceListSize);

	/* Now, get the device list data */
	status = clGetContextInfo(
		context, 
		CL_CONTEXT_DEVICES, 
		deviceListSize, 
		devices, 
		NULL);

	/* Create command queue */

	cl_command_queue_properties prop = 0;

	if(timing)
		prop |= CL_QUEUE_PROFILING_ENABLE;


	commandQueue = clCreateCommandQueue(
		context,
		devices[0],
		CL_QUEUE_PROFILING_ENABLE,
		&status);

	/*
	* Create and initialize memory objects
	*/

	/* Create memory object for input Image */
	inputImageBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		width * height * pixelSize,
		0,
		NULL);

	/* Create memory objects for Image */
	outputImageBuffer = clCreateBuffer(context,
		CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		width * height * pixelSize,
		outputImageData,
		NULL);

	tempImageBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * pixelSize,
		tempImageData,
		NULL);

	tempImageBufferS = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * pixelSize,
		tempImageDataS,
		NULL);

	tempImageBufferN = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * pixelSize,
		tempImageDataN,
		NULL);

	gradDirectionBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_int),
		gradDirection,
		NULL);

	treshb = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		2 * sizeof(cl_int),
		treshd,
		NULL);


	/* create a CL program using the kernel source */
	streamsdk::SDKFile kernelFile;
	std::string kernelPath = sampleCommon->getPath();
	kernelPath.append("Filters_Kernels.cl");
	if(!kernelFile.open(kernelPath.c_str()))
	{
		std::cout << "Loading kernel file error: " << kernelPath << std::endl;
		return SDK_FAILURE;
	}
	const char *source = kernelFile.source().c_str();
	size_t sourceSize[] = {strlen(source)};

	program = clCreateProgramWithSource(context,
		1,
		&source,
		sourceSize,
		&status);

	/* create a cl program executable for all the devices specified */
	clBuildProgram(
		program,
		1,
		devices,
		NULL,
		NULL,
		NULL);

	/* get a kernel object handle for a kernel with the given name */
	gauss_filter = clCreateKernel(
		program,
		"gauss_filter",
		&status);

	sobel_filter = clCreateKernel(
		program,
		"sobel_filter",
		&status);

	nms_dt = clCreateKernel(
		program,
		"nms_dt",
		&status);

	blob = clCreateKernel(
		program,
		"blob",
		&status);

	/* Check group size against group size returned by kernel */
	status = clGetKernelWorkGroupInfo(gauss_filter,
		devices[0],
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(size_t),
		&kernelWorkGroupSize,
		0);

	if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
	{
		if(!quiet)
		{
			std::cout << "Out of Resources!" << std::endl;
			std::cout << "Group Size specified : "
				<< blockSizeX * blockSizeY << std::endl;
			std::cout << "Max Group Size supported on the kernel : " 
				<< kernelWorkGroupSize << std::endl;
			std::cout << "Falling back to " << kernelWorkGroupSize << std::endl;
		}

		/* Three possible cases */
		if(blockSizeX > kernelWorkGroupSize)
		{
			blockSizeX = kernelWorkGroupSize;
			blockSizeY = 1;
		}
	}

	return SDK_SUCCESS;
}

int EdgeDetectionOpenCL::runCLKernels()
{
	cl_int status;
	cl_event events[2];

	status = clEnqueueWriteBuffer(commandQueue,
		inputImageBuffer,
		1,
		0,
		width * height * pixelSize,
		inputImageData,
		0,
		0,
		0);

	status = clEnqueueWriteBuffer(commandQueue,
		treshb,
		1,
		0,
		sizeof(cl_int) * 2,
		treshd,
		0,
		0,
		0);


	DWORD t1, t2 = 0;
	t1 = GetTickCount();
	/** GAUSS **/

	/* input buffer image */
	clSetKernelArg(
		gauss_filter,
		0,
		sizeof(cl_mem),
		&inputImageBuffer);

	/* outBuffer imager */
	clSetKernelArg(
		gauss_filter,
		1,
		sizeof(cl_mem),
		&tempImageBuffer);

	/* 
	* Enqueue a kernel run call.
	*/
	size_t globalThreads[] = {width, height};
	size_t localThreads[] = {blockSizeX, blockSizeY};

	status = clEnqueueNDRangeKernel(
		commandQueue,
		gauss_filter,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&events[0]);

	clFlush(commandQueue);
	clWaitForEvents(1, &events[0]);

	/** pomiar **/
	long long start, end;
	double total;


	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START,
		sizeof start, &start, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_START) failed: %s\n";
		start = 0;
	}
	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END,
		sizeof end, &end, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_END) failed: %s\n";
		end = 0;
	}
	total = (double)(end - start) / 1e6; /* Convert nanoseconds to msecs */
	//printf("GAUSS: Total kernel time was %5.3f msecs.\n", total);
	totalTimeGK = totalTimeGK + total;
	totalTime = totalTime + total;

	/*clEnqueueReadBuffer(
		commandQueue,
		tempImageBuffer,
		CL_TRUE,
		0,
		width * height * pixelSize,
		tempImageData,
		0,
		NULL,
		&events[0]);*/


	t2 = t2 + GetTickCount() - t1;

	totalTimeG = totalTimeG + double(t2);


	//std::cout<<"GAUSS time: "<<totalTimeG<<" [ms]"<<std::endl;


	/** GAUSS END **/
	t1 = 0;
	t2 = 0;
	t1 = GetTickCount();
	/** SOBEL & DIR **/

	/* input buffer image */
	clSetKernelArg(
		sobel_filter,
		0,
		sizeof(cl_mem),
		&tempImageBuffer);

	/* outBuffer image */
	clSetKernelArg(
		sobel_filter,
		1,
		sizeof(cl_mem),
		&tempImageBufferS);

	/* direction buffer*/
	clSetKernelArg(
		sobel_filter,
		2,
		sizeof(cl_mem),
		&gradDirectionBuffer);

	/* 
	* Enqueue a kernel run call.
	*/
	globalThreads[0] = width;
	globalThreads[1] = height;
	localThreads[0] = blockSizeX;
	localThreads[1] = blockSizeY;

	clEnqueueNDRangeKernel(
		commandQueue,
		sobel_filter,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&events[0]);

	clFlush(commandQueue);
	clWaitForEvents(1, &events[0]);

	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START,
		sizeof start, &start, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_START) failed: %s\n";
		start = 0;
	}
	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END,
		sizeof end, &end, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_END) failed: %s\n";
		end = 0;
	}

	total = (double)(end - start) / 1e6; /* Convert nanoseconds to msecs */
	//printf("SOBEL: Total kernel time was %5.3f msecs.\n", total);
	totalTimeSK = totalTimeSK + total;
	totalTime = totalTime + total;

	/*clEnqueueReadBuffer(
		commandQueue,
		tempImageBufferS,
		CL_TRUE,
		0,
		width * height * pixelSize,
		tempImageDataS,
		0,
		NULL,
		&events[0]);*/

	/*clEnqueueReadBuffer(
		commandQueue,
		gradDirectionBuffer,
		CL_TRUE,
		0,
		width * height * sizeof(cl_int),
		gradDirection,
		0,
		NULL,
		&events[0]);*/

	t2 = t2 + GetTickCount() - t1;
	totalTimeS = totalTimeS + double(t2);

	/** SOBEL & DIR END **/
	t1 = 0;
	t2 = 0;
	t1 = GetTickCount();
	/* NMS */

	clSetKernelArg(
		nms_dt,
		0,
		sizeof(cl_mem),
		&gradDirectionBuffer);

	clSetKernelArg(
		nms_dt,
		1,
		sizeof(cl_mem),
		&tempImageBufferS);

	clSetKernelArg(
		nms_dt,
		2,
		sizeof(cl_mem),
		&tempImageBufferN);

	clSetKernelArg(
		nms_dt,
		3,
		sizeof(cl_mem),
		&treshb);


	/* 
	* Enqueue a kernel run call.
	*/
	globalThreads[0] = width;
	globalThreads[1] = height;
	localThreads[0] = blockSizeX;
	localThreads[1] = blockSizeY;

	clEnqueueNDRangeKernel(
		commandQueue,
		nms_dt,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&events[0]);

	clFlush(commandQueue);
	clWaitForEvents(1, &events[0]);

	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START,
		sizeof start, &start, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_START) failed: %s\n";
		start = 0;
	}
	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END,
		sizeof end, &end, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_END) failed: %s\n";
		end = 0;
	}

	total = (double)(end - start) / 1e6; /* Convert nanoseconds to msecs */
	//printf("NMS: Total kernel time was %5.3f msecs.\n", total);
	totalTimeNK = totalTimeNK + total;
	totalTime = totalTime + total;

	/*clEnqueueReadBuffer(
		commandQueue,
		tempImageBufferN,
		CL_TRUE,
		0,
		width * height * pixelSize,
		tempImageDataN,
		0,
		NULL,
		&events[0]);*/

	t2 = t2 + GetTickCount() - t1;
	totalTimeN = totalTimeN + double(t2);

	/** NMS END **/
	t1 = 0;
	t2 = 0;
	t1 = GetTickCount();
	/** BLOB **/
	clSetKernelArg(
		blob,
		0,
		sizeof(cl_mem),
		&tempImageBufferN);


	clSetKernelArg(
		blob,
		1,
		sizeof(cl_mem),
		&outputImageBuffer);

	/* 
	* Enqueue a kernel run call.
	*/
	globalThreads[0] = width;
	globalThreads[1] = height;
	localThreads[0] = blockSizeX;
	localThreads[1] = blockSizeY;

	clEnqueueNDRangeKernel(
		commandQueue,
		blob,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&events[0]);


	clFlush(commandQueue);
	clWaitForEvents(1, &events[0]);


	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START,
		sizeof start, &start, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_START) failed: %s\n";
		start = 0;
	}
	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END,
		sizeof end, &end, NULL);
	if (status != CL_SUCCESS) {
		std::cout<<"clGetEventProfilingInfo(COMMAND_END) failed: %s\n";
		end = 0;
	}

	total = (double)(end - start) / 1e6; /* Convert nanoseconds to msecs */
	//printf("BLOB: Total kernel time was %5.3f ms.\n", total);
	totalTimeBK = totalTimeBK + total;
	totalTime = totalTime + total;
	//printf("ALL: Total kernels time was %5.3f ms.\n", totalTime);
	clEnqueueReadBuffer(
		commandQueue,
		outputImageBuffer,
		CL_TRUE,
		0,
		width * height * pixelSize,
		outputImageData,
		0,
		NULL,
		&events[0]);

	/** BLOB END **/

	t2 = t2 + GetTickCount() - t1;
	totalTimeB = totalTimeB + double(t2);

	totalTimeM = totalTimeB + totalTimeG + totalTimeN + totalTimeS;

	clReleaseEvent(events[0]);
	return SDK_SUCCESS;
}

int EdgeDetectionOpenCL::initialize()
{
	// Call base class Initialize to get default configuration
	if(!this->SDKSample::initialize())
		return SDK_FAILURE;

	streamsdk::Option* iteration_option = new streamsdk::Option;
	if(!iteration_option)
	{
		sampleCommon->error("Memory Allocation error.\n");
		return SDK_FAILURE;
	}
	iteration_option->_sVersion = "i";
	iteration_option->_lVersion = "iterations";
	iteration_option->_description = "Number of iterations to execute kernel";
	iteration_option->_type = streamsdk::CA_ARG_INT;
	iteration_option->_value = &iterations;

	sampleArgs->AddOption(iteration_option);

	delete iteration_option;

	return SDK_SUCCESS;
}

int EdgeDetectionOpenCL::run()
{
	int count = 0;
	if(!byteRWSupport)
		return SDK_SUCCESS;

	/* create and initialize timers */
	int timer = sampleCommon->createTimer();
	sampleCommon->resetTimer(timer);

	if (strcmp(src, "-hd") == 0)
	{
		image = cvLoadImage(input, 1);
		if (!image){
			std::cout<<"Input image error.\n";
			return -1;
		}

		gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		cvCvtColor(image, gray, CV_BGR2GRAY);
		CvScalar s;
		readInputImage();
		setupCL();
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);

		for (int i = 0; i < iterations; i++)
			runCLKernels();		

		

		std::cout<<"Avg KERNEL time: "<<double(totalTime/iterations)<<" [ms]"<<std::endl;
		
		std::cout<<"Avg KERNEL GAUSS time: "<<totalTimeGK/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg KERNEL SOBEL time: "<<totalTimeSK/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg KERNEL NMS time: "<<totalTimeNK/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg KERNEL BLOB time: "<<totalTimeBK/iterations<<" [ms]"<<std::endl;


		std::cout<<std::endl;

		std::cout<<"Avg time: "<<totalTimeM/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg GAUSS time: "<<totalTimeG/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg SOBEL time: "<<totalTimeS/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg NMS time: "<<totalTimeN/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg BLOB time: "<<totalTimeB/iterations<<" [ms]"<<std::endl;


		memcpy(pixelData, outputImageData, width * height * pixelSize);
		for (unsigned int i = 0; i < width * height; i++)
		{
			s.val[0] = pixelData[i].s[0];
			s.val[1] = pixelData[i].s[1];
			s.val[2] = pixelData[i].s[2];
			cvSet1D(edge, i, s);
		}		
		cvSaveImage(output, edge);
		return 0;
	}
	else
	{
		CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
		cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 640 );
		cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 );

		if( !capture ) {
			fprintf( stderr, "ERROR: capture is NULL \n" );
			getchar();
			return -1;
		}
		image = cvQueryFrame( capture );
		// Convert to grayscale
		gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		cvCvtColor(image, gray, CV_BGR2GRAY);
		uchar* frameData = (uchar*)gray->imageData;
		CvScalar s;

		readInputImage();
		setupCL();
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);

		cvNamedWindow( "Edge Detection", CV_WINDOW_AUTOSIZE );
		while( 1 ) {

			image = cvQueryFrame( capture );
			if( !image ) {
				fprintf( stderr, "ERROR: frame is null...\n" );
				getchar();
				break;
			}
			cvCvtColor(image, gray, CV_BGR2GRAY);
			readInputFrame();
			runCLKernels();
			memcpy(pixelData, outputImageData, width * height * pixelSize);

			for (unsigned int i = 0; i < width * height; i++)
			{
				s.val[0] = pixelData[i].s[0];
				s.val[1] = pixelData[i].s[1];
				s.val[2] = pixelData[i].s[2];
				cvSet1D(edge, i, s);
			}		
			cvShowImage( "Edge Detection", edge );
			cvSet(edge, cvScalar(0,0,0));
			if( cvWaitKey(10) == 27 ){
				std::cout<<"Avg KERNEL time: "<<double(totalTime/(count+1))<<" [ms]"<<std::endl;	
				std::cout<<"Avg KERNEL GAUSS time: "<<totalTimeGK/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg KERNEL SOBEL time: "<<totalTimeSK/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg KERNEL NMS time: "<<totalTimeNK/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg KERNEL BLOB time: "<<totalTimeBK/(count+1)<<" [ms]"<<std::endl;

				std::cout<<std::endl;

				std::cout<<"Avg time: "<<totalTimeM/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg GAUSS time: "<<totalTimeG/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg SOBEL time: "<<totalTimeS/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg NMS time: "<<totalTimeN/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg BLOB time: "<<totalTimeB/(count+1)<<" [ms]"<<std::endl;

				break;
			}
			count++;
		}
		cvReleaseCapture( &capture );
		cvDestroyWindow( "Edge Detection" );
	}
	return SDK_SUCCESS;
}

int EdgeDetectionOpenCL::cleanup()
{
	clReleaseKernel(gauss_filter);
	clReleaseKernel(sobel_filter);
	clReleaseKernel(nms_dt);
	clReleaseKernel(blob);
	clReleaseProgram(program);
	clReleaseMemObject(inputImageBuffer);
	clReleaseMemObject(outputImageBuffer);
	clReleaseMemObject(tempImageBuffer);
	clReleaseMemObject(tempImageBufferS);
	clReleaseMemObject(tempImageBufferN);
	clReleaseMemObject(gradDirectionBuffer);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);

	if(inputImageData) 
		free(inputImageData);
	if(outputImageData)
		free(outputImageData);
	if(tempImageData)
		free(tempImageData);
	if(tempImageDataS)
		free(tempImageDataS);
	if(tempImageDataN)
		free(tempImageDataN);
	if(devices)
		free(devices);

	return SDK_SUCCESS;
}

int EdgeDetectionOpenCL::verifyResults(){return SDK_SUCCESS;}
int EdgeDetectionOpenCL::setup(){ return SDK_SUCCESS;}