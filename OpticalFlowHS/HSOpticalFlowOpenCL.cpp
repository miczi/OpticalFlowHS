#include "HSOpticalFlowOpenCL.hpp"

#define CVX_CIRCLE CV_RGB(0,0,255)
#define CVX_LINE CV_RGB(255,0,0)

int HSOpticalFlowOpenCL::readInputImage(cl_float4** inputImageData)
{
	// get the image data
	this->height = image->height;
	this->width = image->width;

	CvScalar s;
	pixelData = (cl_float4*)malloc(width * height * sizeof(cl_float4));

	for(unsigned int i = 0; i < height; i++){
		for(unsigned int j = 0; j < width; j++){
			s = cvGet2D(gray,i,j); 
			pixelData[j + i*width].s[0] = (cl_float)s.val[0];
			pixelData[j + i*width].s[1] = (cl_float)s.val[1];
			pixelData[j + i*width].s[2] = (cl_float)s.val[2];
		}
	}

	//allocate memory for input image data 
	*inputImageData  = (cl_float4*)malloc(width * height * sizeof(cl_float4));

	/* error check */
	if(inputImageData == NULL)
	{
		sampleCommon->error("Failed to allocate memory! (inputImageData)");
		return SDK_FAILURE;
	}

	/* error check */
	if(pixelData == NULL)
	{
		sampleCommon->error("Failed to read pixel Data!");
		return SDK_FAILURE;
	}

	/* Copy pixel data into inputImageData */
	memcpy(*inputImageData, pixelData, width * height *  sizeof(cl_float4));

	return SDK_SUCCESS;
}

int HSOpticalFlowOpenCL::readInputFrame(cl_float4** inputImageData)
{
	CvScalar s;
	memset(pixelData, 0, width * height *  sizeof(cl_float4));
	memset(*inputImageData, 0, width * height *  sizeof(cl_float4));
	for(unsigned int i = 0; i < height; i++){
		for(unsigned int j = 0; j < width; j++){
			s = cvGet2D(gray,i,j); 
			pixelData[j + i*width].s[0] = (cl_float)s.val[0];
			pixelData[j + i*width].s[1] = (cl_float)s.val[1];
			pixelData[j + i*width].s[2] = (cl_float)s.val[2];
		}
	}

	/* Copy pixel data into inputImageData */
	memcpy(*inputImageData, pixelData, width * height *  sizeof(cl_float4));
	return 0;
}


int HSOpticalFlowOpenCL::setupCL()
{

	Ex = (cl_float4*)malloc(width * height * sizeof(cl_float4));
	Ey = (cl_float4*)malloc(width * height * sizeof(cl_float4));
	Et = (cl_float4*)malloc(width * height * sizeof(cl_float4));


	u_avg = (cl_float4*)malloc(width * height * sizeof(cl_float4));
	v_avg = (cl_float4*)malloc(width * height * sizeof(cl_float4));

	u = (cl_float4*)malloc(width * height * sizeof(cl_float4));
	v = (cl_float4*)malloc(width * height * sizeof(cl_float4));

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

	/* Create memory object for input Images */

	inputImageBuffer1 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		width * height * pixelSize,
		0,
		NULL);

	inputImageBuffer2 = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		width * height * pixelSize,
		0,
		NULL);

	/* Create memory object for derivatives */

	ExBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_float4),
		Ex,
		NULL);

	EyBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_float4),
		Ey,
		NULL);

	EtBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_float4),
		Et,
		NULL);

	/* Create memory object for velocities average */

	u_avgBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_float4),
		u_avg,
		NULL);

	v_avgBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_float4),
		v_avg,
		NULL);

	/* Create memory object for velocities */

	uBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_float4),
		u,
		NULL);

	vBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		width * height * sizeof(cl_float4),
		v,
		NULL);


	/* create a CL program using the kernel source */
	streamsdk::SDKFile kernelFile;
	std::string kernelPath = sampleCommon->getPath();
	kernelPath.append("Kernels.cl");
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

	cl_int errNum;

	/* create a cl program executable for all the devices specified */
	errNum = clBuildProgram(
		program,
		1,
		devices,
		NULL,
		NULL,
		NULL);

	if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        float buildLog[16384];
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

	/* get a kernel object handle for a kernel with the given name */
	ComputeDerivativesKernel =  clCreateKernel(
		program,
		"ComputeDerivativesKernel",
		&status);

	u_v_avgKernel =  clCreateKernel(
		program,
		"u_v_avgKernel",
		&status);

	u_v_updateKernel =  clCreateKernel(
		program,
		"u_v_updateKernel",
		&status);

	/* Check group size against group size returned by kernel */

	status = clGetKernelWorkGroupInfo(ComputeDerivativesKernel,
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

int HSOpticalFlowOpenCL::runDerivatives()
{

	memset(Ex, 0, width * height * sizeof(cl_float4));
	memset(Ey, 0, width * height * sizeof(cl_float4));
	memset(Et, 0, width * height * sizeof(cl_float4));

	memset(u_avg, 0, width * height * sizeof(cl_float4));
	memset(v_avg, 0, width * height * sizeof(cl_float4));

	memset(u, 0, width * height * sizeof(cl_float4));
	memset(v, 0, width * height * sizeof(cl_float4));

	cl_int status;
	cl_event events[2];



	status = clEnqueueWriteBuffer(commandQueue,
		inputImageBuffer1,
		1,
		0,
		width * height * sizeof(cl_float4),
		inputImageData1,
		0,
		0,
		0);

	status = clEnqueueWriteBuffer(commandQueue,
		inputImageBuffer2,
		1,
		0,
		width * height * sizeof(cl_float4),
		inputImageData2,
		0,
		0,
		0);


	size_t globalThreads[] = {width, height};
	size_t localThreads[] = {blockSizeX, blockSizeY};

	DWORD t1, t2 = 0;
	t1 = GetTickCount();

clSetKernelArg(
		ComputeDerivativesKernel,
		0,
		sizeof(cl_mem),
		&inputImageBuffer1);

	clSetKernelArg(
		ComputeDerivativesKernel,
		1,
		sizeof(cl_mem),
		&inputImageBuffer2);

	clSetKernelArg(
		ComputeDerivativesKernel,
		2,
		sizeof(cl_mem),
		&ExBuffer);

	clSetKernelArg(
		ComputeDerivativesKernel,
		3,
		sizeof(cl_mem),
		&EyBuffer);

	clSetKernelArg(
		ComputeDerivativesKernel,
		4,
		sizeof(cl_mem),
		&EtBuffer);



	/* 
	* Enqueue a kernel run call.
	*/

	globalThreads[0] = width;
	globalThreads[1] = height;
	localThreads[0] = blockSizeX;
	localThreads[1] = blockSizeY;

	clEnqueueNDRangeKernel(
		commandQueue,
		ComputeDerivativesKernel,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&events[0]);

	clFlush(commandQueue);
	clWaitForEvents(1, &events[0]);

	
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
	//printf("DER: Total kernel time was %5.3f msecs.\n", total);
	totalTimeDK = totalTimeDK + total;
	totalTime = totalTime + total;

	clEnqueueReadBuffer(
		commandQueue,
		ExBuffer,
		CL_TRUE,
		0,
		width * height *sizeof(cl_float4),
		Ex,
		0,
		NULL,
		&events[0]);

	clEnqueueReadBuffer(
		commandQueue,
		EyBuffer,
		CL_TRUE,
		0,
		width * height *sizeof(cl_float4),
		Ey,
		0,
		NULL,
		&events[0]);

	clEnqueueReadBuffer(
		commandQueue,
		EtBuffer,
		CL_TRUE,
		0,
		width * height *sizeof(cl_float4),
		Et,
		0,
		NULL,
		&events[0]);




	t2 = t2 + GetTickCount() - t1;
	totalTimeD = totalTimeD + double(t2);

	// END DERIVATES

	clReleaseEvent(events[0]);
	return SDK_SUCCESS;
}

int HSOpticalFlowOpenCL::runCLKernels()
{
	cl_int status;
	cl_event events[2];



	status = clEnqueueWriteBuffer(commandQueue,
		uBuffer,
		1,
		0,
		width * height * sizeof(cl_float4),
		u,
		0,
		0,
		0);

	status = clEnqueueWriteBuffer(commandQueue,
		vBuffer,
		1,
		0,
		width * height * sizeof(cl_float4),
		v,
		0,
		0,
		0);


	size_t globalThreads[] = {width, height};
	size_t localThreads[] = {blockSizeX, blockSizeY};


	DWORD t1, t2 = 0;
	t1 = GetTickCount();

	// u_v_avgKernel start

	clSetKernelArg(
		u_v_avgKernel,
		0,
		sizeof(cl_mem),
		&uBuffer);

	clSetKernelArg(
		u_v_avgKernel,
		1,
		sizeof(cl_mem),
		&vBuffer);

	clSetKernelArg(
		u_v_avgKernel,
		2,
		sizeof(cl_mem),
		&u_avgBuffer);

	clSetKernelArg(
		u_v_avgKernel,
		3,
		sizeof(cl_mem),
		&v_avgBuffer);

	globalThreads[0] = width;
	globalThreads[1] = height;
	localThreads[0] = blockSizeX;
	localThreads[1] = blockSizeY;

	clEnqueueNDRangeKernel(
		commandQueue,
		u_v_avgKernel,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&events[0]);

	clFlush(commandQueue);
	clWaitForEvents(1, &events[0]);

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
	//printf("u_v_avgKernel: Total kernel time was %5.3f msecs.\n", total);
	totalTimeUVAK = totalTimeUVAK + total;
	totalTime = totalTime + total;

	t2 = t2 + GetTickCount() - t1;
	totalTimeUVA = totalTimeUVA + double(t2);

	// u_v_avgKernel end

	t1 = 0;
	t2 = 0;
	t1 = GetTickCount();

	// u_v_updateKernel start

	clSetKernelArg(
		u_v_updateKernel,
		0,
		sizeof(cl_mem),
		&uBuffer);

	clSetKernelArg(
		u_v_updateKernel,
		1,
		sizeof(cl_mem),
		&vBuffer);

	clSetKernelArg(
		u_v_updateKernel,
		2,
		sizeof(cl_mem),
		&u_avgBuffer);

	clSetKernelArg(
		u_v_updateKernel,
		3,
		sizeof(cl_mem),
		&v_avgBuffer);

	clSetKernelArg(
		u_v_updateKernel,
		4,
		sizeof(cl_mem),
		&ExBuffer);

	clSetKernelArg(
		u_v_updateKernel,
		5,
		sizeof(cl_mem),
		&EyBuffer);

	clSetKernelArg(
		u_v_updateKernel,
		6,
		sizeof(cl_mem),
		&EtBuffer);

	clSetKernelArg(
		u_v_updateKernel,
		7,
		sizeof(cl_float),
		&alpha);

	globalThreads[0] = width;
	globalThreads[1] = height;
	localThreads[0] = blockSizeX;
	localThreads[1] = blockSizeY;

	clEnqueueNDRangeKernel(
		commandQueue,
		u_v_updateKernel,
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
	//printf("u_v_avgKernel: Total kernel time was %5.3f msecs.\n", total);
	totalTimeUVK =  totalTimeUVK + total;
	totalTime = totalTime + total;

	clEnqueueReadBuffer(
		commandQueue,
		uBuffer,
		CL_TRUE,
		0,
		width * height *sizeof(cl_float4),
		u,
		0,
		NULL,
		&events[0]);

	clEnqueueReadBuffer(
		commandQueue,
		vBuffer,
		CL_TRUE,
		0,
		width * height *sizeof(cl_float4),
		v,
		0,
		NULL,
		&events[0]);

	t2 = t2 + GetTickCount() - t1;
	totalTimeUV = totalTimeUV + double(t2);
	//std::cout<<"totalTimeUV "<<GetTickCount()<<"\n";

	totalTimeM = totalTime + totalTimeD + totalTimeUV + totalTimeUVA;

	clReleaseEvent(events[0]);
	return SDK_SUCCESS;
}

int HSOpticalFlowOpenCL::initialize()
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

int HSOpticalFlowOpenCL::run()
{
	int count = 0;
	if(!byteRWSupport)
		return SDK_SUCCESS;

	/* create and initialize timers */
	int timer = sampleCommon->createTimer();
	sampleCommon->resetTimer(timer);

	if (strcmp(src, "-hd") == 0)
	{
		image = cvLoadImage(input1, 1);
		if (!image){
			std::cout<<"Input image error.\n";
			return -1;
		}

		gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		cvCvtColor(image, gray, CV_BGR2GRAY);
		CvScalar s;

		readInputImage(&inputImageData1);
		image = cvLoadImage(input2, 1);
		if (!image){
			std::cout<<"Input image error.\n";
			return -1;
		}

		gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		cvCvtColor(image, gray, CV_BGR2GRAY);
		readInputImage(&inputImageData2);

	
		std::cout<<"przed setupCL\n";
		setupCL();
		std::cout<<"po setupCL\n";
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);

		runDerivatives();	
		for (int i = 0; i < iterations; i++)
			runCLKernels();


		std::cout<<"Avg KERNEL time: "<<(totalTimeDK + totalTimeUVAK + totalTimeUVK)/iterations<<" [ms]"<<std::endl;
		std::cout<<"KERNEL ComputeDerivativesKernel time: "<<totalTimeDK<<" [ms]"<<std::endl;
		std::cout<<"Avg KERNEL u_v_avgKernel time: "<<totalTimeUVAK/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg KERNEL u_v_updateKernel time: "<<totalTimeUVK/iterations<<" [ms]"<<std::endl;
		std::cout<<std::endl;
		std::cout<<"Avg time: "<<double(totalTimeDK + totalTimeM)/iterations<<" [ms]"<<std::endl;
		std::cout<<"Derivatives time: "<<totalTimeD<<" [ms]"<<std::endl;
		std::cout<<"Avg u_v_avg time: "<<totalTimeUVA/iterations<<" [ms]"<<std::endl;
		std::cout<<"Avg u_v time: "<<totalTimeUV/iterations<<" [ms]"<<std::endl;


		IplImage* imgFlow = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U,  3);
		cvZero(imgFlow);

		//drawing optical flow
		int step = 4;
		for(unsigned int i = 0; i < height; i += step){
			for(unsigned int j = 0; j < width; j += step){
				if(u[j + i*width].s[0]>0.5 || v[j + i*width].s[0]>0.5 || u[j + i*width].s[0]<-0.5 || v[j + i*width].s[0]<-0.5){
					cvCircle(imgFlow, cvPoint( j, i ), 2, CVX_CIRCLE, -1);
					cvLine(imgFlow, cvPoint( j, i ), cvPoint( j+u[j + i*width].s[0], i+v[j + i*width].s[0] ), CVX_LINE, 1, 0);
				}
			}
		}

		cvSaveImage(output, imgFlow);
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
		float* frameData = (float*)gray->imageData;
		CvScalar s;

		readInputImage(&inputImageData1);
		setupCL();
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);

		inputImageData2  = (cl_float4*)malloc(width * height * sizeof(cl_float4));

		cvNamedWindow( "Optical flow", CV_WINDOW_AUTOSIZE );
		while( 1 ) {

			image = cvQueryFrame( capture );
			if( !image ) {
				fprintf( stderr, "ERROR: frame is null...\n" );
				getchar();
				break;
			}
			cvCvtColor(image, gray, CV_BGR2GRAY);

			readInputFrame(&inputImageData2);

			runDerivatives();	
			for (int i = 0; i < iterations; i++)
			runCLKernels();

			IplImage* imgFlow = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U,  3);
			cvZero(imgFlow);

			//drawing optical flow
			int step = 4;
			for(unsigned int i = 0; i < height; i += step){
				for(unsigned int j = 0; j < width; j += step){
					if(u[j + i*width].s[0]>0.5 || v[j + i*width].s[0]>0.5 || u[j + i*width].s[0]<-0.5 || v[j + i*width].s[0]<-0.5){
						cvCircle(imgFlow, cvPoint( j, i ), 2, CVX_CIRCLE, -1);
						cvLine(imgFlow, cvPoint( j, i ), cvPoint( j+u[j + i*width].s[0], i+v[j + i*width].s[0] ), CVX_LINE, 1, 0);
					}
				}
			}

			cvShowImage( "Optical flow", imgFlow );

			memcpy(inputImageData1, inputImageData2, width * height * pixelSize);

			if( cvWaitKey(10) == 27 ){

				std::cout<<"Avg KERNEL ComputeDerivativesKernel time: "<<totalTimeDK/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg KERNEL u_v_avgKernel time: "<<totalTimeUVAK/iterations/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg KERNEL u_v_updateKernel time: "<<totalTimeUVK/iterations/(count+1)<<" [ms]"<<std::endl;
				std::cout<<std::endl;
				std::cout<<"Avg time: "<<totalTimeM/iterations/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg Derivatives time: "<<totalTimeD/iterations/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg u_v_avg time: "<<totalTimeUVA/iterations/(count+1)<<" [ms]"<<std::endl;
				std::cout<<"Avg u_v time: "<<totalTimeUV/iterations/(count+1)<<" [ms]"<<std::endl;

				break;
			}
			count++;
		}
		cvReleaseCapture( &capture );
		cvDestroyWindow( "Optical flow" );
	}
	return SDK_SUCCESS;
}

int HSOpticalFlowOpenCL::cleanup()
{

	clReleaseKernel(ComputeDerivativesKernel);
	clReleaseKernel(u_v_avgKernel);
	clReleaseKernel(u_v_updateKernel);
	clReleaseProgram(program);
	clReleaseMemObject(inputImageBuffer1);
	clReleaseMemObject(inputImageBuffer2);
	clReleaseMemObject(ExBuffer);
	clReleaseMemObject(EyBuffer);
	clReleaseMemObject(EtBuffer);
	clReleaseMemObject(u_avgBuffer);
	clReleaseMemObject(v_avgBuffer);
	clReleaseMemObject(uBuffer);
	clReleaseMemObject(vBuffer);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);

	if(pixelData)
		free(pixelData);
	if(inputImageData1) 
		free(inputImageData1);
	if(inputImageData2)
		free(inputImageData2);
	if(Ex)
		free(Ex);
	if(Ey)
		free(Ey);
	if(Et)
		free(Et);
	if(u_avg)
		free(u_avg);
	if(v_avg)
		free(v_avg);
	if(u)
		free(u);
	if(v)
		free(v);
	if(devices)
		free(devices);

	return SDK_SUCCESS;
}

int HSOpticalFlowOpenCL::verifyResults(){return SDK_SUCCESS;}
int HSOpticalFlowOpenCL::setup(){ return SDK_SUCCESS;}