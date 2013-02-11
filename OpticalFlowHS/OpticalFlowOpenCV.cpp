#include "OpticalFlowOpenCV.hpp"

#define CVX_CIRCLE CV_RGB(0,0,255)
#define CVX_LINE CV_RGB(255,0,0)
#define FLOW_TITLE "OpticalFlow"

int OpticalFlowOpenCV::run()
{
	IplImage *imgTmp;
	IplImage *imgNew;
	IplImage *imgOld;
	IplImage *image1;
	IplImage *image2;
	
	CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 640 );
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 );

	if( !capture ) 
	{
		std::cout<<"ERROR: capture is NULL \n";
		getchar();
		return -1;
	}

	cvNamedWindow(FLOW_TITLE, CV_WINDOW_AUTOSIZE);
	imgTmp = cvQueryFrame(capture);
	imgOld = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 1);
	cvCvtColor(imgTmp, imgOld, CV_BGR2GRAY);

	image1 = cvLoadImage("test1.png", 1);
	image2 = cvLoadImage("test2.png", 1);
	
	imgOld = cvCreateImage(cvSize(image1->width,image1->height), IPL_DEPTH_8U, 1);
	imgNew = cvCreateImage(cvSize(image2->width,image2->height), IPL_DEPTH_8U, 1);

	cvCvtColor(image1, imgOld, CV_BGR2GRAY);
	cvCvtColor(image2, imgNew, CV_BGR2GRAY);


	mi();
	/*while (1)
	{*/
		/*imgTmp = cvQueryFrame(capture);
		imgNew = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 1);
		cvCvtColor(imgTmp, imgNew, CV_BGR2GRAY);*/
		
		IplImage* velx = cvCreateImage(cvSize(image1->width,image1->height),IPL_DEPTH_32F,1); 
		IplImage* vely = cvCreateImage(cvSize(image1->width,image1->height),IPL_DEPTH_32F,1);
		IplImage* imgFlow = cvCreateImage(cvSize(image1->width,image1->height), IPL_DEPTH_8U, 3);
		
		ms();
		cvCalcOpticalFlowHS(imgOld, imgNew, 0, velx, vely, .1, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 1e-6));
		me();
		
		cvZero(imgFlow);
		int step = 4;
		for(int y=0; y<imgFlow->height; y += step) 
		{
			float* px = (float*) (velx->imageData + y * velx->widthStep);
			float* py = (float*) (vely->imageData + y * vely->widthStep);
			for(int x=0; x<imgFlow->width; x += step) 
			{
				/*std::cout<< "px[" << x << "]: " << px[x] << "\n";
				std::cout<< "py[" << x << "]: " << py[x] << "\n";*/
				//if( px[x]>1 || py[x]>1 || px[x]<-1 || py[x]<-1) 
				//{
					
					cvCircle(imgFlow, cvPoint( x, y ), 2, CVX_CIRCLE, -1);
					cvLine(imgFlow, cvPoint( x, y ), cvPoint( x+px[x]/2, y+py[x]/2 ), CVX_LINE, 1, 0);
				//}
			}
		}
		cvSaveImage("wynik.png", imgFlow);
		/*std::cout<< "velx->imageData" << velx->imageData;
		std::cout<< "vely->imageData" << vely->imageData;
		std::cout<< "velx->widthStep" << velx->widthStep;
		std::cout<< "vely->widthStep" << vely->widthStep;*/
		cvReleaseImage(&velx);
		cvReleaseImage(&vely);
		cvReleaseImage(&imgOld);
		cvReleaseImage(&imgFlow);
		imgOld = imgNew;
		
		md();
		mr();
		/*if( cvWaitKey(10) == 27 )
			break;*/
	//}
	// destroy windows
	cvDestroyWindow(FLOW_TITLE);
	return 0;
}  

void OpticalFlowOpenCV::mi()
{
	count = 1;
	timeIdx = 0;
}

void OpticalFlowOpenCV::mr()
{
	timeIdx = 0;
	count++;
}

void OpticalFlowOpenCV::ms()
{
	time = GetTickCount();
}

void OpticalFlowOpenCV::me()
{
	if (timeIdx < sizeof(times)/sizeof(times[0]))
	{
		times[timeIdx] += GetTickCount() - time;
		timeIdx++;
	}
}

void OpticalFlowOpenCV::mes()
{
	me();
	ms();
}

void OpticalFlowOpenCV::md()
{
	int global = 0;
	for (int i = 0; i < sizeof(times)/sizeof(times[0]); i++)
	{
		global += times[i];
		std::cout<<double(times[i] / count)<<" ms | ";
	}

	std::cout << "total: " << double(global / count) << " ms" << std::endl;
}
