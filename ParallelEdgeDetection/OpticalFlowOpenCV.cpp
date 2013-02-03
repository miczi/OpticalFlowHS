#include "OpticalFlowOpenCV.hpp"

#define CVX_GRAY50 cvScalar(100)
#define CVX_WHITE  cvScalar(255)
#define FLOW_TITLE "OpticalFlow"

int OpticalFlowOpenCV::run()
{
	IplImage *imgTmp;
	IplImage *imgNew;
	IplImage *imgOld;
	
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
	imgOld = cvCreateImage(cvSize(imgTmp->width,imgTmp->height), IPL_DEPTH_8U, 1);
	cvCvtColor(imgTmp, imgOld, CV_BGR2GRAY);
	
	mi();
	while (1)
	{
		imgTmp = cvQueryFrame(capture);
		imgNew = cvCreateImage(cvSize(imgTmp->width,imgTmp->height), IPL_DEPTH_8U, 1);
		cvCvtColor(imgTmp, imgNew, CV_BGR2GRAY);
		
		IplImage* velx = cvCreateImage(cvGetSize(imgOld),IPL_DEPTH_32F,1); 
		IplImage* vely = cvCreateImage(cvGetSize(imgOld),IPL_DEPTH_32F,1);
		IplImage* imgRes = cvCreateImage(cvGetSize(imgOld),IPL_DEPTH_8U,3);
		
		ms();
		cvCalcOpticalFlowHS(imgOld, imgNew, 0, velx, vely, .10, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, imgOld->width, 1e-6));
		me();
		
		cvZero(imgRes);
		mes();
		int step = 4;
		for(int y=0; y<imgRes->height; y += step) 
		{
			float* px = (float*) (velx->imageData + y * velx->widthStep);
			float* py = (float*) (vely->imageData + y * vely->widthStep);
			for(int x=0; x<imgRes->width; x += step) 
			{
				if( px[x]>1 && py[x]>1 ) 
				{
					cvCircle(imgRes, cvPoint( x, y ), 2, CVX_GRAY50, -1);
					cvLine(imgRes, cvPoint( x, y ), cvPoint( x+px[x]/2, y+py[x]/2 ), CV_RGB(255,0,0), 1, 0);
				}
			}
		}
		cvShowImage(FLOW_TITLE, imgRes);
		cvWaitKey(1);
    
		cvReleaseImage(&velx);
		cvReleaseImage(&vely);
		cvReleaseImage(&imgOld);
		cvReleaseImage(&imgRes);
		imgOld = imgNew;
		
		md();
		mr();
		if( cvWaitKey(10) == 27 )
		{
			break;
		}
	}
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

	std::cout << double(global / count) << " ms" << std::endl;
}
