#include "OpticalFlowOpenCV.hpp"

#define CVX_CIRCLE CV_RGB(0,0,255)
#define CVX_LINE CV_RGB(255,0,0)
#define FLOW_TITLE "OpticalFlow"

int OpticalFlowOpenCV::runFromImg(char* input1, char* input2, char* output, float lambda, int it)
{
	DWORD t1, t2 = 0;

	IplImage *imgTmp;
	IplImage *imgNew;
	IplImage *imgOld;
	
	imgTmp = cvLoadImage(input1, 1);
	imgOld = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 1);
	cvCvtColor(imgTmp, imgOld, CV_BGR2GRAY);
	imgTmp = cvLoadImage(input2, 1);
	imgNew = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 1);
	cvCvtColor(imgTmp, imgNew, CV_BGR2GRAY);
	
	IplImage* velx = cvCreateImage(cvGetSize(imgTmp),IPL_DEPTH_32F,1); 
	IplImage* vely = cvCreateImage(cvGetSize(imgTmp),IPL_DEPTH_32F,1);
	IplImage* imgFlow = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 3);
		
	t1 = GetTickCount();
	cvSmooth( imgOld, imgOld, CV_BLUR, 3, 3, 0, 0 );
	cvSmooth( imgNew, imgNew, CV_BLUR, 3, 3, 0, 0 );
	cvCalcOpticalFlowHS(imgOld, imgNew, 0, velx, vely, lambda, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, it, 1e-6));
	t2 = t2 + GetTickCount() - t1;
		
	cvZero(imgFlow);
	int step = 4;
	for(int y=0; y<imgFlow->height; y += step) 
	{
		float* px = (float*) (velx->imageData + y * velx->widthStep);
		float* py = (float*) (vely->imageData + y * vely->widthStep);
		for(int x=0; x<imgFlow->width; x += step) 
		{
			if( px[x]>1 || py[x]>1 || px[x]<-1 || py[x]<-1) 
			{
				cvCircle(imgFlow, cvPoint( x, y ), 2, CVX_CIRCLE, -1);
				cvLine(imgFlow, cvPoint( x, y ), cvPoint( x+px[x]/2, y+py[x]/2 ), CVX_LINE, 1, 0);
			}
		}
	}
	cvSaveImage(output, imgFlow);
	
	std::cout<<"Avg time: "<<double(t2)<<" [ms]"<<std::endl;	

	return 0;
}  



int OpticalFlowOpenCV::runFromCamera(float lambda, int it)
{
	DWORD t1, t2 = 0;

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
	imgOld = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 1);
	cvCvtColor(imgTmp, imgOld, CV_BGR2GRAY);
	
	int count = 1;
	while (1)
	{
		imgTmp = cvQueryFrame(capture);
		imgNew = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 1);
		cvCvtColor(imgTmp, imgNew, CV_BGR2GRAY);
		
		IplImage* velx = cvCreateImage(cvGetSize(imgTmp),IPL_DEPTH_32F,1); 
		IplImage* vely = cvCreateImage(cvGetSize(imgTmp),IPL_DEPTH_32F,1);
		IplImage* imgFlow = cvCreateImage(cvGetSize(imgTmp), IPL_DEPTH_8U, 3);
		
		t1 = GetTickCount();
		cvSmooth( imgOld, imgOld, CV_BLUR, 3, 3, 0, 0 );
		cvSmooth( imgNew, imgNew, CV_BLUR, 3, 3, 0, 0 );
		cvCalcOpticalFlowHS(imgOld, imgNew, 0, velx, vely, lambda, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, it, 1e-6));
		t2 = t2 + GetTickCount() - t1;
		
		cvZero(imgFlow);
		int step = 4;
		for(int y=0; y<imgFlow->height; y += step) 
		{
			float* px = (float*) (velx->imageData + y * velx->widthStep);
			float* py = (float*) (vely->imageData + y * vely->widthStep);
			for(int x=0; x<imgFlow->width; x += step) 
			{
				if( px[x]>1 || py[x]>1 || px[x]<-1 || py[x]<-1) 
				{
					cvCircle(imgFlow, cvPoint( x, y ), 2, CVX_CIRCLE, -1);
					cvLine(imgFlow, cvPoint( x, y ), cvPoint( x+px[x]/2, y+py[x]/2 ), CVX_LINE, 1, 0);
				}
			}
		}
		cvShowImage(FLOW_TITLE, imgFlow);
    
		cvReleaseImage(&velx);
		cvReleaseImage(&vely);
		cvReleaseImage(&imgOld);
		cvReleaseImage(&imgFlow);
		imgOld = imgNew;
		
		if( cvWaitKey(10) == 27 )
		{
			std::cout<<"Avg time: "<<double(t2/count)<<" [ms]"<<std::endl;	
			break;
		}
		count++;
	}
	// destroy windows
	cvReleaseCapture( &capture );
	cvDestroyWindow(FLOW_TITLE);
	return 0;
}  

