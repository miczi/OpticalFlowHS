#include "EdgeDetectionOpenCV.hpp"

EdgeDetectionOpenCV::EdgeDetectionOpenCV(char* src, char* input, char* output, int lt, int ht, int it)
{
	edge_thresh = ht;
	this->src = src;
	this->input = input;
	this->output = output;
	this->lt = lt;
	this->ht = ht;
	this->iterations = it;
}

EdgeDetectionOpenCV::EdgeDetectionOpenCV(char* src, int lt, int ht)
{
	edge_thresh = ht;
	this->src = src;
	this->lt = lt;
	this->ht = ht;
}

EdgeDetectionOpenCV::~EdgeDetectionOpenCV(void)
{
	delete image;
	delete gray;
	delete edge;
	delete capture;
}

int EdgeDetectionOpenCV::runDetection()
{
	if (strcmp(src, "-hd") == 0)
	{
		DWORD t1, t2 = 0;
		image = cvLoadImage(input, 1);
		if (!image){
			std::cout<<"Input image error.\n";
			return -1;
		}
		gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		cvCvtColor(image, gray, CV_BGR2GRAY);
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);

		t1 = GetTickCount();
		for (int i = 0; i < iterations; i++){
			cvCvtColor(image, gray, CV_BGR2GRAY);
			cvSmooth( gray, edge, CV_BLUR, 3, 3, 0, 0 );
			cvNot( gray, edge );
			cvCanny(gray, edge, (float)edge_thresh, (float)edge_thresh*3, 3);	
		}
		t2 = t2 + GetTickCount() - t1;	
		std::cout<<"Avg time: "<<double(t2/iterations)<<" [ms]"<<std::endl;
		cvSaveImage(output, edge);
		return 0;
	}
	else
	{
		DWORD t1, t2 = 0;
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
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);

		cvNamedWindow( "Edge Detection", CV_WINDOW_AUTOSIZE );
		int count = 1;
		while( 1 ) {

			image = cvQueryFrame( capture );
			if( !image ) {
				fprintf( stderr, "ERROR: frame is null...\n" );
				getchar();
				break;
			}
			t1 = GetTickCount();
			cvCvtColor(image, gray, CV_BGR2GRAY);
			cvSmooth( gray, edge, CV_BLUR, 3, 3, 0, 0 );
			cvNot( gray, edge );
			cvCanny(gray, edge, (float)edge_thresh, (float)edge_thresh*3, 3);
			t2 = t2 + GetTickCount() - t1;
			cvShowImage( "Edge Detection", edge );
			if( cvWaitKey(10) == 27 ){
				std::cout<<"Avg kernel time: "<<double(t2/count)<<" [ms]"<<std::endl;	
				break;
			}
			count++;
		}
		cvReleaseCapture( &capture );
		cvDestroyWindow( "Edge Detection" );
		return 0;
	}
}
