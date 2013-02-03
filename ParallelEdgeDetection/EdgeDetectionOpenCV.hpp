#include <cv.h>
#include <highgui.h>
#include <windows.h>
#include <iostream>

/**
* EdgeDetectionOpenCV 
* Class implements OpenCV Edge detection
*/
class EdgeDetectionOpenCV
{
private:
	int edge_thresh;
	IplImage *image;
	IplImage *gray;
	IplImage *edge;
	CvCapture* capture;

	char* src;
	char* input;
	char* output;
	int lt;
	int ht;
	int iterations; 
public:
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
	*/
	EdgeDetectionOpenCV(char* src, char* input, char* output, int lt, int ht, int it);
	/** 
	* Constructor 
	* Initialize member variables
	* @param name name of sample (const char*) 
	* @param src source (hard disc or camera) (const char*)
	* @param lt low treshold (int)
	* @param ht hight treshold (int)
	*/
	EdgeDetectionOpenCV(char* src,int lt, int ht);
	~EdgeDetectionOpenCV(void);

	/**
	* Run OpenCV edge detection
	*/
	int runDetection();
};

