// source; https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/objectDetection/objectDetection.cpp

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>


using namespace std;
using namespace cv;

String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

void detectAndDisplay(Mat frame);

int main() {
	VideoCapture capture;
	Mat frame;

	if(!face_cascade.load("../xml/haarcascade_frontalface_alt.xml")) {
		printf("--(!)Error loading face cascade\n");
		return -1;
	}

	if(!eyes_cascade.load("../xml/haarcascade_eye_tree_eyeglasses.xml")){
		printf("--(!)Error loading eyes cascade\n");
		return -1;
	}

	capture.open(0);

	if (!capture.isOpened()) {
		printf("--(!)Error opening video capture\n");
		return -1;
	}

	while(capture.read(frame)) {
		if(frame.empty()) {
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		detectAndDisplay(frame);

		char c = (char)waitKey(10);
		if(c == 27) // escape
			break;
	}

	return 0;
}

void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

	for(size_t i = 0; i < faces.size(); i++) {
		Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
		ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;

		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

		for (size_t j = 0; j < eyes.size(); j++) {
			Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
		}
	}

	imshow("face detector 2000", frame);
}