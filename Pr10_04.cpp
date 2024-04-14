#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;

void processEyes(const Mat& gray, Mat& frame, Rect face, CascadeClassifier& eyesCascade) {
    vector<Rect> eyes;
    eyesCascade.detectMultiScale(gray(face), eyes, 3, 2, 0, Size(5, 5));

#pragma omp parallel for
    for (size_t j = 0; j < eyes.size(); ++j) {
        Rect eye = eyes[j];
        Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
        int radius = cvRound((eye.width + eye.height) * 0.25);
        circle(frame(face), center, radius, Scalar(63, 181, 110), 2);
    }
}

void processSmiles(const Mat& gray, Mat& frame, Rect face, CascadeClassifier& smileCascade) {
    vector<Rect> smiles;
    smileCascade.detectMultiScale(gray(face), smiles, 1.565, 30, 0, Size(30, 30));

#pragma omp parallel for
    for (size_t k = 0; k < smiles.size(); ++k) {
        Rect smile = smiles[k];
        rectangle(frame(face), smile, Scalar(181, 63, 169), 2);
    }
}

int main() {
    CascadeClassifier faceCascade, eyesCascade, smileCascade;
    faceCascade.load("haarcascade_frontalface_alt.xml");
    eyesCascade.load("haarcascade_eye_tree_eyeglasses.xml");
    smileCascade.load("haarcascade_smile.xml");

    VideoCapture cap("C:/Users/User/Desktop/ZUA.mp4");

    if (!cap.isOpened())
    {
        cout << "Error" << endl;
        return -1;
    }

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    int video = VideoWriter::fourcc('X', 'V', 'I', 'D');
    VideoWriter videoOutput("C:/Users/User/Desktop/output_video.mp4", video, 20, Size(width, height));

    if (!videoOutput.isOpened())
    {
        cout << "Error with file" << endl;
        return -1;
    }

    auto start = chrono::steady_clock::now();

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "End" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 2, 3, 0, Size(20, 20));

#pragma omp parallel for
        for (size_t i = 0; i < faces.size(); ++i) {
            Rect face = faces[i];
            rectangle(frame, face, Scalar(58, 64, 224), 2);

            processEyes(gray, frame, face, eyesCascade);
            processSmiles(gray, frame, face, smileCascade);

            blur(frame(face), frame(face), Size(3, 3));
        }

        imshow("Video", frame);
        videoOutput.write(frame);

        if (waitKey(25) == 'q')
            break;
    }

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    cout << "The time spent on the program: " << elapsed_seconds.count() << " seconds" << std::endl;

    cap.release();
    videoOutput.release();
    destroyAllWindows();

    return 0;
}
