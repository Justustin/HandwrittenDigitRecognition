#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
using namespace cv;
using namespace cv::ml;
using namespace std;
const int IMAGE_SIZE = 28;


vector<vector<unsigned char>> ReadImages(const string& fileName) {
	ifstream file(fileName, ios::binary);

	char magicNumber[4];
	char nImagesBytes[4];
	char nRowsBytes[4];
	char nColsBytes[4];

	file.read(magicNumber,4);
	file.read(nImagesBytes,4);
	file.read(nRowsBytes,4);
	file.read(nColsBytes,4);

	cout << (int)static_cast<unsigned char>(nImagesBytes[0]) << " " << (int)static_cast<unsigned char>(nImagesBytes[1]) << " " << (int)static_cast<unsigned char>(nImagesBytes[2]) << " " << static_cast<int>(nImagesBytes[3]) << endl;

	int nImages = static_cast<unsigned char>(nImagesBytes[3] << 0) | (static_cast<unsigned char>(nImagesBytes[2]) << 8) | static_cast<unsigned char>(nImagesBytes[1] << 16) | (static_cast<unsigned char>(nImagesBytes[0]) << 24);
	int nRows = ((unsigned)nRowsBytes[3] << 0) | ((unsigned)nRowsBytes[2] << 8) | ((unsigned)nRowsBytes[1] << 16) | ((unsigned)nRowsBytes[0] << 24);
	int nCols = ((unsigned)nColsBytes[3] << 0) | ((unsigned)nColsBytes[2] << 8) | ((unsigned)nColsBytes[1] << 16) | ((unsigned)nColsBytes[0] << 24);

	cout << nImages << " " << nRows << " " << nCols << endl;
	vector<vector<unsigned char>> result;

	for (int i = 0; i < nImages; i++) {
		vector<unsigned char> image(nRows * nCols);
		file.read((char*)(image.data()), nRows * nCols);
		result.push_back(image);
	}
	file.close();
	return result;

}

vector<vector<unsigned char>> ReadLabels(const string& fileName) {
	ifstream file(fileName, ios::binary);

	char magicNumber[4];
	char nImagesBytes[4];

	file.read(magicNumber, 4);
	file.read(nImagesBytes, 4);

	cout << (int)static_cast<unsigned char>(nImagesBytes[0]) << " " << (int)static_cast<unsigned char>(nImagesBytes[1]) << " " << (int)static_cast<unsigned char>(nImagesBytes[2]) << " " << static_cast<int>(nImagesBytes[3]) << endl;

	int nImages = static_cast<unsigned char>(nImagesBytes[3] << 0) | (static_cast<unsigned char>(nImagesBytes[2]) << 8) | static_cast<unsigned char>(nImagesBytes[1] << 16) | (static_cast<unsigned char>(nImagesBytes[0]) << 24);


	vector<vector<unsigned char>> result;

	for (int i = 0; i < nImages; i++) {
		vector<unsigned char> image(1);
		file.read((char*)(image.data()), 1);
		result.push_back(image);
	}
	file.close();
	return result;

}

Ptr<KNearest> trainKNN(const vector<vector<unsigned char>>& images, const vector<vector<unsigned char>>& labels) {
	Mat trainingData(images.size(), IMAGE_SIZE * IMAGE_SIZE, CV_32F);
	Mat labelsMat(labels.size(), 1, CV_32S);

	// Flatten images and convert to CV_32F
	for (size_t i = 0; i < images.size(); ++i) {
		Mat imageMat(IMAGE_SIZE, IMAGE_SIZE, CV_8U, const_cast<uchar*>(images[i].data()));
		Mat imageMat32F;
		imageMat.convertTo(imageMat32F, CV_32F);
		imageMat32F.reshape(1, 1).copyTo(trainingData.row(static_cast<int>(i)));
		labelsMat.at<int>(static_cast<int>(i), 0) = labels[i][0];
	}

	Ptr<KNearest> knn = KNearest::create();
	knn->train(trainingData, ROW_SAMPLE, labelsMat);

	return knn;
}

int recognizeDigit(const Mat& testImage, const Ptr<KNearest>& knn) {
	Mat testImage32F;
	testImage.convertTo(testImage32F, CV_32F);
	Mat testImageReshaped = testImage32F.reshape(1, 1);

	// Perform k-nearest neighbors prediction
	Mat results, neighborResponses, dists;
	knn->findNearest(testImageReshaped, 1, results, neighborResponses, dists);

	return static_cast<int>(results.at<float>(0, 0));
}	
vector<vector<unsigned char>> ReadTestImages(const string& fileName) {
	ifstream file(fileName, ios::binary);

	char magicNumber[4];
	char nImagesBytes[4];
	char nRowsBytes[4];
	char nColsBytes[4];

	file.read(magicNumber, 4);
	file.read(nImagesBytes, 4);
	file.read(nRowsBytes, 4);
	file.read(nColsBytes, 4);

	int nImages = static_cast<unsigned char>(nImagesBytes[3] << 0) |
		(static_cast<unsigned char>(nImagesBytes[2]) << 8) |
		static_cast<unsigned char>(nImagesBytes[1] << 16) |
		(static_cast<unsigned char>(nImagesBytes[0]) << 24);

	int nRows = ((unsigned)nRowsBytes[3] << 0) |
		((unsigned)nRowsBytes[2] << 8) |
		((unsigned)nRowsBytes[1] << 16) |
		((unsigned)nRowsBytes[0] << 24);

	int nCols = ((unsigned)nColsBytes[3] << 0) |
		((unsigned)nColsBytes[2] << 8) |
		((unsigned)nColsBytes[1] << 16) |
		((unsigned)nColsBytes[0] << 24);

	vector<vector<unsigned char>> result;

	for (int i = 0; i < nImages; i++) {
		vector<unsigned char> image(nRows * nCols);
		file.read((char*)(image.data()), nRows * nCols);
		result.push_back(image);
	}

	file.close();
	return result;
}

int main()
{
	vector<vector<unsigned char>> images = ReadImages("D:/CODING/C++/Assignment6/train-images.idx3-ubyte");
	vector<vector<unsigned char>> labels = ReadLabels("D:/CUHK/CSC3002/assignment6/train-labels.idx1-ubyte");

	Ptr<KNearest> knn = trainKNN(images, labels);

	vector<vector<unsigned char>> testImages = ReadImages("D:/CUHK/CSC3002/assignment6/t10k-images.idx3-ubyte");

    // Loop over test images
    for (const auto& testImage : testImages) {
        Mat testImageMat(IMAGE_SIZE, IMAGE_SIZE, CV_8U, const_cast<uchar*>(testImage.data()));

        resize(testImageMat, testImageMat, Size(IMAGE_SIZE, IMAGE_SIZE));
        int recognizedDigit = recognizeDigit(testImageMat, knn);

        cout << "Recognized digit: " << recognizedDigit << endl;

        imshow("Test Image", testImageMat);
        waitKey(0);
    }

	return 0;
}