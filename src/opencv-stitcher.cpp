#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Check if video file was provided
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video file>" << endl;
        return -1;
    }
    string input_path = string(argv[1]);
    cout << "C++: Start stitching file \"" << input_path << "\"" << endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Open the video file
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        cerr << "Could not open video file: " << input_path << endl;
        return -1;
    }

    // Get the video properties
    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    int interval = 10;
    // int level = 7;
    //int frame_couple = 4;
    // vector<int> levels = {128, 32, 8, 2};
    int level = 6;
    int frame_couple = 2;
    vector<int> levels = {32, 16, 8, 4, 2};
    int level_value = pow(2, level);

    cout << "Setup done: " << endl;
    cout << "\tinterval=" << interval << endl;
    cout << "\tlevel=" << level << endl;
    cout << "\tlevel_value=" << level << endl;
    cout << "\tframe_couple=" << level << endl;
    cout << "\tlevels=";
    for (const auto i: levels) {
        cout << (i != levels[0] ? ", " : "") << i;
    }
    cout << endl;

    cout << "Start collecting frames" << endl;
    // Create a vector to store the input frames
    Mat frame;
    vector<Mat> frames;
    // Loop over all frames in the video
    int i = 0;
    while (cap.read(frame)) {
        if (i % interval != 0) {
          i += 1;
          continue;
        }
        // Resize the frame to half its original size
        resize(frame, frame, Size(width/2, height/2));

        // Add the resized frame to the vector of input frames
        frames.push_back(frame);
        // cout << "Added frame " << i << " (" << frames.size() << ")" << endl;

        i += 1;
        if (i >= interval * level_value) {
            break;
        }
    }
    cout << "Added " << frames.size() << " frames" << endl;

    cout << "Create stitcher" << endl;
    // Create a stitching object and stitch the input frames
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
    //auto blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND, true);
    auto blender = detail::Blender::createDefault(detail::Blender::FEATHER, true);
    stitcher->setBlender(blender);
    // stitcher->setRegistrationResol(1.0); // Default: 0.6
    // stitcher->setSeamEstimationResol(1.0); // Default: 0.1
    stitcher->setCompositingResol(2.0);

    cout << "Start stitching" << endl;
    Mat result;
    for (int i = 0; i < levels.size(); i++) {
        vector<Mat> results;
        for (int j = 0; j < levels[i]; j += frame_couple) {
            vector<Mat> pairings;
            for (int k = 0; k < frame_couple; k++) {
                pairings.push_back(frames[j + k]);
            }

            // cout << "Stitching " << j << endl;
            Stitcher::Status status = stitcher->stitch(pairings, result);
            if (status == Stitcher::OK) {
                results.push_back(result);
            } else {
                cerr << "Stitching failed" << endl;
                return -1;
            }
        }

        frames.clear();
        for (int j = 0; j < results.size(); j++) {
            frames.push_back(results[j]);
        }

        // cout << "Results: " << results.size() << endl;
        // imshow("Intermediate Stitched Image", results[0]);
        // waitKey(0);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Stitching took " << duration.count() << " milliseconds" << std::endl;

    imshow("Stitched Image", frames[0]);
    waitKey(0);

    string old_path = string(input_path);
    string output_path = input_path + ".png";
    size_t dot_pos = input_path.find_last_of(".");
    
    if (dot_pos != std::string::npos && dot_pos > input_path.find_last_of("/")) {
        std::string filename_no_ext = input_path.substr(0, dot_pos);
        output_path = filename_no_ext  + ".png";
    }
    cout << "Write file to \"" << output_path << "\"" << endl;
    imwrite(output_path, frames[0]);

    // Release the video file and close the window
    cap.release();
    destroyAllWindows();

    return 0;
}

