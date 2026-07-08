#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <mutex>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include "deps/httplib.h"

// ============================================================================
// Constants - matching Python app.py exactly
// ============================================================================
static const int DISP_WIDTH = 1024;
static const float CAR_CONF_THRESHOLD = 0.25f;
static const float SIGN_CONF_THRESHOLD = 0.30f;
static const float SPEED_CONF_THRESHOLD = 0.60f; // Increased from 0.15 to reduce false positives
static const float NMS_THRESHOLD = 0.45f;
static const int REQUIRED_FRAMES = 3;
static const int COCO_CAR_CLASS = 2;

// Traffic light class IDs in best.onnx
static const int TRAFFIC_LIGHT_RED = 0;
static const int TRAFFIC_LIGHT_GREEN = 1;
static const int TRAFFIC_LIGHT_YELLOW = 2;

// Speed limit class names for best(3).onnx
static const std::map<int, std::string> SPEED_LIMIT_NAMES = {
    {0, "10 MPH"}, {1, "15 MPH"}, {2, "20 MPH"}, {3, "25 MPH"},
    {4, "30 MPH"}, {5, "35 MPH"}, {6, "40 MPH"}, {7, "45 MPH"},
    {8, "50 MPH"}, {9, "55 MPH"}, {10, "60 MPH"}, {11, "65 MPH"},
    {12, "70 MPH"}, {13, "75 MPH"}
};

// Sign model class names for best.onnx
static const std::map<int, std::string> SIGN_MODEL_NAMES = {
    {0, "traffic_light_red"}, {1, "traffic_light_green"}, {2, "traffic_light_yellow"},
    {3, "speed_30"}, {4, "speed_50"}, {5, "speed_60"}, {6, "speed_80"}, {7, "speed_100"}
};

// ============================================================================
// Detection structure
// ============================================================================
struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

// ============================================================================
// YOLOv8 ONNX Model wrapper
// ============================================================================
class YOLOModel {
public:
    YOLOModel(Ort::Env& env, const std::wstring& modelPath, int numClasses, int inputSize = 640)
        : numClasses_(numClasses), inputSize_(inputSize) {
        
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(12); // Use 12 threads for Jetson Orin CPU max performance
        opts.SetInterOpNumThreads(1);
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // CPU ONLY - No CUDA provider added

        // CPU ONLY - No CUDA provider added

        session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        auto inputName = session_->GetInputNameAllocated(0, allocator);
        inputName_ = inputName.get();
        
        auto outputName = session_->GetOutputNameAllocated(0, allocator);
        outputName_ = outputName.get();

        auto inputShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "[INFO] Loaded model, input shape: [";
        for (size_t i = 0; i < inputShape.size(); i++) {
            std::cout << inputShape[i];
            if (i < inputShape.size() - 1) std::cout << ", ";
        }
        std::cout << "], classes: " << numClasses_ << std::endl;
    }

    // Preprocess: letterbox resize + normalize + NCHW
    cv::Mat preprocess(const cv::Mat& frame, float& scaleX, float& scaleY, int& padX, int& padY) {
        int origW = frame.cols, origH = frame.rows;
        float scale = std::min((float)inputSize_ / origW, (float)inputSize_ / origH);
        int newW = (int)(origW * scale), newH = (int)(origH * scale);
        
        padX = (inputSize_ - newW) / 2;
        padY = (inputSize_ - newH) / 2;
        scaleX = scale;
        scaleY = scale;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(newW, newH));

        cv::Mat padded(inputSize_, inputSize_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(padX, padY, newW, newH)));

        // BGR -> RGB
        cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

        // Normalize to [0, 1] and convert to float
        padded.convertTo(padded, CV_32F, 1.0 / 255.0);

        return padded;
    }

    // Run inference and return detections
    std::vector<Detection> detect(const cv::Mat& frame, float confThreshold) {
        float scaleX, scaleY;
        int padX, padY;
        cv::Mat input = preprocess(frame, scaleX, scaleY, padX, padY);

        // HWC -> NCHW
        std::vector<float> inputTensor(1 * 3 * inputSize_ * inputSize_);
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < inputSize_; y++) {
                for (int x = 0; x < inputSize_; x++) {
                    inputTensor[c * inputSize_ * inputSize_ + y * inputSize_ + x] =
                        input.at<cv::Vec3f>(y, x)[c];
                }
            }
        }

        // Create input tensor
        std::array<int64_t, 4> inputShape = {1, 3, inputSize_, inputSize_};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensor.data(), inputTensor.size(),
            inputShape.data(), inputShape.size());

        // Run inference
        const char* inputNames[] = {inputName_.c_str()};
        const char* outputNames[] = {outputName_.c_str()};
        auto outputTensors = session_->Run(Ort::RunOptions{nullptr},
            inputNames, &inputOrt, 1, outputNames, 1);

        // Parse output: YOLOv8 output is [1, 4+numClasses, numDetections]
        auto& output = outputTensors[0];
        auto outputShape = output.GetTensorTypeAndShapeInfo().GetShape();
        const float* outputData = output.GetTensorData<float>();

        int channels = (int)outputShape[1]; // 4 + numClasses
        int numDets = (int)outputShape[2];  // e.g. 8400

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        for (int i = 0; i < numDets; i++) {
            // Extract box: cx, cy, w, h
            float cx = outputData[0 * numDets + i];
            float cy = outputData[1 * numDets + i];
            float w  = outputData[2 * numDets + i];
            float h  = outputData[3 * numDets + i];

            // Find best class
            float maxScore = 0;
            int maxClass = 0;
            for (int c = 0; c < numClasses_; c++) {
                float score = outputData[(4 + c) * numDets + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClass = c;
                }
            }

            if (maxScore < confThreshold) continue;

            // Convert from letterbox coords to original frame coords
            float x1 = (cx - w / 2 - padX) / scaleX;
            float y1 = (cy - h / 2 - padY) / scaleY;
            float x2 = (cx + w / 2 - padX) / scaleX;
            float y2 = (cy + h / 2 - padY) / scaleY;

            // Clamp
            x1 = std::max(0.f, x1);
            y1 = std::max(0.f, y1);
            x2 = std::min((float)frame.cols, x2);
            y2 = std::min((float)frame.rows, y2);

            boxes.push_back(cv::Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1)));
            confidences.push_back(maxScore);
            classIds.push_back(maxClass);
        }

        // Apply NMS
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, NMS_THRESHOLD, nmsIndices);

        std::vector<Detection> results;
        for (int idx : nmsIndices) {
            results.push_back({boxes[idx], confidences[idx], classIds[idx]});
        }
        return results;
    }

private:
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;
    int numClasses_;
    int inputSize_;
};

// ============================================================================
// Dark-pixel analysis (balloon killer) - same as Python
// ============================================================================
bool isRealTrafficLight(const cv::Mat& frame, int x1, int y1, int x2, int y2) {
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(frame.cols, x2);
    y2 = std::min(frame.rows, y2);
    if (x2 <= x1 || y2 <= y1) return false;

    cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    // Extract V channel
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat vChannel = channels[2];

    // Count dark pixels (V < 80)
    int darkPixels = cv::countNonZero(vChannel < 80);
    float ratio = (float)darkPixels / (float)vChannel.total();
    return ratio >= 0.25f;
}

// ============================================================================
// Point-in-polygon test (same as cv2.pointPolygonTest)
// ============================================================================
bool pointInPolygon(const std::vector<cv::Point>& polygon, cv::Point pt) {
    return cv::pointPolygonTest(polygon, cv::Point2f((float)pt.x, (float)pt.y), false) >= 0;
}

// ============================================================================
// HTML page (same as Python)
// ============================================================================
static const char* HTML_PAGE = R"(
<html>
  <head>
    <title>Lane & Sign Detection</title>
    <style>
      body { text-align: center; font-family: sans-serif; background: #0f1115; color: #e5e7eb; margin: 0; padding: 20px;}
      img { max-width: 100%; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    </style>
  </head>
  <body>
    <h2>Live Lane & Traffic Light Detection (C++ Pipeline)</h2>
    <img src="/video_feed">
  </body>
</html>
)";

// ============================================================================
// Global state for video pipeline
// ============================================================================
struct PipelineState {
    std::mutex mtx;
    std::vector<uchar> currentJpeg;
    double currentVideoTimeSec = 0.0;
    bool running = true;
    bool switchSource = false;
    std::string newSourceType = "";
};

// ============================================================================
// Main video processing loop (runs in its own thread)
// ============================================================================
void videoProcessingLoop(
    YOLOModel& carModel,
    YOLOModel& signModel,
    YOLOModel& speedModel,
    PipelineState& state)
{
    try {
    // Open video source
    std::string videoFile = "edited_ultimate_video.mp4";
    cv::VideoCapture cap;
    std::string source;

    if (std::ifstream(videoFile).good()) {
        cap.open(videoFile);
        source = "VIDEO";
        std::cout << "[INFO] Using video file: " << videoFile << std::endl;
    } else {
        cap.open(0);
        source = "LIVE";
        std::cout << "[INFO] Video not found! Using LIVE dashcam" << std::endl;
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Could not open video source!" << std::endl;
        return;
    }

    int origWidth = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int origHeight = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = 640;
    int height = (int)(origHeight * ((float)width / origWidth));
    int frameArea = width * height;

    std::cout << "[INFO] Resolution: " << width << "x" << height << " | Source: " << source << std::endl;

    // Lane polygon (same as Python)
    std::vector<cv::Point> lanePts = {
        cv::Point((int)(width * 0.35), height),
        cv::Point((int)(width * 0.45), (int)(height * 0.60)),
        cv::Point((int)(width * 0.55), (int)(height * 0.60)),
        cv::Point((int)(width * 0.65), height)
    };

    // Temporal smoothing
    int redCount = 0, greenCount = 0, yellowCount = 0;
    std::string currentSpeedLimit;

    while (state.running) {
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            if (state.switchSource) {
                cap.release();
                if (state.newSourceType == "camera") {
                    cap.open(0);
                    source = "LIVE";
                    std::cout << "[INFO] Switched to LIVE camera" << std::endl;
                } else {
                    cap.open("edited_ultimate_video.mp4");
                    source = "VIDEO";
                    std::cout << "[INFO] Switched to VIDEO file" << std::endl;
                }
                state.switchSource = false;
            }
        }

        cv::Mat frame;
        bool success = cap.read(frame);

        if (!success) {
            if (source == "VIDEO") {
                // Loop video
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                redCount = greenCount = yellowCount = 0;
                continue;
            } else {
                std::cerr << "[ERROR] Could not read frame from camera!" << std::endl;
                break;
            }
        }

        cv::resize(frame, frame, cv::Size(width, height));

        // Run inference on all 3 models (GPU)
        auto carDets = carModel.detect(frame, CAR_CONF_THRESHOLD);
        auto signDets = signModel.detect(frame, SIGN_CONF_THRESHOLD);
        auto speedDets = speedModel.detect(frame, SPEED_CONF_THRESHOLD);

        bool carAheadDetected = false;
        bool redThisFrame = false, greenThisFrame = false, yellowThisFrame = false;

        // 1. Filter Cars (lane detection) - only COCO class 2 (car)
        for (auto& det : carDets) {
            if (det.classId != COCO_CAR_CLASS) continue;

            int cx = det.box.x + det.box.width / 2;
            int cy = det.box.y + det.box.height;
            int carArea = det.box.width * det.box.height;

            if (pointInPolygon(lanePts, cv::Point(cx, cy)) && carArea > (int)(frameArea * 0.003)) {
                carAheadDetected = true;
                // Draw orange bounding box for the car ahead
                cv::rectangle(frame, det.box, cv::Scalar(0, 165, 255), 2);
                cv::putText(frame, "CAR AHEAD", cv::Point(det.box.x, det.box.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
            }
        }

        // 2. Filter Speed Limits (speed_model)
        for (auto& det : speedDets) {
            float aspect = (float)det.box.height / std::max(det.box.width, 1);
            if (det.confidence > SPEED_CONF_THRESHOLD && aspect >= 1.0f && aspect <= 1.8f) {
                auto it = SPEED_LIMIT_NAMES.find(det.classId);
                std::string name = (it != SPEED_LIMIT_NAMES.end()) ? it->second : "";
                
                // Strip " MPH" for the HUD display
                std::string limit = name;
                size_t pos = limit.find(" MPH");
                if (pos != std::string::npos) limit = limit.substr(0, pos);
                currentSpeedLimit = limit;

                // Draw bounding box
                cv::rectangle(frame, det.box, cv::Scalar(255, 255, 255), 2);
                cv::putText(frame, name, cv::Point(det.box.x, std::max(det.box.y - 10, 0)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            }
        }

        // 3. Filter Traffic Signs & Lights
        for (auto& det : signDets) {
            int cls = det.classId;
            float conf = det.confidence;

            // Skip non-traffic-light classes
            if (cls != TRAFFIC_LIGHT_RED && cls != TRAFFIC_LIGHT_GREEN && cls != TRAFFIC_LIGHT_YELLOW) {
                continue;
            }

            // Confidence filter
            if (conf < SIGN_CONF_THRESHOLD) continue;

            // Dark pixel analysis (balloon killer)
            if (!isRealTrafficLight(frame, det.box.x, det.box.y,
                    det.box.x + det.box.width, det.box.y + det.box.height)) {
                continue;
            }

            cv::Scalar color;
            if (cls == TRAFFIC_LIGHT_RED) {
                color = cv::Scalar(0, 0, 255);
                redThisFrame = true;
            } else if (cls == TRAFFIC_LIGHT_GREEN) {
                color = cv::Scalar(0, 255, 0);
                greenThisFrame = true;
            } else if (cls == TRAFFIC_LIGHT_YELLOW) {
                color = cv::Scalar(0, 255, 255);
                yellowThisFrame = true;
            }

            auto it = SIGN_MODEL_NAMES.find(cls);
            std::string name = (it != SIGN_MODEL_NAMES.end()) ? it->second : ("Sign " + std::to_string(cls));

            cv::rectangle(frame, det.box, color, 2);
            cv::putText(frame, name, cv::Point(det.box.x, std::max(det.box.y - 10, 0)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }

        // Temporal smoothing
        redCount = redThisFrame ? (redCount + 1) : 0;
        greenCount = greenThisFrame ? (greenCount + 1) : 0;
        yellowCount = yellowThisFrame ? (yellowCount + 1) : 0;

        bool redOk = redCount >= REQUIRED_FRAMES;
        bool greenOk = greenCount >= REQUIRED_FRAMES;
        bool yellowOk = yellowCount >= REQUIRED_FRAMES;

        // Dashboard HUD
        std::string statusMsg = "CLEAR";
        cv::Scalar statusColor(0, 255, 0);

        if (redOk) {
            statusMsg = "RED LIGHT: STOP";
            statusColor = cv::Scalar(0, 0, 255);
        } else if (yellowOk) {
            statusMsg = "YELLOW LIGHT: SLOW DOWN";
            statusColor = cv::Scalar(0, 255, 255);
        } else if (greenOk && carAheadDetected) {
            statusMsg = "GREEN LIGHT BUT CAR AHEAD: Maintain Distance";
            statusColor = cv::Scalar(0, 165, 255);
        } else if (greenOk) {
            statusMsg = "GREEN LIGHT: GO AHEAD";
            statusColor = cv::Scalar(0, 255, 0);
        } else if (carAheadDetected) {
            statusMsg = "CAR AHEAD: Maintain Distance";
            statusColor = cv::Scalar(0, 165, 255);
        }

        // Scale back up for display
        int dispHeight = (int)(height * ((float)DISP_WIDTH / width));
        cv::resize(frame, frame, cv::Size(DISP_WIDTH, dispHeight));

        // LIVE badge
        if (source == "LIVE") {
            cv::circle(frame, cv::Point(DISP_WIDTH - 30, 30), 10, cv::Scalar(0, 0, 255), -1);
            cv::putText(frame, "LIVE", cv::Point(DISP_WIDTH - 80, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        // Draw main HUD
        cv::rectangle(frame, cv::Point(10, 10), cv::Point(850, 95), cv::Scalar(0, 0, 0), -1);
        cv::putText(frame, "STATUS: " + statusMsg, cv::Point(20, 45),
            cv::FONT_HERSHEY_SIMPLEX, 0.9, statusColor, 2, cv::LINE_AA);

        std::string speedText = currentSpeedLimit.empty() ?
            "SPEED LIMIT: UNKNOWN" : ("SPEED LIMIT: " + currentSpeedLimit);
        cv::putText(frame, speedText, cv::Point(20, 80),
            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        // Encode to JPEG
        std::vector<uchar> jpegBuf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 75};
        cv::imencode(".jpg", frame, jpegBuf, params);

        double videoTimeSec = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;

        // Update shared state
        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.currentJpeg = std::move(jpegBuf);
            state.currentVideoTimeSec = videoTimeSec;
        }
    }

    cap.release();
    } catch (const std::exception& e) {
        std::cerr << "Thread Exception: " << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "Unknown thread exception." << std::endl;
        exit(1);
    }
}
// ============================================================================
// Main
// ============================================================================
int main() {
    try {
        std::cout << "\n=== Traffic Light & Car Detection (C++ Pipeline) ===" << std::endl;
        std::cout << "Open http://localhost:5000 in your browser\n" << std::endl;

        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TrafficDetection");

        std::cout << "[INFO] Loading car model (yolov8n_quant.onnx)..." << std::endl;
        YOLOModel carModel(env, L"yolov8n_quant.onnx", 80);     // COCO: 80 classes

        std::cout << "[INFO] Loading sign model (best_quant.onnx)..." << std::endl;
        YOLOModel signModel(env, L"best_quant.onnx", 8);          // 8 sign classes

        std::cout << "[INFO] Loading speed model (best (3)_quant.onnx)..." << std::endl;
        YOLOModel speedModel(env, L"best (3)_quant.onnx", 14, 960);    // 14 speed limit classes, 960x960 input

        // Shared state for MJPEG streaming
        PipelineState state;

        // Start video processing in background thread
        std::thread processingThread(videoProcessingLoop,
            std::ref(carModel), std::ref(signModel), std::ref(speedModel), std::ref(state));

        // HTTP Server
        httplib::Server svr;

        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            std::ifstream ifs("index.html");
            if (ifs.good()) {
                std::stringstream buffer;
                buffer << ifs.rdbuf();
                res.set_content(buffer.str(), "text/html");
            } else {
                res.set_content(HTML_PAGE, "text/html");
            }
        });

        svr.Get("/image.png", [](const httplib::Request&, httplib::Response& res) {
            std::ifstream ifs("image.png", std::ios::binary);
            if (ifs.good()) {
                std::stringstream buffer;
                buffer << ifs.rdbuf();
                res.set_content(buffer.str(), "image/png");
            } else {
                res.status = 404;
            }
        });

        svr.Get("/time", [&state](const httplib::Request&, httplib::Response& res) {
            double current_time = 0.0;
            {
                std::lock_guard<std::mutex> lock(state.mtx);
                current_time = state.currentVideoTimeSec;
            }
            res.set_content(std::to_string(current_time), "text/plain");
        });

        svr.Post("/set_source", [&state](const httplib::Request& req, httplib::Response& res) {
            if (req.has_param("type")) {
                std::string type = req.get_param_value("type");
                std::lock_guard<std::mutex> lock(state.mtx);
                state.switchSource = true;
                state.newSourceType = type;
            }
            res.set_content("OK", "text/plain");
        });

        svr.Get("/video_feed", [&state](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider(
                "multipart/x-mixed-replace; boundary=frame",
                [&state](size_t /*offset*/, httplib::DataSink& sink) {
                    while (state.running) {
                        std::vector<uchar> jpeg;
                        {
                            std::lock_guard<std::mutex> lock(state.mtx);
                            jpeg = state.currentJpeg;
                        }

                        if (!jpeg.empty()) {
                            std::string header = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
                            if (!sink.write(header.data(), header.size())) break;
                            if (!sink.write((const char*)jpeg.data(), jpeg.size())) break;
                            if (!sink.write("\r\n", 2)) break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // Limit to ~33fps
                    }
                    return true;
                },
                [](bool /*success*/) {}
            );
        });

        std::cout << "[INFO] Server starting on http://0.0.0.0:5000" << std::endl;
        svr.listen("0.0.0.0", 5000);

        state.running = false;
        processingThread.join();
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
        return 1;
    }

    return 0;
}
