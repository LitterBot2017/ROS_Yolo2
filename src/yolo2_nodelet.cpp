/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 ThundeRatz

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <yolo2/ImageDetections.h>

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

#include "darknet/yolo2.h"
#include "ros/ros.h"
#include "std_msgs/Int8.h"

// Camera Indices
#define DOWNWARD_CAMERA 0
#define FORWARD_CAMERA 1

// Image sizes
#define DOWNWARD_WIDTH 640
#define DOWNWARD_HEIGHT 360 // 530
#define FORWARD_WIDTH 640
#define FORWARD_HEIGHT 480

namespace
{
darknet::Detector yoloForward;
darknet::Detector yoloDownward;
ros::Publisher detectionsPublisher;
image_transport::Publisher imagePublisher;
image im = {};
float *image_data = nullptr;
ros::Time timestamp;
std::mutex mutex;
std::condition_variable im_condition;
const std::string NET_DATA = ros::package::getPath("yolo2") + "/data/";
double confidence, nms;
int cameraSelect = FORWARD_CAMERA;

sensor_msgs::Image::Ptr resizeImage(const sensor_msgs::ImageConstPtr& image, uint32_t width, uint32_t height) {
  sensor_msgs::Image::Ptr resized_image = boost::make_shared<sensor_msgs::Image>();

  resized_image->header = image->header;
  resized_image->height = height; //image->height;//height;
  resized_image->width = width; //image->width;//width;
  resized_image->encoding = image->encoding;
  resized_image->is_bigendian = image->is_bigendian;
  resized_image->step = image->step;

  resized_image->data.resize(width * height * 3);

  uint i = 0;
  for (uint32_t line = height; line; line--) {
    for (uint32_t column = width; column; column--) {
      for (uint32_t channel = 0; channel < 3; channel++) {
        resized_image->data[i] = image->data[i];
        i++;
      }
    }
  }

  return resized_image;
}

void setImage(const sensor_msgs::ImageConstPtr& resizedImage, const sensor_msgs::ImageConstPtr& image) {
  ROS_INFO("Height = %d and Width = %d", resizedImage->height, resizedImage->width);
  im = yoloDownward.convert_image(resizedImage);
  std::unique_lock<std::mutex> lock(mutex);
  if (image_data)
    free(image_data);
  timestamp = resizedImage->header.stamp;
  image_data = im.data;
  lock.unlock();
  im_condition.notify_one();
  imagePublisher.publish(image);
}

void downwardImageCallback(const sensor_msgs::ImageConstPtr& image) {
  if (cameraSelect == DOWNWARD_CAMERA) {
    ROS_INFO("Received downward camera image");
    //setImage(resizeImage(image, DOWNWARD_WIDTH, DOWNWARD_HEIGHT));
    //sensor_msgs::Image::Ptr resizedImage = resizeImage(image, DOWNWARD_WIDTH, DOWNWARD_HEIGHT);
    setImage(image, image);
  }
}

void forwardImageCallback(const sensor_msgs::ImageConstPtr& image) {
  if (cameraSelect == FORWARD_CAMERA) {
    ROS_INFO("Received forward camera image");
    //setImage(resizeImage(image, FORWARD_WIDTH, FORWARD_HEIGHT));
    //sensor_msgs::Image::Ptr resizedImage = resizeImage(image, FORWARD_WIDTH, FORWARD_HEIGHT);
    setImage(image, image);
  }
}

void cameraSelectCallback(const std_msgs::Int8::ConstPtr& msg) {
    ROS_INFO("I heard: [%d]", msg->data);
    cameraSelect = msg->data;
}

}  // namespace

namespace yolo2
{
class Yolo2Nodelet : public nodelet::Nodelet
{
 public:
  virtual void onInit()
  {
    ros::NodeHandle& node = getPrivateNodeHandle();

    node.param<double>("confidence", confidence, .8);
    node.param<double>("nms", nms, .4);

    std::string config = NET_DATA + "downward.cfg", weights = NET_DATA + "downward_grass.weights";
    yoloDownward.load(config, weights, 0.45, nms);

    config = NET_DATA + "forward.cfg";
    weights = NET_DATA + "forward_grass.weights";
    yoloForward.load(config, weights, 0.45, nms);

    image_transport::ImageTransport transport = image_transport::ImageTransport(node);
    downwardSubscriber = transport.subscribe("left/image", 1, downwardImageCallback);
    forwardSubscriber = transport.subscribe("right/image", 1, forwardImageCallback);

    cameraSelectSubscriber = node.subscribe("camera_select", 1, cameraSelectCallback);
    detectionsPublisher = node.advertise<yolo2::ImageDetections>("detections", 5);
    imagePublisher = transport.advertise("image_raw", 5);

    ROS_INFO("Initialized YOLO");
    yolo_thread = new std::thread(run_yolo);
  }

  ~Yolo2Nodelet() {
    yolo_thread->join();
    delete yolo_thread;
  }

 private:
  image_transport::Subscriber forwardSubscriber;
  image_transport::Subscriber downwardSubscriber;
  ros::Subscriber cameraSelectSubscriber;

  std::thread *yolo_thread;

  static void run_yolo() {

    while (ros::ok()) {

      ROS_INFO("Running YOLO");
      float *data;
      ros::Time stamp;
      {
        std::unique_lock<std::mutex> lock(mutex);
        while (!image_data)
          im_condition.wait(lock);
        data = image_data;
        image_data = nullptr;
        stamp = timestamp;
      }
      boost::shared_ptr<yolo2::ImageDetections> detections(new yolo2::ImageDetections);
      if (cameraSelect == DOWNWARD_CAMERA) {
        *detections = getValidDetections(yoloDownward.detect(data, DOWNWARD_WIDTH, FORWARD_HEIGHT),DOWNWARD_WIDTH, DOWNWARD_HEIGHT);
      } else if (cameraSelect == FORWARD_CAMERA) {
        *detections = getValidDetections(yoloForward.detect(data, FORWARD_WIDTH, FORWARD_HEIGHT), FORWARD_WIDTH, FORWARD_HEIGHT);
      }
      detections->header.stamp = stamp;
      detections->camera = cameraSelect;
      detectionsPublisher.publish(detections);
      free(data);
    }
  }


  static bool isValidDetection(yolo2::Detection& detection, int width, int height) {
    bool valid = true;
    if (detection.x < 0 || (detection.x + detection.width) > width)
      valid = false;

    if (detection.y < 0 || (detection.y) > height)//(detection.y + detection.height) > height)
      valid = false;

    return valid;
  }

  static yolo2::ImageDetections getValidDetections(yolo2::ImageDetections currDetections, int width, int height) {
    
    yolo2::ImageDetections detections;
    std::vector<yolo2::Detection> detectionsList;
    int numDetections = 0;
    float currMaxConfidence = 0;
    for (yolo2::Detection& detection : currDetections.detections) {
      if (isValidDetection(detection, width, height)) {
        if (detection.confidence > currMaxConfidence) {
          currMaxConfidence = detection.confidence;
          detectionsList.insert(detectionsList.begin(), detection);
          numDetections++;
        } else {
          detectionsList.push_back(detection);
          numDetections++;
        }
      }
    }
    detections.detections = detectionsList;
    detections.num_detections = numDetections;
    //publishBBox(detections);
    return detections;
  }
};
}  // namespace yolo2

PLUGINLIB_EXPORT_CLASS(yolo2::Yolo2Nodelet, nodelet::Nodelet)
