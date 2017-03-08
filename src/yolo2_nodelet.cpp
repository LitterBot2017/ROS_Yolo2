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

namespace
{
darknet::Detector yoloForward;
darknet::Detector yoloDownward;
ros::Publisher publisher;
image im_forward = {};
image im_downward = {};
float *image_data_forward = nullptr;
float *image_data_downward = nullptr;
ros::Time timestamp;
std::mutex mutex;
std::condition_variable im_condition_forward;
std::condition_variable im_condition_downward;
const std::string NET_DATA = ros::package::getPath("yolo2") + "/data/";
double confidence, nms;
int cameraSelect = 0;

void downwardImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_INFO("Received downward camera image");
  im_downward = yoloDownward.convert_image(msg);
  std::unique_lock<std::mutex> lock(mutex);
  if (image_data_downward)
    free(image_data_downward);
  timestamp = msg->header.stamp;
  image_data_downward = im_downward.data;
  lock.unlock();
  im_condition_downward.notify_one();
}

void forwardImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_INFO("Received forward camera image");
  im_forward = yoloForward.convert_image(msg);
  std::unique_lock<std::mutex> lock(mutex);
  if (image_data_forward)
    free(image_data_forward);
  timestamp = msg->header.stamp;
  image_data_forward = im_forward.data;
  lock.unlock();
  im_condition_forward.notify_one();
}

void cameraSelectCallback(const std_msgs::Int8::ConstPtr& msg)
{
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

    std::string config = NET_DATA + "downward.cfg", weights = NET_DATA + "downward.weights";
    yoloDownward.load(config, weights, confidence, nms);

    config = NET_DATA + "forward.cfg";
    weights = NET_DATA + "forward.weights";
    yoloForward.load(config, weights, confidence, nms);

    image_transport::ImageTransport transport = image_transport::ImageTransport(node);
    downwardSubscriber = transport.subscribe("left/image", 1, downwardImageCallback);
    forwardSubscriber = transport.subscribe("right/image", 1, forwardImageCallback);
    ROS_INFO("Subscribed to downward camera");
    cameraSelectSubscriber = node.subscribe("cameraSelect", 1, cameraSelectCallback);
    publisher = node.advertise<yolo2::ImageDetections>("detections", 5);

    yolo_thread = new std::thread(run_yolo);
  }

  ~Yolo2Nodelet()
  {
    yolo_thread->join();
    delete yolo_thread;
  }

 private:
  image_transport::Subscriber forwardSubscriber;
  image_transport::Subscriber downwardSubscriber;
  ros::Subscriber cameraSelectSubscriber;

  std::thread *yolo_thread;

  static void run_yolo()
  {
    while (ros::ok())
    {
      float *data;
      ros::Time stamp;
      {
        std::unique_lock<std::mutex> lock(mutex);
        if (cameraSelect == 0) {
          while (!image_data_downward)
            im_condition_downward.wait(lock);
          data = image_data_downward;
          image_data_downward = nullptr;
        } else if (cameraSelect == 1) {
          while (!image_data_forward)
            im_condition_forward.wait(lock);
          data = image_data_forward;
          image_data_forward = nullptr;
        }
        stamp = timestamp;
      }
      boost::shared_ptr<yolo2::ImageDetections> detections(new yolo2::ImageDetections);
      if (cameraSelect == 0) {
        *detections = yoloDownward.detect(data);
      } else if (cameraSelect == 1) {
        *detections = yoloForward.detect(data);
      }
      detections->header.stamp = stamp;
      publisher.publish(detections);
      free(data);
    }
  }
};
}  // namespace yolo2

PLUGINLIB_EXPORT_CLASS(yolo2::Yolo2Nodelet, nodelet::Nodelet)
