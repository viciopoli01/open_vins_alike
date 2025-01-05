//
// Created by viciopoli on 02/01/25.
//

#ifndef SRC_EXTRACTORALIKE_H
#define SRC_EXTRACTORALIKE_H

#include "ExtractorBase.h"
#include <vector>

#include <ALIKE_cpp/image_loader.hpp>
#include <ALIKE_cpp/alike.hpp>
#include <ALIKE_cpp/simple_tracker.hpp>
#include <ALIKE_cpp/utils.hpp>

namespace ov_core {

    class ExtractorALIKE : public ExtractorBase {
    public:
        ExtractorALIKE(
                const std::string &model_path,
                bool use_cuda = true,
                int top_k = -1,
                float scores_th = 0.2,
                int n_limit = 1000,
                bool subpixel = true
        )
                : alike(model_path, use_cuda, 2, top_k, scores_th, n_limit, subpixel),
                  model_path_(model_path),
                  use_cuda_(use_cuda),
                  top_k_(top_k),
                  scores_th_(scores_th),
                  n_limit_(n_limit),
                  subpixel_(subpixel),
        device(use_cuda ? torch::kCUDA : torch::kCPU){}

        void detectAndCompute(
                const cv::Mat &img,
                std::vector <cv::KeyPoint> &keypoints,
                cv::Mat &descriptors
        ) override {
            // Clear keypoints if not empty
            if (!keypoints.empty()) {
                keypoints.clear();
            }
            descriptors.release();

            torch::Tensor score_map, descriptor_map;
            torch::Tensor keypoints_t, dispersitys_t, kptscores_t, descriptors_t;

            // Convert to RGB for consistency with model requirements
            cv::Mat img_rgb;
            cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

            // Convert to tensor and normalize
            auto img_tensor = alike::mat2Tensor(img_rgb).permute({2, 0, 1}).unsqueeze(0)
                                      .to(device)
                                      .to(torch::kFloat) / 255;

            // Perform extraction and detection
            alike.extract(img_tensor, score_map, descriptor_map);
            alike.detectAndCompute(score_map, descriptor_map,
                                   keypoints_t, dispersitys_t,
                                   kptscores_t, descriptors_t);

            // Convert results back to OpenCV format
            alike.toOpenCVFormat(keypoints_t, dispersitys_t, kptscores_t, descriptors_t, keypoints, descriptors);
        }

        static cv::Ptr <ExtractorBase> createExtractor(
                const std::string &model_path,
                bool use_cuda = true,
                int top_k = -1,
                float scores_th = 0.2,
                int n_limit = 1000,
                bool subpixel = true
        ) {
            return cv::makePtr<ExtractorALIKE>(model_path, use_cuda, top_k, scores_th, n_limit, subpixel);
        }

        cv::Ptr <ExtractorBase> clone() const override {
            this->createExtractor(model_path_, use_cuda_, top_k_, scores_th_, n_limit_, subpixel_);
        }

        const std::string type() const override {
            return "ALIKE";
        }

    private:
        alike::ALIKE alike;
        std::string model_path_;
        bool use_cuda_ = true;
        int top_k_ = -1;
        float scores_th_ = 0.2;
        int n_limit_ = 1000;
        bool subpixel_ = true;
        torch::Device device;
    };


}

#endif //SRC_EXTRACTORALIKE_H
