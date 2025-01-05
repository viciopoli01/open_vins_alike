//
// Created by viciopoli on 02/01/25.
//

#ifndef SRC_EXTRACTORORB_H
#define SRC_EXTRACTORORB_H

#include "ExtractorBase.h"
#include <vector>

namespace ov_core {

    class ExtractorORB : public ExtractorBase {
    public:
        // Default constructor with optional parameters for ORB creation
        explicit ExtractorORB(int nFeatures = 500, float scaleFactor = 1.2f, int nLevels = 8)
                : orb(cv::ORB::create(nFeatures, scaleFactor, nLevels)) {
        }

        // Override detectAndCompute
        void detectAndCompute(
                const cv::Mat &img,
                std::vector <cv::KeyPoint> &keypoints,
                cv::Mat &descriptors) override {

            if (!keypoints.empty()) {
                orb->compute(img, keypoints, descriptors);
            } else {
                orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
            }
        }

        static cv::Ptr <ExtractorBase> createExtractor() {
            return cv::makePtr<ExtractorORB>();
        }

        cv::Ptr <ExtractorBase> clone() const override {
            return this->createExtractor();
        }

        const std::string type() const override {
            return "ORB";
        }

    private:
        cv::Ptr <cv::ORB> orb;
    };


}

#endif //SRC_EXTRACTORORB_H
