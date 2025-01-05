//
// Created by viciopoli on 02/01/25.
//

#ifndef SRC_EXTRACTORBASE_H
#define SRC_EXTRACTORBASE_H

#include <opencv2/opencv.hpp>

namespace ov_core {

    class ExtractorBase {
    public:
        ExtractorBase() = default;

        virtual ~ExtractorBase() = default;

        // Pure virtual method for feature extraction
        virtual void detectAndCompute(
                const cv::Mat &img,
                std::vector <cv::KeyPoint> &keypoints,
                cv::Mat &descriptors) = 0;

        // Static factory method to create extractors
        static cv::Ptr <ExtractorBase> createExtractor();

        // Clone method for duplicating
        virtual cv::Ptr <ExtractorBase> clone() const = 0;

        virtual const std::string type() const = 0;
    };
}
#endif //SRC_EXTRACTORBASE_H
