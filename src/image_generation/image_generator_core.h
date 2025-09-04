// src/image_generation/image_generator_core.h

#ifndef IMAGE_GENERATOR_CORE_H
#define IMAGE_GENERATOR_CORE_H

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <map>

namespace py = pybind11;

class ImageGeneratorCore {
public:
    ImageGeneratorCore(int conceptual_dim, int image_width, int image_height);

    // Generates a pixel-based image from a conceptual vector.
    py::array_t<unsigned char> generate_image(py::array_t<double> conceptual_vector);

    // Ensures conceptual consistency, such as persistent colors and motion.
    py::array_t<unsigned char> ensure_consistency(py::array_t<unsigned char> current_image,
                                                  py::array_t<double> conceptual_vector);
    
private:
    int conceptual_dim_;
    int image_width_;
    int image_height_;
    
    // A mock representation of a trained generator model.
    std::vector<double> generator_weights;

    // Helper function to convert conceptual vector to image data.
    std::vector<unsigned char> _conceptual_to_pixels(const std::vector<double>& conceptual_vector);
};

#endif // IMAGE_GENERATOR_CORE_H
