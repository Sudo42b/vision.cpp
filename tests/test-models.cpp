#include "visp/vision.hpp"
#include "util/string.hpp"

#include "testing.hpp"

namespace visp {

void compare_images(std::string_view name, image_view result, float tolerance = 0.01f) {
    path reference_path = test_dir().reference / name;
    path result_path = test_dir().results / name;

    image_save(result, result_path.string().c_str());
    image_data reference = image_load(reference_path.string().c_str());

    test_set_info(
        format(
            "while comparing images {} and {}", relative(result_path).string(),
            relative(reference_path).string()));
    test_with_tolerance with(tolerance);
    CHECK_IMAGES_EQUAL(result, reference);
}

void test_mobile_sam(backend_type bt) {
    path model_path = test_dir().models / "mobile_sam.gguf";
    path input_path = test_dir().input / "cat-and-hat.jpg";

    backend_device b = backend_init(bt);
    sam_model model = sam_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    sam_encode(model, input);
    image_data mask_box = sam_compute(model, box_2d{{180, 110}, {505, 330}});
    image_data mask_point =  sam_compute(model, i32x2{200, 300});

    char const* suffix = bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";
    compare_images(format("mobile_sam-box{}", suffix), mask_box);
    compare_images(format("mobile_sam-point{}", suffix), mask_point);
}

TEST_CASE(test_mobile_sam_cpu) {
    test_mobile_sam(backend_type::cpu);
}
TEST_CASE(test_mobile_sam_gpu) {
    test_mobile_sam(backend_type::gpu);
}

void test_birefnet(backend_type bt) {
    path model_path = test_dir().models / "birefnet_lite-f16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";
    std::string name = "birefnet";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    birefnet_model model = birefnet_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    image_data output = birefnet_compute(model, input);

    compare_images(name, output);
}

TEST_CASE(test_birefnet_cpu) {
    test_birefnet(backend_type::cpu);
}
TEST_CASE(test_birefnet_gpu) {
    test_birefnet(backend_type::gpu);
}

void test_migan(backend_type bt) {
    path model_path = test_dir().models / "migan_512_places2-f16.gguf";
    path image_path = test_dir().input / "bench-image.jpg";
    path mask_path = test_dir().input / "bench-mask.png";
    std::string name = "migan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    migan_model model = migan_load_model(model_path.string().c_str(), b);
    image_data image = image_load(image_path.string().c_str());
    image_data mask = image_load(mask_path.string().c_str());
    image_data output = migan_compute(model, image, mask);
    image_data composited = image_alpha_composite(output, image, mask);

    compare_images(name, composited);
}

TEST_CASE(test_migan_cpu) {
    test_migan(backend_type::cpu);
}
TEST_CASE(test_migan_gpu) {
    test_migan(backend_type::gpu);
}

void test_esrgan(backend_type bt) {
    path model_path = test_dir().models / "RealESRGAN_x4plus_anime_6Bh.gguf";
    path input_path = test_dir().input / "vase-and-bowl.jpg";
    std::string name = "esrgan";
    name += bt == backend_type::cpu ? "-cpu.png" : "-gpu.png";

    backend_device b = backend_init(bt);
    esrgan_model model = esrgan_load_model(model_path.string().c_str(), b);
    image_data input = image_load(input_path.string().c_str());
    image_data output = esrgan_compute(model, input);

    compare_images(name, output);
}

TEST_CASE(test_esrgan_cpu) {
    test_esrgan(backend_type::cpu);
}
TEST_CASE(test_esrgan_gpu) {
    test_esrgan(backend_type::gpu);
}

} // namespace visp