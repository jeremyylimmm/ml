#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "matrix.h"

static FILE* load_buffer(const char* path) {
    FILE* file;
    if (fopen_s(&file, path, "rb")) {
        printf("Failed to load buffer '%s'.\n", path);
        exit(1);
    }
    return file;
}

template<typename T>
inline T reverse_integer(T x) {
    static_assert(std::is_integral<T>(), "not an integral type");

    uint8_t* in = (uint8_t*)&x;
    uint8_t out[sizeof(T)];

    for (size_t i = 0; i < sizeof(T); ++i) {
        out[sizeof(T) - i - 1] = in[i];
    }

    return *(T*)out;
}

template<typename T>
inline T load(FILE* file) {
    T result;
    fread(&result, sizeof(T), 1, file);
    return result;
}

template<typename T>
inline T load_big_endian(FILE* file) {
    T result = load<T>(file);
    return reverse_integer(result);
}

int main() {
    FILE* file = load_buffer("data/train-labels.idx1-ubyte");

    if (load_big_endian<uint32_t>(file) != 2049) {
        printf("Unrecognized dataset.\n");
        return 1;
    }

    uint32_t num_labels = load_big_endian<uint32_t>(file);
    char* labels = new char[num_labels];
    fread(labels, 1, num_labels, file);

    fclose(file);
    file = load_buffer("data/train-images.idx3-ubyte");

    if (load_big_endian<uint32_t>(file) != 2051) {
        printf("Unrecognized dataset.\n");
        return 1;
    }

    uint32_t num_images = load_big_endian<uint32_t>(file);
    assert(num_images == num_labels);

    uint32_t image_rows = load_big_endian<uint32_t>(file);
    uint32_t image_cols = load_big_endian<uint32_t>(file);
    (void)image_rows;
    (void)image_cols;
    assert(image_rows == 28 && image_cols == 28);

    uint8_t* image_data = new uint8_t[num_images * 28 * 28];
    fread(image_data, 28 * 28, num_images, file);

    const size_t batch_size = 128;
    const size_t epochs = 64;
    const float learning_rate = 1.0f;

    std::default_random_engine generator;

    auto w1 = Matrix<16, 28*28>::randn(generator);
    auto b1 = Matrix<16, 1>::zero();

    auto w2 = Matrix<16, 16>::randn(generator);
    auto b2 = Matrix<16, 1>::zero();

    auto w3 = Matrix<10, 16>::randn(generator);
    auto b3 = Matrix<10, 1>::zero();

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (uint32_t image_base = 0; image_base < num_images; image_base += batch_size)
        {
            auto a_dw1 = Matrix<16, 28 * 28>::zero();
            auto a_db1 = Matrix<16, 1>::zero();

            auto a_dw2 = Matrix<16, 16>::zero();
            auto a_db2 = Matrix<16, 1>::zero();

            auto a_dw3 = Matrix<10, 16>::zero();
            auto a_db3 = Matrix<10, 1>::zero();

            float loss = 0.0f;
            size_t num_correct = 0;

            for (size_t batch_i = 0; batch_i < batch_size; ++batch_i)
            {
                size_t image = (image_base + batch_i) % num_images;

                Matrix<28 * 28, 1> a0;
                for (size_t texel = 0; texel < 28 * 28; ++texel)
                    a0.at(texel, 0) = ((float)image_data[image * 28 * 28 + texel]) / 255.0f;

                auto z1 = dot(w1, a0) + b1;
                auto a1 = sigmoid(z1);

                auto z2 = dot(w2, a1) + b2;
                auto a2 = sigmoid(z2);

                auto z3 = dot(w3, a2) + b3;
                auto a3 = softmax(z3);

                size_t digit = 0;
                float digit_prob = -INFINITY;
                for (size_t i = 0; i < 10; ++i) {
                    float x = a3.at(i, 0);
                    if (x > digit_prob) {
                        digit_prob = x;
                        digit = i;
                    }
                }

                if (digit == labels[image]) {
                    num_correct++;
                }

                auto y = Matrix<10, 1>::zero();
                y.at(labels[image], 0) = 1.0f;

                loss += sum(pow(a3 - y, 2.0f));

                auto dz3 = 2 * (a3 - y);
                auto dw3 = dot(dz3, a2.T());
                auto db3 = dz3;

                auto da2 = dot(w3.T(), dz3);
                auto dz2 = da2 * d_sigmoid(z2);
                auto dw2 = dot(dz2, a1.T());
                auto db2 = dz2;

                auto da1 = dot(w2.T(), dz2);
                auto dz1 = da1 * d_sigmoid(z1);
                auto dw1 = dot(dz1, a0.T());
                auto db1 = dz1;

                a_dw1 += dw1;
                a_db1 += db1;

                a_dw2 += dw2;
                a_db2 += db2;

                a_dw3 += dw3;
                a_db3 += db3;
            }

            loss /= (float)batch_size;

            float factor = learning_rate / (float)batch_size;

            w1 -= factor * a_dw1;
            b1 -= factor * a_db1;

            w2 -= factor * a_dw2;
            b2 -= factor * a_db2;

            w3 -= factor * a_dw3;
            b3 -= factor * a_db3;

            float accuracy = (float)num_correct / (float)batch_size * 100.0f;

            printf("Loss: %f (accuracy %2.0f%%)\n", loss, accuracy);
        }
    }

    w1.print("w1");
    b1.print("b1");
    w2.print("w2");
    b2.print("b2");
    w3.print("w3");
    b3.print("b3");

    return 0;
}