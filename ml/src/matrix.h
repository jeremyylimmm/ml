#pragma once

#include <cmath>
#include <initializer_list>
#include <cassert>
#include <random>
#include <functional>
#include <cstdio>

template <size_t NR, size_t NC>
struct Matrix {
    float data[NR * NC];

    Matrix(const std::initializer_list<float>& input) {
        assert(input.size() == NR * NC);
        auto it = input.begin();
        for (size_t i = 0; i < NR * NC; ++i) {
            data[i] = *it;
            ++it;
        }
    }

    Matrix()
    {
    }

    void dump(const char* path) {
        FILE* file;
        if (fopen_s(&file, path, "wb")) {
            assert(false && "couldn't dump matrix");
        }

        size_t nr = NR;
        size_t nc = NC;
        fwrite(&nr, sizeof(nr), 1, file);
        fwrite(&nc, sizeof(nc), 1, file);
        fwrite(data, sizeof(float), NR * NC, file);

        fclose(file);
    }

    void print(const char* name) {
        printf("%s = [ ", name);
        for (size_t i = 0; i < NR * NC; ++i) {
            printf("%f, ", data[i]);
        }
        printf("]\n");
    }

    static Matrix zero() {
        Matrix result;
        memset(result.data, 0, NR * NC * sizeof(float));
        return result;
    }

    static Matrix randn(std::default_random_engine generator) {
        std::normal_distribution<float> distribution(0.0f, 1.0f);

        Matrix result;
        for (size_t i = 0; i < NR * NC; ++i) {
            result.data[i] = distribution(generator);
        }
        return result;
    }

    float& at(size_t r, size_t c) {
        assert(r < NR && c < NC);
        return data[r * NC + c];
    }

    const float& at(size_t r, size_t c) const {
        return data[r * NC + c];
    }

    Matrix<NC, NR> T() const {
        Matrix<NC, NR> result;

        for (size_t r = 0; r < NR; ++r) {
            for (size_t c = 0; c < NC; ++c) {
                result.at(c, r) = at(r, c);
            }
        }

        return result;
    }

    Matrix<NR, NC>& operator+=(const Matrix<NR, NC>& right) {
        return *this = *this + right;
    }

    Matrix<NR, NC>& operator-=(const Matrix<NR, NC>& right) {
        return *this = *this - right;
    }

    Matrix<NR, NC> apply(const std::function<float(float)>& fn) const {
        Matrix<NR, NC> result;

        for (size_t i = 0; i < NR * NC; ++i) {
            result.data[i] = fn(data[i]);
        }

        return result;
    }
};

template<size_t NR1, size_t NC1, size_t NR2, size_t NC2>
inline Matrix<NR1, NC2> dot(const Matrix<NR1, NC1>& left, const Matrix<NR2, NC2>& right) {
    static_assert(NC1 == NR2, "mismatched matrix size");

    Matrix<NR1, NC2> result;

    for (size_t r = 0; r < NR1; ++r) {
        for (size_t c = 0; c < NC2; ++c)
        {
            float dot_product = 0.0f;

            for (size_t i = 0; i < NC1; ++i) {
                dot_product += left.at(r, i) * right.at(i, c);
            }

            result.at(r, c) = dot_product;
        }
    }

    return result;
}


template<size_t NR, size_t NC>
inline Matrix<NR, NC> operator+(const Matrix<NR, NC>& left, const Matrix<NR, NC>& right) {
    Matrix<NR, NC> result;

    for (size_t i = 0; i < NR * NC; ++i)
        result.data[i] = left.data[i] + right.data[i];

    return result;
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> operator-(const Matrix<NR, NC>& left, const Matrix<NR, NC>& right) {
    Matrix<NR, NC> result;

    for (size_t i = 0; i < NR * NC; ++i)
        result.data[i] = left.data[i] - right.data[i];

    return result;
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> operator*(const Matrix<NR, NC>& left, const Matrix<NR, NC>& right) {
    Matrix<NR, NC> result;

    for (size_t i = 0; i < NR * NC; ++i)
        result.data[i] = left.data[i] * right.data[i];

    return result;
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> operator*(float left, const Matrix<NR, NC>& right) {
    return right.apply([=](float x) {
        return left * x;
    });
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> operator/(float left, const Matrix<NR, NC>& right) {
    return right.apply([](float x) {
        return left / x;
    });
}

template<size_t NR, size_t NC>
inline float sum(const Matrix<NR, NC>& input) {
    float result = 0.0f;

    for (size_t i = 0; i < NR * NC; ++i)
    {
        result += input.data[i];
    }

    return result;
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> pow(const Matrix<NR, NC>& base, float power) {
    return base.apply([=](float x) {
        return powf(x, power);
    });
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> relu(const Matrix<NR, NC>& input) {
    return input.apply([](float x) {
        return fmaxf(0.0f, x);
    });
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> d_relu(const Matrix<NR, NC>& input) {
    return input.apply([](float x) {
        return (float)(x > 0.0f);
    });
}

float sigmoidf(float x) {
    return 1.0f/(1.0f + expf(-x));
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> sigmoid(const Matrix<NR, NC>& input) {
    return input.apply(sigmoidf);
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> d_sigmoid(const Matrix<NR, NC>& input) {
    return input.apply([](float x) {
        return sigmoidf(x) * (1.0f - sigmoidf(x));
    });
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> exp(const Matrix<NR, NC>& input) {
    return input.apply(expf);
}

template<size_t NR, size_t NC>
inline Matrix<NR, NC> softmax(const Matrix<NR, NC>& input) {
    auto expon = exp(input);
    float denom = sum(expon);

    return input.apply([=](float x) {
        return expf(x) / denom;
    });
}
