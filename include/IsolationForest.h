#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

class IsolationForest
{
public:
    using VectorD = std::vector<double>;
    using VectorI = std::vector<int>;
    using VectorUI8 = std::vector<uint8_t>;
    using Matrix = std::vector<VectorD>;

    using MatrixPtr = std::shared_ptr<const Matrix>;

    struct Result
    {
        VectorD scores;
        VectorUI8 labels;
        VectorI anomalyIndices;
        double threshold = 0.0f;
    };

    struct Params
    {
        int nTrees = 300;
        int sampleSize = 256;
        uint32_t seed = 42;
        double contamination = 0.01; // expected fraction of anomalies in [0, 1)
    };

    explicit IsolationForest(const Params& params);
    explicit IsolationForest(int nTrees = 300, int sampleSize = 256, uint32_t seed = 42);

    static Result detect(const Matrix& X, const Params& params);

    void fit(MatrixPtr X);

    [[nodiscard]]
    double scoreOne(const VectorD& x) const;

    [[nodiscard]]
    VectorD scoreAll(MatrixPtr X) const;

private:
    struct Node
    {
        bool isLeaf = false;
        int feature = -1;
        double threshold = 0.0;
        int size = 0;

        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

private:
    [[nodiscard]]
    std::unique_ptr<Node> buildTree(const VectorI& indices, int depth) const;

    [[nodiscard]]
    std::pair<double, double> minmaxFeature(const VectorI& indices, int j) const;

    [[nodiscard]]
    double pathLength(const Node* node, const VectorD& x, int depth) const;

    [[nodiscard]]
    double cFactor(int m) const;

    void precomputeHarmonics(int maxN);
    void checkFitted() const;

private:
    int _nTrees;
    int _sampleSize;

    int _nSub = 0;
    int _maxDepth = 0;

    MatrixPtr _X;
    int _N = 0;
    int _D = 0;

    mutable std::mt19937 _rng;

    std::vector<std::unique_ptr<Node>> _trees;
    std::vector<double> _harmonic;
};
