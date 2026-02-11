#include "../include/IsolationForest.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

IsolationForest::IsolationForest(const Params& params)
    : IsolationForest(params.nTrees, params.sampleSize, params.seed)
{
}

IsolationForest::IsolationForest(int nTrees, int sampleSize, uint32_t seed)
    : _nTrees(nTrees)
      , _sampleSize(sampleSize)
      , _rng(seed)
{
    if (_nTrees <= 0 || _sampleSize <= 1)
    {
        throw std::invalid_argument("Invalid IsolationForest parameters");
    }
}

IsolationForest::Result IsolationForest::detect(const Matrix& X, const Params& params)
{
    if (X.empty())
    {
        throw std::invalid_argument("IsolationForest::detect: empty matrix");
    }
    if (params.nTrees <= 0)
    {
        throw std::invalid_argument("IsolationForest::detect: nTrees <= 0");
    }
    if (params.sampleSize <= 1)
    {
        throw std::invalid_argument("IsolationForest::detect: sampleSize <= 1");
    }
    if (!(params.contamination >= 0.0 && params.contamination < 1.0))
    {
        throw std::invalid_argument("IsolationForest::detect: contamination should be in [0, 1)");
    }

    IsolationForest model(params);

    model.fit(std::make_shared<Matrix>(X));

    const auto scores = model.scoreAll(std::make_shared<Matrix>(X));

    const int N = static_cast<int>(scores.size());

    int k = static_cast<int>(std::floor(params.contamination * N));
    if (k < 0)
    {
        k = 0;
    }
    if (k > N)
    {
        k = N;
    }

    double threshold = std::numeric_limits<double>::infinity();
    if (k > 0)
    {
        VectorD tmp = scores;
        auto kth = tmp.begin() + (N - k);
        std::nth_element(tmp.begin(), kth, tmp.end());
        threshold = *kth;
    }

    Result res;
    res.scores = std::move(scores);
    res.labels.assign(N, 0);
    res.threshold = threshold;

    if (k > 0)
    {
        for (int i = 0; i < N; ++i)
        {
            if (res.scores[i] >= threshold)
            {
                res.labels[i] = 1;
                res.anomalyIndices.push_back(i);
            }
        }
    }

    return res;
}

void IsolationForest::fit(MatrixPtr X)
{
    if (!X || X->empty())
    {
        throw std::invalid_argument("fit(): empty dataset");
    }

    _X = std::move(X);
    _N = static_cast<int>(_X->size());
    _D = static_cast<int>((*_X)[0].size());

    for (const auto& row : *_X)
    {
        if ((int)row.size() != _D)
            throw std::invalid_argument("fit(): ragged matrix");
    }

    _nSub = std::min(_sampleSize, _N);
    _maxDepth = static_cast<int>(std::ceil(std::log2(_nSub)));

    precomputeHarmonics(_nSub);

    _trees.clear();
    _trees.reserve(_nTrees);

    VectorI indices(_N);
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i < _nTrees; ++i)
    {
        std::shuffle(indices.begin(), indices.end(), _rng);
        VectorI subs(indices.begin(), indices.begin() + _nSub);
        _trees.emplace_back(buildTree(subs, 0));
    }
}

double IsolationForest::scoreOne(const VectorD& x) const
{
    checkFitted();
    if ((int)x.size() != _D)
        throw std::invalid_argument("scoreOne(): dimension mismatch");

    double sum = 0.0;
    for (const auto& tree : _trees)
    {
        sum += pathLength(tree.get(), x, 0);
    }

    double Eh = sum / _trees.size();
    return std::pow(2.0, -Eh / cFactor(_nSub));
}

IsolationForest::VectorD
IsolationForest::scoreAll(MatrixPtr X) const
{
    if (!X) throw std::invalid_argument("scoreAll(): null matrix");

    VectorD out;
    out.reserve(X->size());
    for (const auto& row : *X)
        out.push_back(scoreOne(row));
    return out;
}

std::unique_ptr<IsolationForest::Node>
IsolationForest::buildTree(const VectorI& indices, int depth) const
{
    auto node = std::make_unique<Node>();
    node->size = indices.size();

    if (node->size <= 1 || depth >= _maxDepth)
    {
        node->isLeaf = true;
        return node;
    }

    std::uniform_int_distribution<int> featDist(0, _D - 1);

    int feature = -1;
    double minv = 0.0, maxv = 0.0;

    for (int t = 0; t < 32; ++t)
    {
        int j = featDist(_rng);
        auto [mn, mx] = minmaxFeature(indices, j);
        if (mn < mx)
        {
            feature = j;
            minv = mn;
            maxv = mx;
            break;
        }
    }

    if (feature == -1)
    {
        node->isLeaf = true;
        return node;
    }

    std::uniform_real_distribution<double> thrDist(minv, maxv);
    double thr = thrDist(_rng);

    VectorI left, right;
    for (int idx : indices)
    {
        ((*_X)[idx][feature] <= thr ? left : right).push_back(idx);
    }

    if (left.empty() || right.empty())
    {
        node->isLeaf = true;
        return node;
    }

    node->feature = feature;
    node->threshold = thr;
    node->left = buildTree(left, depth + 1);
    node->right = buildTree(right, depth + 1);

    return node;
}

std::pair<double, double>
IsolationForest::minmaxFeature(const VectorI& indices, int j) const
{
    double mn = (*_X)[indices[0]][j];
    double mx = mn;

    for (int i : indices)
    {
        double v = (*_X)[i][j];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    return {mn, mx};
}

double IsolationForest::pathLength(const Node* node,
                                   const VectorD& x,
                                   int depth) const
{
    if (node->isLeaf)
    {
        return depth + cFactor(node->size);
    }

    if (x[node->feature] <= node->threshold)
        return pathLength(node->left.get(), x, depth + 1);

    return pathLength(node->right.get(), x, depth + 1);
}

double IsolationForest::cFactor(int m) const
{
    if (m <= 1) return 0.0;
    double H = _harmonic[m - 1];
    return 2.0 * H - 2.0 * (m - 1.0) / m;
}

void IsolationForest::precomputeHarmonics(int maxN)
{
    _harmonic.assign(maxN + 1, 0.0);
    for (int i = 1; i <= maxN; ++i)
        _harmonic[i] = _harmonic[i - 1] + 1.0 / i;
}

void IsolationForest::checkFitted() const
{
    if (!_X || _trees.empty())
        throw std::runtime_error("IsolationForest not fitted");
}
