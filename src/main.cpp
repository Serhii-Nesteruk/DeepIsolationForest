#include <iostream>
#include "../include/IsolationForest.h"

int main()
{
    using IF = IsolationForest;

    auto X = IF::Matrix{
        {0.0, 0.0},
        {0.1, 0.0},
        {0.0, 0.1},
        {0.1, 0.1},
        {10.0, 10.0}   // <- anomaly
    };

    IsolationForest::Params params;
    params.contamination = 0.2;
    auto res = IsolationForest::detect(X, params);

    for (int idx : res.anomalyIndices)
    {
        std::cout << "anomaly idx=" << idx << " score=" << res.scores[idx] <<
            " el: { " << X[idx][0] << ", " << X[idx][1] << " }" << std::endl;
    }

    return 0;
}