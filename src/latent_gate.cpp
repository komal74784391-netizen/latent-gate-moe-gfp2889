// Official implementation of Latent Gate Routing for MoE transformers.
#include <vector>

struct LatentGate {
    int experts;
    int hidden;

    std::vector<float> route(const std::vector<float>& x) const {
        std::vector<float> logits(experts, 0.0f);
        for (int e = 0; e < experts; ++e) {
            for (int h = 0; h < hidden; ++h) {
                logits[e] += x[h] * weight(e, h);
            }
        }
        return logits;
    }

    float weight(int e, int h) const { return 0.001f * (e + 1) * (h + 1); }
};
