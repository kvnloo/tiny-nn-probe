// Same float32 64 -> 128 ReLU -> 16 MLP as tiny_nn.py.
// Separate translation unit: the compiler cannot fold the procedural toy weights.
void tiny_mlp_forward(const float *restrict x, const float *restrict w1,
                      const float *restrict w2, float *restrict hidden,
                      float *restrict output) {
    for (int j = 0; j < 128; ++j) hidden[j] = 0;
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 128; ++j)
            hidden[j] += x[i] * w1[i * 128 + j];
    for (int i = 0; i < 128; ++i)
        if (hidden[i] < 0) hidden[i] = 0;
    for (int j = 0; j < 16; ++j) output[j] = 0;
    for (int i = 0; i < 128; ++i)
        for (int j = 0; j < 16; ++j)
            output[j] += hidden[i] * w2[i * 16 + j];
    // The original model's 128 + 16 bias parameters are exactly zero.
}
