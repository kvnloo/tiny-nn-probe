#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void tiny_mlp_forward(const float *, const float *, const float *, float *, float *);
static float input[64], w1[64 * 128], w2[128 * 16], hidden[128], output[16];
static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}
static int compare_ns(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
static void json_string(const char *text) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p < 0x20) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}
int main(void) {
    enum { REPEATS = 1000 };
    uint64_t samples[REPEATS], sum_ns = 0;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    int affinity_known = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    if (!affinity_known) perror("sched_getaffinity");
    uint64_t mask = 0;
    if (affinity_known)
        for (int i = 0; i < 64; ++i) if (CPU_ISSET(i, &allowed)) mask |= 1ULL << i;
    char cpuset[256];
    FILE *cpuset_file = fopen("/proc/self/cpuset", "r");
    int cpuset_known = cpuset_file && fgets(cpuset, sizeof(cpuset), cpuset_file);
    if (!cpuset_known) perror("/proc/self/cpuset");
    if (cpuset_file) fclose(cpuset_file);
    if (cpuset_known) cpuset[strcspn(cpuset, "\r\n")] = 0;
    for (int i = 0; i < 64; ++i) {
        input[i] = 1;
        for (int j = 0; j < 128; ++j) w1[i * 128 + j] = (i % 4 + 1) / 64.0f;
    }
    for (int i = 0; i < 128; ++i)
        for (int j = 0; j < 16; ++j) w2[i * 16 + j] = (j % 8 - 3) / 128.0f;
    tiny_mlp_forward(input, w1, w2, hidden, output);
    for (int j = 0; j < 16; ++j)
        if (!isfinite(output[j]) || fabsf(output[j] - 2.5f * (j % 8 - 3)) > 1e-6f) {
            fprintf(stderr, "fixture mismatch at output %d: %.9g\n", j, output[j]);
            return 3;
        }
    for (int i = 0; i < 20; ++i) tiny_mlp_forward(input, w1, w2, hidden, output);
    for (int i = 0; i < REPEATS; ++i) {
        uint64_t start = now_ns();
        tiny_mlp_forward(input, w1, w2, hidden, output);
        samples[i] = now_ns() - start;
        sum_ns += samples[i];
    }
    if (!sum_ns) { fputs("clock did not advance\n", stderr); return 4; }
    qsort(samples, REPEATS, sizeof(samples[0]), compare_ns);
    printf("{\"pid\":%ld,\"cpu_allowed_mask_low64\":", (long)getpid());
    if (affinity_known) printf("\"%llx\"", (unsigned long long)mask); else fputs("null", stdout);
    fputs(",\"cpuset\":", stdout);
    if (cpuset_known) json_string(cpuset); else fputs("null", stdout);
    printf(",\"parameters\":10384,\"dtype\":\"float32\",\"shape\":[64,128,16],\"batch\":1,\"expected_ok\":true,\"actual\":[[");
    for (int j = 0; j < 16; ++j) printf("%s%.9g", j ? "," : "", output[j]);
    printf("]],\"native_c\":{\"repetitions\":1000,\"median_ms\":%.9g,\"p95_ms\":%.9g,\"throughput_items_s\":%.9g},\"method\":\"20 warmups; 1000 independent CLOCK_MONOTONIC wall samples; dense float32 C, not BLAS; throughput=1e9/mean_ns; zero biases elided\"}\n",
           (samples[499] + samples[500]) / 2e6, samples[949] / 1e6,
           1e9 * REPEATS / (double)sum_ns);
    return 0;
}
