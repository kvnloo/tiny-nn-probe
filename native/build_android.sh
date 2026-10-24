#!/bin/sh
set -eu
: "${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT on the admitted Linux build worker}"
out=${1:-results/native-nn}
mkdir -p "$out"
compiler="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
"$compiler" --version
sha256sum native/tiny_nn.c native/mlp.c native/build_android.sh
"$compiler" -std=c11 -O3 -fno-lto -ffp-contract=off -fPIE -pie native/tiny_nn.c native/mlp.c -lm -o "$out/tiny-native"
sha256sum "$out/tiny-native"
printf 'EDGE_NATIVE_BINARY_BEGIN\n'
base64 "$out/tiny-native"
printf '\nEDGE_NATIVE_BINARY_END\n'
