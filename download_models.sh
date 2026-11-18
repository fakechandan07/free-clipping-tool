#!/bin/bash
# Downloads the free AI models the clipper uses into ./models (about 240 MB).
# Safe to run again: anything already downloaded is skipped.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p models/yamnet

get() {  # get <url> <destination> <name>
  if [ -s "$2" ]; then echo "  ✓ $3 (already there)"; return 0; fi
  echo "  ↓ $3"
  if curl -fL --retry 3 --progress-bar -o "$2.part" "$1"; then
    mv "$2.part" "$2"
  else
    rm -f "$2.part"; echo "  ✗ couldn't download $3. Check your internet and run this again."; return 1
  fi
}

echo "Downloading AI models (free, about 240 MB, only the first time)…"
ok=0
get "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" \
    models/ggml-base.en.bin "speech-to-text (Whisper base.en, 148 MB)" || ok=1
get "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    models/face_detection_yunet_2023mar.onnx "face finder (YuNet)" || ok=1
get "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
    models/face_recognition_sface_2021dec.onnx "face recognizer (SFace, 39 MB)" || ok=1
get "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" \
    models/emotion-ferplus-8.onnx "expression reader (FER+, 35 MB)" || ok=1

if [ -s models/yamnet/yamnet.onnx ] && [ -s models/yamnet/yamnet.data ]; then
  echo "  ✓ sound detector (YAMNet) (already there)"
else
  echo "  ↓ sound detector (YAMNet, 15 MB)"
  tmp="$(mktemp -d)"
  if curl -fL --retry 3 --progress-bar -o "$tmp/y.zip" \
       "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/yamnet/releases/v0.66.0/yamnet-onnx-float.zip" \
     && unzip -q -o "$tmp/y.zip" -d "$tmp/y"; then
    onnx="$(find "$tmp/y" -name '*.onnx' | head -1)"; data="$(find "$tmp/y" -name '*.data' | head -1)"
    if [ -n "$onnx" ] && [ -n "$data" ]; then
      cp "$onnx" models/yamnet/yamnet.onnx && cp "$data" models/yamnet/yamnet.data
    else
      echo "  ✗ the YAMNet download looked different than expected"; ok=1
    fi
  else
    echo "  ✗ couldn't download YAMNet. Check your internet and run this again."; ok=1
  fi
  rm -rf "${tmp:?}"
fi
[ $ok -eq 0 ] && echo "Models ready." || echo "Some models are missing (the clipper still works, with fewer features)."
exit $ok
