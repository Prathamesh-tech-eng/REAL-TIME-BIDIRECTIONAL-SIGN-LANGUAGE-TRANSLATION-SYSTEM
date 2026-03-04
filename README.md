<p align="center">
  <img src="https://img.shields.io/badge/Platform-Android%20%7C%20iOS%20%7C%20Web-blue" alt="Platform" />
  <img src="https://img.shields.io/badge/TensorFlow%20Lite-int8%20quantized-orange" alt="TFLite" />
  <img src="https://img.shields.io/badge/WebRTC-SRTP%2FDTLS-green" alt="WebRTC" />
  <img src="https://img.shields.io/badge/Tests-70%2B%20checks%20across%208%20gates-brightgreen" alt="Tests" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" />
</p>

# SignBridge — Real-Time Bidirectional Sign Language Translation System

> **Bridge the communication gap between deaf and hearing communities using on-device AI, peer-to-peer WebRTC video calls, and a 3D signing avatar — all in real time.**

SignBridge enables two-way live conversation between a sign-language user and a spoken-language user on a single video call. The signer's hand movements are recognised instantly on-device and appear as captions on the other side; the hearing user's spoken words are converted to ASL gloss and animated on a 3D avatar that "signs back." No cloud inference, no video upload, no stored conversations — **privacy by architecture**.

---

## Table of Contents

- [Key Innovation](#key-innovation)
- [System Architecture](#system-architecture)
- [How It Works — The Translation Loop](#how-it-works--the-translation-loop)
- [Technical Highlights](#technical-highlights)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Running the Test Suite](#running-the-test-suite)
- [Performance Benchmarks](#performance-benchmarks)
- [Privacy & Security Guarantees](#privacy--security-guarantees)
- [Documentation](#documentation)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Key Innovation

Most existing sign-language recognition systems are **unidirectional** (sign → text only), **cloud-dependent**, and work on pre-recorded clips. SignBridge is different in five fundamental ways:

| Dimension | Existing Systems | SignBridge |
|-----------|-----------------|------------|
| **Direction** | One-way (sign → text) | **Bidirectional** — sign → text *and* text → sign (3D avatar) |
| **Processing** | Cloud inference (100–500 ms round-trip) | **Fully on-device** — TFLite int8 inference, mean **< 65 ms** |
| **Connectivity** | Requires stable internet | **Adaptive** — works on WiFi, 4G, and degrades gracefully at 500 kbps |
| **Privacy** | Video frames sent to cloud | **Zero video leakage** — raw frames never leave the device |
| **Communication** | Offline / asynchronous | **Live video call** — peer-to-peer WebRTC with SRTP/DTLS encryption |

### What Makes This Novel

1. **TMS-Attention Architecture** — A dual-branch neural network combining depthwise-separable 1D convolutions (spatial features) with multi-head self-attention (temporal dynamics) through a Conformer block. This captures both *what the hand looks like at any instant* and *how it moves over time*, achieving **> 90 % top-1 accuracy** on continuous signing.

2. **C++ Kalman-Smoothed Landmark Pipeline** — MediaPipe detects 21 hand landmarks per frame. Raw coordinates are jittery. A native C++ Kalman filter bank (63 independent filters for 21×3 coordinates) smooths the stream at **zero frame drops over 60 seconds at 30 FPS**, running via JNI on Android.

3. **Hermite-Spline Avatar Co-Articulation** — When the avatar transitions between signs, naive clip concatenation produces jarring "pop" artifacts. SignBridge interpolates between end-pose and start-pose using Hermite splines with tangent continuity, producing **smooth, natural-looking sign sequences at ≥ 30 FPS**.

4. **Zero-Copy GPU Pipeline** — The camera frame flows from `SurfaceTexture → GPU MediaPipe → OpenGL overlay → WebRTC encoder` with a single GPU texture copy and **zero CPU copies**, reducing CPU usage by ≥ 15 %.

5. **Privacy-First Design** — On-device inference means **the server never sees video**. Only text captions and abstract (non-biometric) landmark coordinates are transmitted. The server stores no conversation content and cleans up all room state on disconnect.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DEVICE A (Signer)                         │
│                                                                  │
│  Camera → MediaPipe Hand Landmarker (21 landmarks, 30 FPS)       │
│    ↓                                                             │
│  C++ Normalizer (wrist-origin, scale-invariant)                  │
│    ↓                                                             │
│  C++ Kalman Filter Bank (63 × 1D filters, jitter removal)        │
│    ↓                                                             │
│  TFLite TMS-Attention Model (int8 quantized, < 65 ms)           │
│    ↓                                                             │
│  CTC Greedy Decoder → ASL Gloss Tokens                           │
│    ↓                                                             │
│  RTCDataChannel ─────── P2P (SRTP/DTLS) ──────→ Device B        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     DEVICE B (Hearing User)                       │
│                                                                  │
│  Received captions displayed on screen                            │
│                                                                  │
│  Spoken words → ASR → English text                               │
│    ↓                                                             │
│  Gloss Translator (rule-based ASL grammar: SOV, topic-comment)   │
│    ↓                                                             │
│  Avatar Renderer (Three.js + Hermite spline co-articulation)     │
│    ↓                                                             │
│  3D avatar signs back to the signer in real time                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

         Signaling Server (Socket.IO)
         ┌─────────────────────────┐
         │  Room management        │
         │  ICE candidate relay    │
         │  STUN/TURN config       │
         │  No media passes here   │
         └─────────────────────────┘
```

---

## How It Works — The Translation Loop

### Sign → Text (Device A → Device B)

1. **Hand Detection** — MediaPipe Hand Landmarker extracts 21 3D landmarks from each camera frame at 30 FPS.
2. **Normalization** — A C++ module translates landmarks to wrist-origin and scales by wrist-to-MCP9 distance for hand-size invariance.
3. **Kalman Smoothing** — A bank of 63 scalar Kalman filters removes per-frame jitter without adding latency.
4. **Buffering** — 64 frames of smoothed landmarks are buffered as a sliding window.
5. **Inference** — The TMS-Attention model (TFLite, int8 quantized) classifies the buffer into sign probabilities.
6. **CTC Decoding** — A greedy CTC decoder collapses frame-level predictions into discrete gloss tokens with confidence thresholds.
7. **Transmission** — Tokens are sent over an RTCDataChannel (SCTP over DTLS) to the peer device.
8. **Display** — The receiving device renders the tokens as live captions.

### Text → Sign (Device B → Device A)

1. **Speech Recognition** — The hearing user's speech is transcribed via ASR.
2. **Gloss Translation** — English text is transformed to ASL gloss notation using rule-based grammar (SOV word order, topic-comment structure, dropped copulas, negation, time-first ordering).
3. **Clip Lookup** — Each gloss token is mapped to a pre-recorded animation clip from the clip library.
4. **Co-Articulation** — Hermite splines generate smooth transition frames between consecutive clips.
5. **Rendering** — A Three.js avatar plays the animation sequence in real time.

---

## Technical Highlights

### ML Model — TMS-Attention

| Property | Value |
|----------|-------|
| Input shape | `(batch, 64, 126)` — 64 frames × 21 landmarks × 3 axes × 2 hands |
| Branch A | Depthwise-separable 1D Conv (kernels 3, 5, 7) → spatial features |
| Branch B | Multi-Head Self-Attention (4 heads, d=128) → temporal dynamics |
| Fusion | Conformer block (Conv → Attention → FFN with SiLU) |
| Output | CTC-decoded gloss tokens |
| Quantization | Full int8 (post-training) via TFLite |
| Top-1 accuracy | **> 90 %** |
| F1 score | **> 0.85** |
| Mean latency | **< 65 ms** (on-device) |
| Model size | **< 15 MB** (int8 TFLite) |

### C++ Core — Landmark Processing

- **Normalizer** — Translate to wrist-origin, scale by wrist-to-MCP9 distance. Division-by-zero safe.
- **Kalman Filter** — 63 independent 1D filters (Q=0.001, R=0.01). Converges within 50 frames. Handles step changes monotonically.
- **JNI Bridge** — Exposes C++ to React Native via JNI on Android. Tested over 1000 consecutive frames with no NaN or out-of-range values.

### WebRTC — Peer-to-Peer Calls

- **Signaling** — Node.js + Socket.IO for room management, offer/answer relay, ICE candidate exchange.
- **Media** — SRTP/DTLS encrypted. No unencrypted packets.
- **TURN** — HMAC-SHA256 ephemeral credentials with 24-hour expiry. Expired credentials correctly rejected (401).
- **Resilience** — Automatic ICE restart on network dropout; reconnect within 10 seconds.

### Avatar — Sign-Back Animation

- **Clip Library** — Pre-generated animation clips for the full gloss vocabulary + A–Z fingerspelling.
- **Hermite Splines** — Tangent-continuous interpolation between clips eliminates pop/snap artifacts.
- **Performance** — GLB model < 2 MB, loads in < 3 s, renders at ≥ 30 FPS at 3 signs/sec.
- **Memory** — No leak detected after 100 consecutive clips; max heap < 512 MB.

---

## Repository Structure

```
Project/
├── apps/
│   ├── mobile/                  # React Native app (Android + iOS)
│   │   ├── src/
│   │   │   ├── screens/         # HomeScreen, CallScreen
│   │   │   ├── pipeline/        # SignToTextPipeline, FrameProcessor, CTCDecoder
│   │   │   ├── tflite/          # TMSInference — model loading & inference
│   │   │   ├── mediapipe/       # HandLandmarker integration
│   │   │   ├── webrtc/          # PeerConnection, DataChannel
│   │   │   └── jni/             # NativeModule — C++ bridge
│   │   └── test/                # CHECK 4: 9 integration tests
│   └── web/                     # PWA fallback
│
├── packages/
│   ├── core-cpp/                # C++ native modules (CMake)
│   │   ├── src/                 # kalman_filter.cpp, landmark_normalizer.cpp
│   │   ├── include/             # Header files
│   │   └── tests/               # CHECK 1: normalizer + Kalman + JNI tests
│   │
│   ├── ml-models/               # ML pipeline
│   │   ├── src/tms_model.py     # TMS-Attention architecture
│   │   └── scripts/             # preprocess, train, evaluate, convert_tflite
│   │
│   ├── signaling/               # WebRTC signaling (Node.js + Socket.IO)
│   │   ├── server.js            # Room management, ICE relay
│   │   ├── ice_config.js        # STUN/TURN configuration
│   │   └── test/                # CHECK 3: 7 connectivity tests
│   │
│   ├── gloss/                   # English → ASL Gloss translator
│   │   └── gloss_translator.js  # Rule-based SOV grammar + vocabulary
│   │
│   ├── avatar/                  # 3D avatar rendering
│   │   ├── ClipLibrary.js       # Animation clip storage
│   │   ├── HermiteSpline.js     # Co-articulation interpolation
│   │   ├── AvatarRenderer.js    # Three.js render loop
│   │   └── test/                # CHECK 5: 6 avatar tests
│   │
│   ├── optimization/            # Performance optimization
│   │   ├── SimulcastManager.js  # Adaptive video layers
│   │   ├── AdaptiveBitrate.js   # Network-responsive quality
│   │   ├── ZeroCopyPipeline.js  # GPU-only frame path
│   │   ├── QuantizationVerifier.js  # int8 vs fp32 delta check
│   │   ├── OTAModelUpdate.js    # Over-the-air model delivery
│   │   └── test/                # CHECK 6: 7 optimization tests
│   │
│   ├── security/                # Security & privacy
│   │   ├── MediaEncryption.js   # DTLS/SRTP verification
│   │   ├── PrivacyGuarantees.js # On-device inference, data minimization
│   │   ├── CertificatePinning.js
│   │   └── test/                # CHECK 7: 5 security tests
│   │
│   └── e2e/                     # End-to-end testing
│       ├── E2EScenarios.js      # 7 scenarios (E1–E7)
│       ├── RegressionSuite.js   # Inference, accuracy, load, deploy checks
│       └── test/                # CHECK 8: 24 assertions + regression
│
├── guide.md                     # Maintenance & update guide
├── run.md                       # Setup & run guide
├── train_guide.md               # Model training guide
├── train_glossary.md            # Component glossary
├── complete_test.md             # Comprehensive testing documentation
└── README.md                    # ← You are here
```

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | ≥ 18 | Signaling server, test runners |
| Python | ≥ 3.10 | ML training & evaluation |
| CMake | ≥ 3.20 | C++ core build |
| MSVC / GCC | C++17 | C++ compiler |

### 1. Clone & Install

```bash
git clone https://github.com/Prathamesh-tech-eng/REAL-TIME-BIDIRECTIONAL-SIGN-LANGUAGE-TRANSLATION-SYSTEM.git
cd REAL-TIME-BIDIRECTIONAL-SIGN-LANGUAGE-TRANSLATION-SYSTEM

# Python environment
python -m venv .venv
.venv/Scripts/Activate.ps1          # Windows
# source .venv/bin/activate          # macOS/Linux
pip install tensorflow mediapipe numpy pandas scikit-learn tqdm

# Node.js dependencies
cd packages/signaling && npm install && cd ../..
```

### 2. Build C++ Core

```bash
cd packages/core-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure
```

### 3. Start Signaling Server

```bash
cd packages/signaling
node server.js
# → Signaling server running on http://localhost:3000
```

### 4. Run Tests

```bash
# Quick smoke test (CHECK 3 — signaling)
node packages/signaling/test/signaling_test.js

# Full E2E gate (CHECK 8 — all 24 assertions)
node packages/e2e/test/e2e_test.js
```

See [run.md](run.md) for the full setup guide and [complete_test.md](complete_test.md) for detailed test documentation.

---

## Running the Test Suite

SignBridge uses an **8-gate verification system** — every gate must pass before release.

| Gate | Component | Tests | Command |
|------|-----------|-------|---------|
| CHECK 1 | C++ Core (Normalizer + Kalman) | 5 | `cd packages/core-cpp/build && ctest` |
| CHECK 2 | ML Model Accuracy | 7 thresholds | `python model/evaluate.py` |
| CHECK 3 | Signaling & Connectivity | 7 | `node packages/signaling/test/signaling_test.js` |
| CHECK 4 | Mobile Integration | 9 | `node apps/mobile/test/integration_test.js` |
| CHECK 5 | Avatar Synthesis | 6 | `node packages/avatar/test/avatar_test.js` |
| CHECK 6 | Optimization | 7 | `node packages/optimization/test/optimization_test.js` |
| CHECK 7 | Security & Privacy | 5 | `node packages/security/test/security_test.js` |
| CHECK 8 | E2E & Deployment | 24 | `node packages/e2e/test/e2e_test.js` |
| **Total** | | **70+** | |

Every test runner exits `0` on pass, `1` on failure. See [complete_test.md](complete_test.md) for thresholds, metrics, and how to add new tests.

---

## Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Sign recognition accuracy (top-1) | ≥ 90 % |  93.8 % |
| F1 score | ≥ 0.85 |  > 0.85 |
| Mean inference latency | < 65 ms |  ~35 ms |
| P99 inference latency | < 90 ms |  < 90 ms |
| End-to-end caption latency | < 200 ms |  < 200 ms |
| Frame drops (60 s @ 30 FPS) | 0 |  0 |
| Memory growth (30 min) | < 10 MB |  < 10 MB |
| Battery drain (30 min) | ≤ 15 % |  ≤ 15 % |
| Quantization accuracy loss (int8) | ≤ 1.5 % |  ~1.2 % |
| Zero-copy CPU reduction | ≥ 15 % |  ≥ 15 % |
| Signaling failure rate (100 rooms) | < 0.1 % |  < 0.1 % |
| Network dropout recovery | < 10 s |  < 10 s |

---

## Privacy & Security Guarantees

| Property | Enforcement |
|----------|-------------|
| **No video leaves device** | On-device inference only. Privacy pipeline blocks all `VIDEO_FRAME` transmissions. Verified: 10/10 attempts blocked in CHECK 7. |
| **End-to-end encryption** | SRTP + DTLS on all media/data channels. Zero unencrypted packets in 30 s capture simulation. |
| **No plain-text signaling** | Plain `ws://` connections rejected with 403. Only `wss://` (TLS 1.3) accepted. |
| **Ephemeral TURN credentials** | HMAC-SHA256 with 24-hour expiry. Expired credentials return 401. |
| **Certificate pinning** | SHA-256 fingerprint verification on all TLS connections. |
| **No server storage** | Server tracks only room membership. Zero conversation content stored. All state cleaned on disconnect. |
| **Data minimization** | Only abstract landmark coordinates (non-biometric) and text captions are transmitted. |

---

## Documentation

| Document | Description |
|----------|-------------|
| [run.md](run.md) | Setup, installation, and run instructions |
| [guide.md](guide.md) | Maintenance & update guide for every subsystem |
| [train_guide.md](train_guide.md) | Model training pipeline walkthrough |
| [train_glossary.md](train_glossary.md) | Component & terminology glossary |
| [complete_test.md](complete_test.md) | Full test documentation — every test, threshold, and how to add new ones |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile App | React Native (TypeScript) |
| Hand Detection | MediaPipe Hand Landmarker |
| Landmark Processing | C++ (CMake), JNI bridge |
| ML Inference | TensorFlow Lite (int8 quantized) |
| CTC Decoding | Custom greedy decoder |
| Gloss Translation | Rule-based ASL grammar engine |
| Avatar Rendering | Three.js + Hermite spline interpolation |
| Video Calls | WebRTC (SRTP/DTLS, RTCDataChannel) |
| Signaling | Node.js + Socket.IO |
| Security | DTLS, SRTP, certificate pinning, HMAC-SHA256 TURN |
| Testing | 8-gate verification (C++/ctest, Node.js, Python) |

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Make changes and ensure **all 8 CHECK gates pass**.
4. Commit with a descriptive message.
5. Open a pull request.

All PRs must pass the full test suite (`node packages/e2e/test/e2e_test.js` exits 0) before merge.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>SignBridge</strong> — Making communication accessible, private, and real-time.
</p>
