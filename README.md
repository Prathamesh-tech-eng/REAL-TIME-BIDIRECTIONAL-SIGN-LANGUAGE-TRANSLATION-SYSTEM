# SignBridge - Real-Time Bidirectional Sign Language Translation System

## Monorepo Structure

```
signbridge/
  apps/
    mobile/          # React Native app (Android + iOS)
    web/             # PWA fallback
  packages/
    core-cpp/        # C++ Kalman + landmark normalization
    ml-models/       # TFLite models, training scripts
    signaling/       # Node.js WebRTC signaling server
    avatar/          # Three.js avatar + animation
    gloss/           # Text-to-Gloss transformer
  docs/              # Architecture docs
  scripts/           # Build & test automation
```
