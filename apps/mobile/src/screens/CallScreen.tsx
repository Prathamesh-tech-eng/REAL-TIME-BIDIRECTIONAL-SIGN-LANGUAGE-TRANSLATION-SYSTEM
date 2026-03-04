/**
 * CallScreen — Main video call screen with sign language translation.
 *
 * Layout:
 *   ┌────────────────────────┐
 *   │  Remote Video (large)  │
 *   │                        │
 *   │  Caption overlay       │
 *   ├────────┬───────────────┤
 *   │ Local  │  Controls     │
 *   │ Video  │  (end, mute)  │
 *   └────────┴───────────────┘
 *
 * The pipeline runs in the background:
 *   Camera → MediaPipe → normalize → Kalman → TFLite → CTC → caption
 *   Caption → DataChannel → remote peer
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  SafeAreaView,
} from 'react-native';
import { SignToTextPipeline } from '../pipeline/SignToTextPipeline';

interface CallScreenProps {
  signalingUrl: string;
  roomId: string;
  onEndCall?: () => void;
}

const CallScreen: React.FC<CallScreenProps> = ({
  signalingUrl,
  roomId,
  onEndCall,
}) => {
  const [caption, setCaption] = useState<string>('');
  const [connected, setConnected] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const pipelineRef = useRef<SignToTextPipeline | null>(null);

  useEffect(() => {
    const pipeline = new SignToTextPipeline({
      signalingUrl,
      roomId,
      modelPath: './tms_model_int8.tflite',
    });

    pipeline.onCaption((text) => {
      setCaption(text);
    });

    pipelineRef.current = pipeline;
    pipeline.start();
    setConnected(true);

    // FPS monitoring
    const fpsInterval = setInterval(() => {
      const stats = pipeline.getStats();
      setFps(Math.round(stats.fps));
    }, 1000);

    return () => {
      clearInterval(fpsInterval);
      pipeline.stop();
    };
  }, [signalingUrl, roomId]);

  const handleEndCall = useCallback(() => {
    pipelineRef.current?.stop();
    setConnected(false);
    onEndCall?.();
  }, [onEndCall]);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* Remote Video Area */}
      <View style={styles.remoteVideo}>
        <Text style={styles.placeholder}>Remote Video Feed</Text>

        {/* Caption Overlay */}
        {caption ? (
          <View style={styles.captionOverlay}>
            <Text style={styles.captionText}>{caption}</Text>
          </View>
        ) : null}

        {/* FPS Indicator */}
        <View style={styles.fpsIndicator}>
          <Text style={styles.fpsText}>{fps} FPS</Text>
        </View>
      </View>

      {/* Bottom Bar */}
      <View style={styles.bottomBar}>
        {/* Local Video Preview */}
        <View style={styles.localVideo}>
          <Text style={styles.localPlaceholder}>Camera</Text>
        </View>

        {/* Controls */}
        <View style={styles.controls}>
          <View
            style={[
              styles.statusDot,
              { backgroundColor: connected ? '#4CAF50' : '#F44336' },
            ]}
          />
          <Text style={styles.statusText}>
            {connected ? 'Connected' : 'Disconnected'}
          </Text>

          <TouchableOpacity
            style={styles.endCallButton}
            onPress={handleEndCall}
          >
            <Text style={styles.endCallText}>End Call</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  remoteVideo: {
    flex: 3,
    backgroundColor: '#1a1a2e',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  placeholder: {
    color: '#666',
    fontSize: 18,
  },
  captionOverlay: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: 12,
    padding: 12,
  },
  captionText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    textAlign: 'center',
  },
  fpsIndicator: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  fpsText: {
    color: '#4CAF50',
    fontSize: 12,
    fontFamily: 'monospace',
  },
  bottomBar: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: '#16213e',
    padding: 10,
  },
  localVideo: {
    width: 120,
    height: '100%',
    backgroundColor: '#0f3460',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  localPlaceholder: {
    color: '#666',
    fontSize: 12,
  },
  controls: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingLeft: 20,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginBottom: 8,
  },
  statusText: {
    color: '#ccc',
    fontSize: 14,
    marginBottom: 16,
  },
  endCallButton: {
    backgroundColor: '#e74c3c',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 30,
  },
  endCallText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default CallScreen;
