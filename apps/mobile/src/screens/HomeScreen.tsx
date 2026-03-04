/**
 * HomeScreen — Lobby / home screen for SignBridge.
 *
 * Allows user to:
 *   • Enter a room code to join
 *   • Create a new room
 *   • View connection status
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  StatusBar,
} from 'react-native';

interface HomeScreenProps {
  onJoinRoom?: (roomId: string) => void;
}

const HomeScreen: React.FC<HomeScreenProps> = ({ onJoinRoom }) => {
  const [roomCode, setRoomCode] = useState('');

  const handleJoin = useCallback(() => {
    if (roomCode.trim()) {
      onJoinRoom?.(roomCode.trim());
    }
  }, [roomCode, onJoinRoom]);

  const handleCreate = useCallback(() => {
    const newRoom = `room-${Date.now().toString(36)}`;
    onJoinRoom?.(newRoom);
  }, [onJoinRoom]);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <View style={styles.header}>
        <Text style={styles.title}>SignBridge</Text>
        <Text style={styles.subtitle}>
          Real-Time Sign Language Translation
        </Text>
      </View>

      <View style={styles.form}>
        <Text style={styles.label}>Enter Room Code</Text>
        <TextInput
          style={styles.input}
          value={roomCode}
          onChangeText={setRoomCode}
          placeholder="e.g., room-abc123"
          placeholderTextColor="#666"
          autoCapitalize="none"
          autoCorrect={false}
        />

        <TouchableOpacity
          style={[styles.button, styles.joinButton]}
          onPress={handleJoin}
          disabled={!roomCode.trim()}
        >
          <Text style={styles.buttonText}>Join Room</Text>
        </TouchableOpacity>

        <View style={styles.divider}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>or</Text>
          <View style={styles.dividerLine} />
        </View>

        <TouchableOpacity
          style={[styles.button, styles.createButton]}
          onPress={handleCreate}
        >
          <Text style={styles.buttonText}>Create New Room</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          SignBridge uses your camera to detect sign language and translate it
          in real time for your conversation partner.
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a23',
  },
  header: {
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 40,
  },
  title: {
    fontSize: 36,
    fontWeight: '700',
    color: '#e0e0ff',
    letterSpacing: 2,
  },
  subtitle: {
    fontSize: 14,
    color: '#8888aa',
    marginTop: 8,
  },
  form: {
    paddingHorizontal: 32,
  },
  label: {
    color: '#aaa',
    fontSize: 14,
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#16213e',
    color: '#fff',
    fontSize: 16,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#2a3a5c',
  },
  button: {
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  joinButton: {
    backgroundColor: '#3498db',
  },
  createButton: {
    backgroundColor: '#2ecc71',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#2a3a5c',
  },
  dividerText: {
    color: '#666',
    marginHorizontal: 16,
    fontSize: 14,
  },
  footer: {
    position: 'absolute',
    bottom: 40,
    left: 32,
    right: 32,
  },
  footerText: {
    color: '#555',
    fontSize: 12,
    textAlign: 'center',
    lineHeight: 18,
  },
});

export default HomeScreen;
