/**
 * App.tsx — Root component for the SignBridge mobile app.
 */

import React, { useState, useCallback } from 'react';
import HomeScreen from './src/screens/HomeScreen';
import CallScreen from './src/screens/CallScreen';

const SIGNALING_URL = 'http://localhost:3000';

type Screen = 'home' | 'call';

const App: React.FC = () => {
  const [screen, setScreen] = useState<Screen>('home');
  const [roomId, setRoomId] = useState<string>('');

  const handleJoinRoom = useCallback((id: string) => {
    setRoomId(id);
    setScreen('call');
  }, []);

  const handleEndCall = useCallback(() => {
    setScreen('home');
    setRoomId('');
  }, []);

  switch (screen) {
    case 'call':
      return (
        <CallScreen
          signalingUrl={SIGNALING_URL}
          roomId={roomId}
          onEndCall={handleEndCall}
        />
      );
    case 'home':
    default:
      return <HomeScreen onJoinRoom={handleJoinRoom} />;
  }
};

export default App;
