/**
 * PeerConnection — WebRTC peer connection manager for SignBridge.
 *
 * Handles:
 *   • Creating RTCPeerConnection with ICE servers (STUN + TURN)
 *   • Offer/Answer SDP exchange via the signaling server
 *   • ICE candidate trickle
 *   • Connection state monitoring
 */

export interface ICEServerConfig {
  urls: string | string[];
  username?: string;
  credential?: string;
}

export interface PeerConnectionConfig {
  iceServers: ICEServerConfig[];
  signalingUrl: string;
  roomId: string;
  role: 'caller' | 'callee';
}

/**
 * Connection state events.
 */
export type ConnectionState =
  | 'new'
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'failed'
  | 'closed';

export type StateChangeHandler = (state: ConnectionState) => void;
export type DataHandler = (data: string | ArrayBuffer) => void;

/**
 * Abstract PeerConnection interface.
 * Platform-specific implementations (react-native-webrtc or browser)
 * extend this base class.
 */
export abstract class PeerConnectionBase {
  protected config: PeerConnectionConfig;
  protected state: ConnectionState;
  private stateHandlers: StateChangeHandler[];
  private dataHandlers: DataHandler[];

  constructor(config: PeerConnectionConfig) {
    this.config = config;
    this.state = 'new';
    this.stateHandlers = [];
    this.dataHandlers = [];
  }

  /** Register connection state change listener. */
  onStateChange(handler: StateChangeHandler): void {
    this.stateHandlers.push(handler);
  }

  /** Register incoming data handler. */
  onData(handler: DataHandler): void {
    this.dataHandlers.push(handler);
  }

  /** Get current connection state. */
  getState(): ConnectionState {
    return this.state;
  }

  /** Notify state listeners. */
  protected emitStateChange(state: ConnectionState): void {
    this.state = state;
    for (const handler of this.stateHandlers) {
      handler(state);
    }
  }

  /** Notify data listeners. */
  protected emitData(data: string | ArrayBuffer): void {
    for (const handler of this.dataHandlers) {
      handler(data);
    }
  }

  abstract connect(): Promise<void>;
  abstract sendData(data: string | ArrayBuffer): void;
  abstract close(): void;
}

/**
 * MockPeerConnection — Simulates WebRTC P2P connection for testing.
 * Two MockPeerConnection instances can be linked to simulate a full
 * caller/callee connection without actual WebRTC.
 */
export class MockPeerConnection extends PeerConnectionBase {
  private peer: MockPeerConnection | null = null;
  private connected: boolean = false;

  /** Link two mock peers together. */
  static createPair(
    config1: PeerConnectionConfig,
    config2: PeerConnectionConfig,
  ): [MockPeerConnection, MockPeerConnection] {
    const pc1 = new MockPeerConnection(config1);
    const pc2 = new MockPeerConnection(config2);
    pc1.peer = pc2;
    pc2.peer = pc1;
    return [pc1, pc2];
  }

  async connect(): Promise<void> {
    this.emitStateChange('connecting');

    // Simulate ICE negotiation delay
    await new Promise((resolve) => setTimeout(resolve, 50));

    if (this.peer) {
      this.connected = true;
      this.emitStateChange('connected');
      this.peer.connected = true;
      this.peer.emitStateChange('connected');
    } else {
      this.emitStateChange('failed');
    }
  }

  sendData(data: string | ArrayBuffer): void {
    if (!this.connected || !this.peer) {
      throw new Error('Not connected');
    }
    // Simulate network latency (1-5ms for local)
    setTimeout(() => {
      this.peer!.emitData(data);
    }, 1 + Math.random() * 4);
  }

  close(): void {
    this.connected = false;
    this.emitStateChange('closed');
    if (this.peer) {
      this.peer.connected = false;
      this.peer.emitStateChange('closed');
      this.peer.peer = null;
    }
    this.peer = null;
  }
}
