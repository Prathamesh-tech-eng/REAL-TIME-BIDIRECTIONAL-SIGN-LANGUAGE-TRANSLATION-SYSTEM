/**
 * DataChannel — RTCDataChannel wrapper for sending/receiving text
 * messages (captions, landmark data) between peers.
 *
 * Supports:
 *   • Ordered, reliable delivery (TCP semantics)
 *   • Message sequencing and loss detection
 *   • Throughput measurement
 *
 * For integration tests, uses the signaling server's socket.io relay
 * as a stand-in for actual RTCDataChannel (both provide ordered
 * delivery of arbitrary JSON payloads).
 */

export interface DataChannelMessage {
  seq: number;
  type: 'caption' | 'landmark' | 'control';
  payload: string;
  timestamp: number;
}

export interface DataChannelStats {
  messagesSent: number;
  messagesReceived: number;
  messagesLost: number;
  outOfOrder: number;
  avgLatencyMs: number;
  maxLatencyMs: number;
}

export type MessageHandler = (msg: DataChannelMessage) => void;

/**
 * DataChannel — manages sequenced, ordered message delivery.
 */
export class DataChannel {
  private sendSeq: number;
  private expectedRecvSeq: number;
  private stats: DataChannelStats;
  private handlers: MessageHandler[];
  private latencies: number[];
  private sendFn: ((data: string) => void) | null;

  constructor() {
    this.sendSeq = 0;
    this.expectedRecvSeq = 0;
    this.stats = {
      messagesSent: 0,
      messagesReceived: 0,
      messagesLost: 0,
      outOfOrder: 0,
      avgLatencyMs: 0,
      maxLatencyMs: 0,
    };
    this.handlers = [];
    this.latencies = [];
    this.sendFn = null;
  }

  /** Set the underlying send function (WebRTC dc.send or socket.emit). */
  setSendFunction(fn: (data: string) => void): void {
    this.sendFn = fn;
  }

  /** Register a message handler. */
  onMessage(handler: MessageHandler): void {
    this.handlers.push(handler);
  }

  /** Send a message with automatic sequencing. */
  send(type: DataChannelMessage['type'], payload: string): void {
    if (!this.sendFn) throw new Error('Send function not set');

    const msg: DataChannelMessage = {
      seq: this.sendSeq++,
      type,
      payload,
      timestamp: Date.now(),
    };

    this.sendFn(JSON.stringify(msg));
    this.stats.messagesSent++;
  }

  /**
   * Process a received message string.
   * Call this from the WebRTC onmessage handler or socket.io event.
   */
  receive(data: string): void {
    const msg: DataChannelMessage = JSON.parse(data);
    const now = Date.now();
    const latency = now - msg.timestamp;

    this.latencies.push(latency);
    this.stats.messagesReceived++;

    // Check ordering
    if (msg.seq < this.expectedRecvSeq) {
      this.stats.outOfOrder++;
    } else if (msg.seq > this.expectedRecvSeq) {
      // Gap detected — count missing messages
      this.stats.messagesLost += msg.seq - this.expectedRecvSeq;
      this.expectedRecvSeq = msg.seq + 1;
    } else {
      this.expectedRecvSeq = msg.seq + 1;
    }

    // Update latency stats
    this.stats.avgLatencyMs =
      this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length;
    this.stats.maxLatencyMs = Math.max(...this.latencies);

    // Emit to handlers
    for (const handler of this.handlers) {
      handler(msg);
    }
  }

  /** Get channel statistics. */
  getStats(): DataChannelStats {
    return { ...this.stats };
  }

  /** Reset state and statistics. */
  reset(): void {
    this.sendSeq = 0;
    this.expectedRecvSeq = 0;
    this.stats = {
      messagesSent: 0,
      messagesReceived: 0,
      messagesLost: 0,
      outOfOrder: 0,
      avgLatencyMs: 0,
      maxLatencyMs: 0,
    };
    this.latencies = [];
  }
}
