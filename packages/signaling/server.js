/**
 * server.js
 * SignBridge — WebRTC Signaling Server
 *
 * Stack: Node.js 20 + socket.io 4.x + Express 4.x
 * Purpose: Handles WebRTC offer/answer/ICE exchange.
 *          Stateless — no media passes through this server.
 *          Generates time-limited TURN credentials (HMAC-SHA1, RFC 5766).
 */

'use strict';

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const crypto = require('crypto');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');

// Load env vars (optional .env file for local dev)
try { require('dotenv').config(); } catch (_) { /* dotenv not required */ }

// ============================================================
// Configuration
// ============================================================

const PORT = parseInt(process.env.PORT, 10) || 3000;
const HOST = process.env.HOST || '0.0.0.0';
const TURN_SECRET = process.env.TURN_SECRET || 'dev-secret-replace-in-production';
const TURN_SERVER = process.env.TURN_SERVER || 'turn:127.0.0.1:3478';
const TURN_REALM = process.env.TURN_REALM || 'signbridge.example.com';
const TURN_CREDENTIAL_TTL = parseInt(process.env.TURN_CREDENTIAL_TTL, 10) || 86400; // 24h
const MAX_ROOMS = parseInt(process.env.MAX_ROOMS, 10) || 500;
const MAX_USERS_PER_ROOM = parseInt(process.env.MAX_USERS_PER_ROOM, 10) || 2;

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '*').split(',').map(s => s.trim());

// ============================================================
// Express + HTTP Server
// ============================================================

const app = express();
app.use(cors({ origin: ALLOWED_ORIGINS.includes('*') ? true : ALLOWED_ORIGINS }));
app.use(express.json());

// Health check endpoint
app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    rooms: rooms.size,
    timestamp: new Date().toISOString(),
  });
});

// ICE servers endpoint (REST fallback)
app.get('/ice-servers', (req, res) => {
  const userId = req.query.userId || uuidv4();
  res.json({ iceServers: buildIceServers(userId) });
});

const server = http.createServer(app);

// ============================================================
// Socket.io
// ============================================================

const io = new Server(server, {
  cors: {
    origin: ALLOWED_ORIGINS.includes('*') ? '*' : ALLOWED_ORIGINS,
    methods: ['GET', 'POST'],
  },
  pingInterval: 10000,
  pingTimeout: 5000,
  maxHttpBufferSize: 1e6, // 1 MB max message
});

// ============================================================
// Room State (in-memory, stateless)
// ============================================================

/** @type {Map<string, Set<string>>} roomId → Set of socket IDs */
const rooms = new Map();

/** @type {Map<string, { roomId: string, userId: string }>} socketId → metadata */
const socketMeta = new Map();

// ============================================================
// TURN Credential Generation (RFC 5766 long-term cred)
// ============================================================

/**
 * Generate time-limited TURN credentials using HMAC-SHA1.
 * Format: username = "expiry_timestamp:userId"
 *         credential = HMAC-SHA1(secret, username)
 *
 * @param {string} userId
 * @returns {{ username: string, credential: string, ttl: number }}
 */
function generateTurnCredentials(userId) {
  const expiry = Math.floor(Date.now() / 1000) + TURN_CREDENTIAL_TTL;
  const username = `${expiry}:${userId}`;
  const credential = crypto
    .createHmac('sha1', TURN_SECRET)
    .update(username)
    .digest('base64');

  return { username, credential, ttl: TURN_CREDENTIAL_TTL };
}

/**
 * Build full ICE servers array with TURN credentials.
 * @param {string} userId
 * @returns {Array<Object>}
 */
function buildIceServers(userId) {
  const { username, credential } = generateTurnCredentials(userId);

  return [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    {
      urls: TURN_SERVER,
      username,
      credential,
    },
  ];
}

// ============================================================
// Socket Event Handlers
// ============================================================

io.on('connection', (socket) => {
  const socketId = socket.id;
  console.log(`[connect] ${socketId}`);

  // ----------------------------------------------------------
  // join-room
  // ----------------------------------------------------------
  socket.on('join-room', (roomId, userId, callback) => {
    // Validate
    if (!roomId || typeof roomId !== 'string') {
      return typeof callback === 'function'
        ? callback({ error: 'Invalid roomId' })
        : null;
    }

    userId = userId || uuidv4();

    // Enforce room limit
    if (!rooms.has(roomId) && rooms.size >= MAX_ROOMS) {
      return typeof callback === 'function'
        ? callback({ error: 'Server room limit reached' })
        : null;
    }

    // Enforce per-room user limit
    if (rooms.has(roomId) && rooms.get(roomId).size >= MAX_USERS_PER_ROOM) {
      return typeof callback === 'function'
        ? callback({ error: 'Room is full' })
        : null;
    }

    // Join socket.io room
    socket.join(roomId);

    // Track room membership
    if (!rooms.has(roomId)) {
      rooms.set(roomId, new Set());
    }
    rooms.get(roomId).add(socketId);

    // Track socket metadata
    socketMeta.set(socketId, { roomId, userId });

    // Generate TURN credentials for this user
    const iceServers = buildIceServers(userId);

    console.log(`[join-room] ${userId} → room ${roomId} (${rooms.get(roomId).size} users)`);

    // Notify other users in the room
    socket.to(roomId).emit('user-connected', {
      userId,
      socketId,
      timestamp: Date.now(),
    });

    // Acknowledge join with ICE servers
    if (typeof callback === 'function') {
      callback({
        success: true,
        userId,
        roomId,
        iceServers,
        users: Array.from(rooms.get(roomId)).filter(id => id !== socketId),
      });
    }
  });

  // ----------------------------------------------------------
  // offer (SDP offer relay)
  // ----------------------------------------------------------
  socket.on('offer', (roomId, offer) => {
    if (!roomId || !offer) return;
    console.log(`[offer] ${socketId} → room ${roomId}`);
    socket.to(roomId).emit('offer', {
      offer,
      from: socketId,
      timestamp: Date.now(),
    });
  });

  // ----------------------------------------------------------
  // answer (SDP answer relay)
  // ----------------------------------------------------------
  socket.on('answer', (roomId, answer) => {
    if (!roomId || !answer) return;
    console.log(`[answer] ${socketId} → room ${roomId}`);
    socket.to(roomId).emit('answer', {
      answer,
      from: socketId,
      timestamp: Date.now(),
    });
  });

  // ----------------------------------------------------------
  // ice-candidate (ICE candidate relay)
  // ----------------------------------------------------------
  socket.on('ice-candidate', (roomId, candidate) => {
    if (!roomId || !candidate) return;
    socket.to(roomId).emit('ice-candidate', {
      candidate,
      from: socketId,
    });
  });

  // ----------------------------------------------------------
  // disconnect
  // ----------------------------------------------------------
  socket.on('disconnect', (reason) => {
    console.log(`[disconnect] ${socketId} — ${reason}`);

    const meta = socketMeta.get(socketId);
    if (meta) {
      const { roomId, userId } = meta;

      // Remove from room
      if (rooms.has(roomId)) {
        rooms.get(roomId).delete(socketId);
        if (rooms.get(roomId).size === 0) {
          rooms.delete(roomId); // Clean up empty rooms
        }
      }

      // Notify remaining users
      socket.to(roomId).emit('user-disconnected', {
        userId,
        socketId,
        timestamp: Date.now(),
      });

      socketMeta.delete(socketId);
    }
  });
});

// ============================================================
// Graceful Shutdown
// ============================================================

function shutdown() {
  console.log('\n[shutdown] Closing server...');
  io.close(() => {
    server.close(() => {
      console.log('[shutdown] Done.');
      process.exit(0);
    });
  });
  // Force exit after 5s
  setTimeout(() => process.exit(1), 5000);
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// ============================================================
// Start
// ============================================================

// Only start listening when run directly (not when require()'d by tests)
if (require.main === module) {
  server.listen(PORT, HOST, () => {
    console.log(`[SignBridge Signaling] listening on ${HOST}:${PORT}`);
    console.log(`  TURN server: ${TURN_SERVER}`);
    console.log(`  TURN realm:  ${TURN_REALM}`);
    console.log(`  Max rooms:   ${MAX_ROOMS}`);
    console.log(`  Health:      http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}/health`);
  });
}

// Export for testing
module.exports = { server, io, rooms, socketMeta, generateTurnCredentials, buildIceServers, PORT, HOST };
