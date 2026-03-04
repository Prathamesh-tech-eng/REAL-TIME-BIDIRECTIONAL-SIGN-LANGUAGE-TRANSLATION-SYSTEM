/**
 * ice_config.js
 * SignBridge — ICE Server Configuration (Client-Side Reference)
 *
 * This module provides the ICE server configuration that clients
 * should use when creating RTCPeerConnection.
 *
 * Usage (React Native / Web):
 *   import { getIceServers } from '@signbridge/signaling/ice_config';
 *   const pc = new RTCPeerConnection({ iceServers: getIceServers(turnUsername, turnCredential) });
 */

'use strict';

/**
 * Build ICE server configuration for WebRTC peer connections.
 *
 * @param {string} turnUsername  - Time-limited TURN username from signaling server
 * @param {string} turnCredential - HMAC-SHA1 credential from signaling server
 * @param {string} [turnServer]  - TURN server URL (default from env)
 * @returns {Array<RTCIceServer>}
 */
function getIceServers(turnUsername, turnCredential, turnServer) {
  const servers = [
    // Free public STUN servers
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ];

  // Add TURN server if credentials are provided
  if (turnUsername && turnCredential) {
    servers.push({
      urls: turnServer || 'turn:your-vps-ip:3478',
      username: turnUsername,
      credential: turnCredential,
    });
  }

  return servers;
}

module.exports = { getIceServers };
