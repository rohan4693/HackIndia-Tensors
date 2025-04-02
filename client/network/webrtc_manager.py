import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay
from typing import Dict, List, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebRTCManager:
    def __init__(self):
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.relay = MediaRelay()
        self.on_message_callbacks: List[Callable] = []
        
    async def create_peer_connection(self, peer_id: str) -> RTCPeerConnection:
        """Create a new peer connection."""
        pc = RTCPeerConnection()
        self.peer_connections[peer_id] = pc
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                self._handle_message(peer_id, message)
        
        return pc
    
    def _handle_message(self, peer_id: str, message: str):
        """Handle incoming messages from peers."""
        try:
            data = json.loads(message)
            for callback in self.on_message_callbacks:
                callback(peer_id, data)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from peer {peer_id}")
    
    def add_message_callback(self, callback: Callable):
        """Add callback for handling incoming messages."""
        self.on_message_callbacks.append(callback)
    
    async def create_offer(self, peer_id: str) -> dict:
        """Create and return an offer for a peer connection."""
        pc = await self.create_peer_connection(peer_id)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    
    async def handle_answer(self, peer_id: str, answer: dict):
        """Handle incoming answer from peer."""
        pc = self.peer_connections.get(peer_id)
        if pc:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
    
    async def handle_ice_candidate(self, peer_id: str, candidate: dict):
        """Handle incoming ICE candidate from peer."""
        pc = self.peer_connections.get(peer_id)
        if pc:
            await pc.addIceCandidate(
                RTCIceCandidate(
                    sdpMid=candidate["sdpMid"],
                    sdpMLineIndex=candidate["sdpMLineIndex"],
                    candidate=candidate["candidate"]
                )
            )
    
    async def send_message(self, peer_id: str, message: dict):
        """Send message to a specific peer."""
        pc = self.peer_connections.get(peer_id)
        if pc and pc.sctp:
            channel = pc.sctp.channels[0]
            await channel.send(json.dumps(message))
    
    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected peers."""
        for peer_id in self.peer_connections:
            await self.send_message(peer_id, message)
    
    async def close_connection(self, peer_id: str):
        """Close connection with a specific peer."""
        pc = self.peer_connections.pop(peer_id, None)
        if pc:
            await pc.close()
    
    async def close_all_connections(self):
        """Close all peer connections."""
        for peer_id in list(self.peer_connections.keys()):
            await self.close_connection(peer_id)

class SignalingServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.webrtc_manager = WebRTCManager()
        self.connected_peers: Dict[str, dict] = {}
    
    async def start(self):
        """Start the signaling server."""
        server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        logger.info(f"Signaling server running on {self.host}:{self.port}")
        await server.serve_forever()
    
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle new client connections."""
        peer_id = id(writer)
        self.connected_peers[peer_id] = {"writer": writer}
        
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                    
                message = json.loads(data.decode())
                await self._handle_message(peer_id, message)
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            await self._cleanup_connection(peer_id)
    
    async def _handle_message(self, peer_id: str, message: dict):
        """Handle incoming messages from clients."""
        message_type = message.get("type")
        if message_type == "offer":
            offer = await self.webrtc_manager.create_offer(peer_id)
            await self._send_message(peer_id, {
                "type": "offer",
                "offer": offer
            })
        elif message_type == "answer":
            await self.webrtc_manager.handle_answer(peer_id, message["answer"])
        elif message_type == "ice_candidate":
            await self.webrtc_manager.handle_ice_candidate(peer_id, message["candidate"])
    
    async def _send_message(self, peer_id: str, message: dict):
        """Send message to a specific client."""
        writer = self.connected_peers[peer_id]["writer"]
        writer.write(json.dumps(message).encode())
        await writer.drain()
    
    async def _cleanup_connection(self, peer_id: str):
        """Clean up when a client disconnects."""
        writer = self.connected_peers[peer_id]["writer"]
        writer.close()
        await writer.wait_closed()
        del self.connected_peers[peer_id]
        await self.webrtc_manager.close_connection(peer_id) 