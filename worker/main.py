import asyncio
import logging
from typing import Dict, Any
import json
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.compute.gpu_worker import GPUWorker, TrainingTask
from client.network.webrtc_manager import WebRTCManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkerNode:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.gpu_worker = GPUWorker()
        self.webrtc = WebRTCManager()
        self.current_tasks: Dict[str, TrainingTask] = {}
        
        # Verify GPU capability
        if not self.gpu_worker.verify_gpu_capability():
            logger.warning("GPU verification failed. Running on CPU.")
        
        # Get initial GPU stats
        self.gpu_stats = self.gpu_worker.get_gpu_stats()
        logger.info(f"Initial GPU stats: {self.gpu_stats}")
    
    async def start(self):
        """Start the worker node."""
        # Set up WebRTC message handling
        self.webrtc.add_message_callback(self._handle_message)
        
        # Start WebRTC connection
        await self._connect_to_network()
        
        # Start periodic status updates
        asyncio.create_task(self._periodic_status_update())
    
    async def _connect_to_network(self):
        """Connect to the P2P network."""
        try:
            # In production, this would connect to a signaling server
            # and establish P2P connections with clients
            logger.info("Connecting to P2P network...")
            # Placeholder for actual connection logic
        except Exception as e:
            logger.error(f"Failed to connect to network: {e}")
    
    async def _handle_message(self, peer_id: str, message: Dict[str, Any]):
        """Handle incoming messages from peers."""
        try:
            message_type = message.get("type")
            
            if message_type == "task":
                await self._handle_task(peer_id, message)
            elif message_type == "status_request":
                await self._send_status(peer_id)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_task(self, peer_id: str, message: Dict[str, Any]):
        """Handle incoming training task."""
        try:
            task_data = message.get("task")
            if not task_data:
                raise ValueError("No task data provided")
            
            # Create training task
            task = TrainingTask(
                task_id=task_data["task_id"],
                model_config=task_data["model_config"],
                encrypted_data=task_data["encrypted_data"],
                batch_size=task_data["batch_size"],
                epochs=task_data["epochs"],
                learning_rate=task_data["learning_rate"]
            )
            
            # Store task
            self.current_tasks[task.task_id] = task
            
            # Process task
            result = self.gpu_worker.process_task(task)
            
            # Send result back
            await self.webrtc.send_message(peer_id, {
                "type": "task_result",
                "task_id": task.task_id,
                "result": result
            })
            
            # Clean up
            del self.current_tasks[task.task_id]
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            # Send error back to client
            await self.webrtc.send_message(peer_id, {
                "type": "task_error",
                "task_id": task_data.get("task_id"),
                "error": str(e)
            })
    
    async def _send_status(self, peer_id: str):
        """Send current worker status to peer."""
        try:
            status = {
                "type": "status",
                "worker_id": self.worker_id,
                "gpu_stats": self.gpu_worker.get_gpu_stats(),
                "active_tasks": len(self.current_tasks)
            }
            await self.webrtc.send_message(peer_id, status)
        except Exception as e:
            logger.error(f"Error sending status: {e}")
    
    async def _periodic_status_update(self):
        """Periodically update and broadcast status."""
        while True:
            try:
                self.gpu_stats = self.gpu_worker.get_gpu_stats()
                await self.webrtc.broadcast_message({
                    "type": "status_update",
                    "worker_id": self.worker_id,
                    "gpu_stats": self.gpu_stats,
                    "active_tasks": len(self.current_tasks)
                })
            except Exception as e:
                logger.error(f"Error in periodic status update: {e}")
            
            await asyncio.sleep(30)  # Update every 30 seconds

async def main():
    """Main entry point for the worker node."""
    # Generate a unique worker ID (in production, this would be more robust)
    import uuid
    worker_id = str(uuid.uuid4())
    
    worker = WorkerNode(worker_id)
    await worker.start()
    
    # Keep the worker running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down worker node...")
        await worker.webrtc.close_all_connections()

if __name__ == "__main__":
    asyncio.run(main()) 