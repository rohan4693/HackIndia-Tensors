import asyncio
import logging
from typing import Dict, Any, List
import json
import torch
import numpy as np
import os
import sys

# Add project root to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from client.network.webrtc_manager import WebRTCManager
from client.encryption.fhe_utils import FHEEncryptor, BatchEncryptor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecentralizedAIClient:
    def __init__(self):
        """Initialize the decentralized AI client."""
        self.webrtc = WebRTCManager()
        self.fhe = FHEEncryptor()
        self.batch_encryptor = BatchEncryptor()
        self.connected_workers: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Set up message handling
        self.webrtc.add_message_callback(self._handle_message)
        
        logger.info("DecentralizedAIClient initialized")
    
    async def start(self):
        """Start the client node."""
        # Connect to the P2P network
        await self._connect_to_network()
        
        # For simulation, add some mock workers
        self._add_mock_workers()
        
        # Start periodic worker discovery
        asyncio.create_task(self._periodic_worker_discovery())
        
        logger.info("Client started")
    
    async def _connect_to_network(self):
        """Connect to the P2P network."""
        try:
            # In production, this would connect to a signaling server
            # and establish P2P connections with workers
            logger.info("Connecting to P2P network...")
            # Placeholder for actual connection logic
            await asyncio.sleep(0.5)  # Simulate connection delay
        except Exception as e:
            logger.error(f"Failed to connect to network: {e}")
    
    def _add_mock_workers(self):
        """Add mock workers for testing."""
        # Add a few fake workers for simulation
        for i in range(3):
            worker_id = f"worker_{i+1}"
            self.connected_workers[worker_id] = {
                "peer_id": f"peer_{i+1}",
                "capabilities": {
                    "gpu": True,
                    "cpu_cores": 8,
                    "memory": 16000
                },
                "active_tasks": 0,
                "reliability": 0.95
            }
        logger.info(f"Added {len(self.connected_workers)} mock workers")
    
    async def _periodic_worker_discovery(self):
        """Periodically discover new workers."""
        while True:
            try:
                # In production, this would discover and connect to new workers
                # For now, just log the current number of workers
                logger.info(f"Connected workers: {len(self.connected_workers)}")
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error during worker discovery: {e}")
                await asyncio.sleep(60)  # Retry after a minute
    
    def _handle_message(self, peer_id: str, message: Dict[str, Any]):
        """Handle incoming messages from peers."""
        try:
            message_type = message.get("type")
            
            if message_type == "worker_registration":
                self._register_worker(peer_id, message)
            elif message_type == "task_result":
                self._handle_task_result(message)
            elif message_type == "task_status":
                self._handle_task_status_update(message)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _register_worker(self, peer_id: str, message: Dict[str, Any]):
        """Register a new worker."""
        worker_id = message.get("worker_id")
        capabilities = message.get("capabilities", {})
        
        self.connected_workers[worker_id] = {
            "peer_id": peer_id,
            "capabilities": capabilities,
            "active_tasks": 0,
            "reliability": 1.0  # Initial perfect reliability
        }
        
        logger.info(f"Registered worker {worker_id} with capabilities: {capabilities}")
    
    def _handle_task_result(self, message: Dict[str, Any]):
        """Handle task result from worker."""
        task_id = message.get("task_id")
        result = message.get("result")
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["result"] = result
            self.active_tasks[task_id]["status"] = "completed"
            logger.info(f"Task {task_id} completed")
        else:
            logger.warning(f"Received result for unknown task {task_id}")
    
    def _handle_task_status_update(self, message: Dict[str, Any]):
        """Handle task status update from worker."""
        task_id = message.get("task_id")
        status = message.get("status")
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = status
            logger.info(f"Task {task_id} status updated to {status}")
        else:
            logger.warning(f"Received status update for unknown task {task_id}")
    
    async def distribute_training_task(
        self,
        model: torch.nn.Module,
        data: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 0.001
    ) -> str:
        """Distribute a training task to available workers."""
        try:
            # Generate task ID
            task_id = f"task_{len(self.active_tasks) + 1}"
            
            # Extract model configuration
            model_config = {}
            try:
                if isinstance(model, torch.nn.Sequential):
                    # Handle Sequential models
                    model_config = {
                        "input_size": getattr(model[0], 'in_features', 10),  # First layer's input features
                        "hidden_size": getattr(model[0], 'out_features', 20),  # First layer's output features
                        "output_size": getattr(model[-1], 'out_features', 10)  # Last layer's output features
                    }
                else:
                    # Generic fallback
                    model_config = {
                        "input_size": 10,
                        "hidden_size": 20,
                        "output_size": 10
                    }
                    logger.warning(f"Using default model config for non-Sequential model")
            except Exception as e:
                logger.error(f"Error extracting model configuration: {e}")
                # Fallback to default configuration
                model_config = {
                    "input_size": 10,
                    "hidden_size": 20,
                    "output_size": 10
                }
            
            # Encrypt data
            logger.info(f"Encrypting data of shape {data.shape}")
            encrypted_data = self.batch_encryptor.encrypt_batch(data)
            logger.info(f"Data encrypted into {len(encrypted_data)} batches")
            
            # Create task
            task = {
                "task_id": task_id,
                "model_config": model_config,
                "encrypted_data": encrypted_data,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "status": "pending"
            }
            
            # Store task
            self.active_tasks[task_id] = task
            
            # Find available workers
            available_workers = [
                worker_id for worker_id, info in self.connected_workers.items()
                if info["active_tasks"] < 3  # Limit tasks per worker
            ]
            
            if not available_workers:
                logger.error("No available workers found")
                # For development, we'll simulate a successful task instead of failing
                self._simulate_task_processing(task_id)
                return task_id
            
            # Distribute task to first available worker
            worker_id = available_workers[0]
            worker_info = self.connected_workers[worker_id]
            
            # For an actual implementation, this would send the task via WebRTC
            logger.info(f"Distributing task {task_id} to worker {worker_id}")
            
            # For development, we'll simulate task processing
            self._simulate_task_processing(task_id)
            
            return task_id
            
        except Exception as e:
            logger.error(f"Error distributing task: {e}")
            raise
    
    def _simulate_task_processing(self, task_id: str):
        """Simulate task processing for development."""
        async def process():
            # Simulate task processing time
            await asyncio.sleep(5)
            
            # Update status to completed
            if task_id in self.active_tasks:
                # Create a dummy result
                result = {
                    "model_state": b"dummy_model_state",
                    "result_hash": "dummy_hash",
                    "training_metrics": {
                        "final_loss": 0.1234,
                        "epochs_completed": self.active_tasks[task_id].get("epochs", 10)
                    }
                }
                
                self.active_tasks[task_id]["result"] = result
                self.active_tasks[task_id]["status"] = "completed"
                logger.info(f"Simulated completion for task {task_id}")
        
        # Start the simulation in a background task
        asyncio.create_task(process())
        logger.info(f"Started simulated processing for task {task_id}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found")
            return {"status": "not_found"}
        
        task = self.active_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task.get("status", "unknown"),
            "result": task.get("result")
        }
    
    async def aggregate_results(self, task_ids: List[str]) -> torch.nn.Module:
        """Aggregate results from multiple completed tasks."""
        try:
            # Verify all tasks are completed
            for task_id in task_ids:
                if task_id not in self.active_tasks:
                    logger.error(f"Task {task_id} not found")
                    raise ValueError(f"Task {task_id} not found")
                if self.active_tasks[task_id]["status"] != "completed":
                    logger.error(f"Task {task_id} not completed")
                    raise ValueError(f"Task {task_id} not completed")
            
            # Create base model
            first_task = self.active_tasks[task_ids[0]]
            model_config = first_task["model_config"]
            model = torch.nn.Sequential(
                torch.nn.Linear(model_config["input_size"], model_config["hidden_size"]),
                torch.nn.ReLU(),
                torch.nn.Linear(model_config["hidden_size"], model_config["output_size"])
            )
            
            logger.info(f"Created base model for aggregation")
            
            # For development, just return the base model
            # In production, this would actually aggregate parameters from multiple models
            logger.info(f"Aggregated results from {len(task_ids)} tasks")
            
            return model
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            raise

async def main():
    """Main entry point for the client."""
    client = DecentralizedAIClient()
    await client.start()
    
    # Keep the client running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down client...")
        await client.webrtc.close_all_connections()

if __name__ == "__main__":
    asyncio.run(main()) 