import torch
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
import hashlib
import json
import base64
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingTask:
    task_id: str
    model_config: Dict[str, Any]
    encrypted_data: Any
    batch_size: int
    epochs: int
    learning_rate: float

class GPUWorker:
    def __init__(self, device: Optional[str] = None):
        """Initialize GPU worker with specified device."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized GPU worker with device: {self.device}")
        
        if self.device == 'cuda':
            self.gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(0)
            }
            logger.info(f"GPU Info: {self.gpu_info}")
    
    def verify_gpu_capability(self) -> bool:
        """Verify if GPU is available and has sufficient memory."""
        if self.device != 'cuda':
            logger.warning("No CUDA-capable GPU found")
            return False
            
        try:
            # Test GPU memory allocation
            test_tensor = torch.zeros((1000, 1000), device=self.device)
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"GPU verification failed: {e}")
            return False
    
    def create_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Create a neural network from configuration."""
        input_size = config.get('input_size', 10)
        hidden_size = config.get('hidden_size', 20)
        output_size = config.get('output_size', 10)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        ).to(self.device)
        
        logger.info(f"Created model with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        return model
    
    def process_task(self, task: TrainingTask) -> Dict[str, Any]:
        """Process a training task and return results."""
        try:
            logger.info(f"Processing task {task.task_id}")
            
            # Create model from configuration
            model = self.create_model(task.model_config)
            
            # Decrypt and prepare data
            data = self._prepare_data(task.encrypted_data)
            
            # Training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=task.learning_rate)
            criterion = torch.nn.MSELoss()
            
            for epoch in range(task.epochs):
                model.train()
                total_loss = 0
                
                for batch_idx in range(0, len(data), task.batch_size):
                    batch = data[batch_idx:batch_idx + task.batch_size]
                    batch = torch.tensor(batch, device=self.device, dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    output = model(batch)
                    loss = criterion(output, batch)  # Autoencoder-style training
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / (len(data) / task.batch_size)
                logger.info(f"Epoch {epoch + 1}/{task.epochs}, Loss: {avg_loss:.4f}")
            
            # Generate result hash
            result_hash = self._generate_result_hash(model)
            
            # Save model state
            model_state = self._serialize_model(model)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
            return {
                'task_id': task.task_id,
                'model_state': model_state,
                'result_hash': result_hash,
                'training_metrics': {
                    'final_loss': avg_loss,
                    'epochs_completed': task.epochs
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            raise
    
    def _prepare_data(self, encrypted_data: Any) -> np.ndarray:
        """Prepare encrypted data for training."""
        try:
            # If using our simplified encryption
            if isinstance(encrypted_data, (str, bytes)):
                try:
                    # Try decoding base64
                    decoded = base64.b64decode(encrypted_data)
                    data = pickle.loads(decoded)
                    logger.info(f"Decrypted data with shape {data.shape}")
                    return data
                except:
                    # Fallback to assuming it's already a byte array
                    return np.frombuffer(encrypted_data, dtype=np.float32)
            # For list of encrypted batches
            elif isinstance(encrypted_data, list):
                all_data = []
                for batch in encrypted_data:
                    batch_data = self._prepare_data(batch)
                    all_data.append(batch_data)
                return np.concatenate(all_data)
            # Already decrypted
            elif isinstance(encrypted_data, np.ndarray):
                return encrypted_data
            else:
                logger.warning(f"Unknown data type: {type(encrypted_data)}, attempting to convert")
                return np.array(encrypted_data, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            # Return empty array as fallback
            return np.array([])
    
    def _serialize_model(self, model: torch.nn.Module) -> bytes:
        """Serialize model state to bytes."""
        buffer = pickle.dumps(model.state_dict())
        logger.info(f"Serialized model state: {len(buffer)} bytes")
        return buffer
    
    def _generate_result_hash(self, model: torch.nn.Module) -> str:
        """Generate a hash of the model state for verification."""
        state_dict = model.state_dict()
        state_bytes = pickle.dumps(state_dict)
        hash_value = hashlib.sha256(state_bytes).hexdigest()
        logger.info(f"Generated result hash: {hash_value[:10]}...")
        return hash_value
    
    def verify_result(self, result: Dict[str, Any], task_config: Dict[str, Any]) -> bool:
        """Verify the integrity of a training result."""
        try:
            # Reconstruct model
            model = self.create_model(task_config['model_config'])
            
            # Load model state
            if isinstance(result['model_state'], bytes):
                state_dict = pickle.loads(result['model_state'])
            else:
                state_dict = result['model_state']
                
            model.load_state_dict(state_dict)
            
            # Generate hash of reconstructed model
            computed_hash = self._generate_result_hash(model)
            
            # Compare hashes
            is_valid = computed_hash == result['result_hash']
            logger.info(f"Result verification: {'Success' if is_valid else 'Failed'}")
            return is_valid
        except Exception as e:
            logger.error(f"Result verification failed: {e}")
            return False
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics."""
        if self.device != 'cuda':
            return {'device': 'cpu'}
            
        return {
            'device': self.device,
            'name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0),
            'memory_cached': torch.cuda.memory_reserved(0),
            'utilization': torch.cuda.utilization(0)
        } 