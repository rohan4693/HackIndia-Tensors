import asyncio
import numpy as np
import torch
import sys
import os
import logging

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.main import DecentralizedAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Starting decentralized AI training example")
        
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )
        logger.info(f"Created model: {model}")
        
        # Generate some sample data
        data = np.random.randn(100, 10).astype(np.float32)  # Using smaller data for faster processing
        logger.info(f"Generated sample data with shape: {data.shape}")
        
        # Initialize the client
        client = DecentralizedAIClient()
        logger.info("Client initialized")
        
        await client.start()
        logger.info("Client started")
        
        # Distribute training tasks
        task_ids = []
        for i in range(2):  # Create 2 tasks instead of 3 for faster testing
            logger.info(f"Creating task {i+1}")
            task_id = await client.distribute_training_task(
                model=model,
                data=data,
                batch_size=32,
                epochs=2,  # Reducing epochs for faster testing
                learning_rate=0.001
            )
            task_ids.append(task_id)
            logger.info(f"Created task: {task_id}")
        
        # Wait for tasks to complete
        all_completed = False
        max_retries = 10
        retry_count = 0
        
        while not all_completed and retry_count < max_retries:
            all_completed = True
            status_summary = []
            
            for task_id in task_ids:
                status = client.get_task_status(task_id)
                status_summary.append(f"{task_id}: {status['status']}")
                
                if status["status"] == "completed":
                    logger.info(f"Task {task_id} completed successfully")
                elif status["status"] == "error":
                    logger.error(f"Task {task_id} failed: {status.get('error')}")
                    all_completed = False
                else:
                    all_completed = False
                    logger.info(f"Task {task_id} still running...")
            
            logger.info(f"Task status: {', '.join(status_summary)}")
            
            if not all_completed:
                retry_count += 1
                logger.info(f"Waiting for tasks to complete (attempt {retry_count}/{max_retries})...")
                await asyncio.sleep(2)  # Check every 2 seconds instead of 5
        
        if not all_completed:
            logger.warning("Not all tasks completed. Proceeding with available results.")
        
        # Aggregate results from completed tasks
        completed_task_ids = [
            task_id for task_id in task_ids 
            if client.get_task_status(task_id)["status"] == "completed"
        ]
        
        if not completed_task_ids:
            logger.error("No tasks completed successfully.")
            return
        
        logger.info(f"Aggregating results from {len(completed_task_ids)} completed tasks")
        final_model = await client.aggregate_results(completed_task_ids)
        logger.info("Training completed successfully!")
        
        # Test the final model
        test_data = torch.tensor(np.random.randn(5, 10), dtype=torch.float32)
        logger.info(f"Testing model with test data shape: {test_data.shape}")
        
        with torch.no_grad():
            output = final_model(test_data)
            logger.info(f"Model output shape: {output.shape}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Clean up
        if 'client' in locals():
            try:
                await client.webrtc.close_all_connections()
                logger.info("Closed all connections")
            except Exception as e:
                logger.error(f"Error closing connections: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True) 