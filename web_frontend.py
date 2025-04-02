import asyncio
import os
import sys
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional
import json
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Add project root to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from client.main import DecentralizedAIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Decentralized AI Training Dashboard")

# Setup templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global client and task storage
client = None
tasks = {}
training_in_progress = False

@app.on_event("startup")
async def startup_event():
    """Initialize the decentralized AI client when the server starts."""
    global client
    client = DecentralizedAIClient()
    await client.start()
    logger.info("Decentralized AI client started")
    
    # Start background task for status updates
    asyncio.create_task(periodic_status_update())

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the server shuts down."""
    global client
    if client:
        await client.webrtc.close_all_connections()
        logger.info("Closed all client connections")

async def periodic_status_update():
    """Update task statuses periodically."""
    global tasks, client
    while True:
        try:
            for task_id in list(tasks.keys()):
                status = client.get_task_status(task_id)
                if task_id in tasks:  # Check again in case task was deleted
                    tasks[task_id]["status"] = status["status"]
                    tasks[task_id]["last_updated"] = str(datetime.now())
                    
                    if "result" in status and status["result"]:
                        if "training_metrics" in status["result"]:
                            tasks[task_id]["metrics"] = status["result"]["training_metrics"]
            
            # Check for worker status
            worker_count = len(client.connected_workers)
            logger.info(f"Connected workers: {worker_count}, Active tasks: {len(tasks)}")
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
        
        await asyncio.sleep(2)  # Update every 2 seconds

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse(
        "dashboard.html", 
        {"request": request, "tasks": tasks}
    )

@app.post("/create_task")
async def create_task(
    background_tasks: BackgroundTasks,
    input_size: int = Form(10),
    hidden_size: int = Form(20),
    output_size: int = Form(10),
    data_size: int = Form(100),
    batch_size: int = Form(32),
    epochs: int = Form(5),
    learning_rate: float = Form(0.001)
):
    """Create a new training task with the specified parameters."""
    global client, tasks, training_in_progress
    
    if training_in_progress:
        return {"success": False, "error": "Training already in progress"}
    
    training_in_progress = True
    
    try:
        # Create model based on parameters
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        
        # Generate random data for training
        data = np.random.randn(data_size, input_size).astype(np.float32)
        
        # Create the task
        task_id = await client.distribute_training_task(
            model=model,
            data=data,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Store task information
        tasks[task_id] = {
            "parameters": {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
                "data_size": data_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate
            },
            "status": "pending",
            "created_at": str(datetime.now()),
            "last_updated": str(datetime.now())
        }
        
        logger.info(f"Created task: {task_id}")
        training_in_progress = False
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        training_in_progress = False
        return {"success": False, "error": str(e)}

@app.get("/tasks")
async def get_tasks():
    """Return all tasks and their status."""
    return {"tasks": tasks}

@app.get("/task/{task_id}")
async def get_task(task_id: str):
    """Get details for a specific task."""
    if task_id not in tasks:
        return {"success": False, "error": "Task not found"}
    
    return {"success": True, "task": tasks[task_id]}

@app.post("/aggregate_tasks")
async def aggregate_tasks(task_ids: List[str]):
    """Aggregate results from multiple completed tasks."""
    global client, tasks
    
    try:
        # Filter for completed tasks
        completed_tasks = [
            task_id for task_id in task_ids 
            if task_id in tasks and tasks[task_id]["status"] == "completed"
        ]
        
        if not completed_tasks:
            return {"success": False, "error": "No completed tasks to aggregate"}
        
        # Aggregate the models
        final_model = await client.aggregate_results(completed_tasks)
        
        # Generate a unique ID for the aggregated model
        aggregated_id = f"aggregated_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Store information about the aggregation
        tasks[aggregated_id] = {
            "type": "aggregated",
            "source_tasks": completed_tasks,
            "created_at": str(datetime.now()),
            "status": "completed"
        }
        
        return {
            "success": True,
            "aggregated_id": aggregated_id,
            "message": f"Successfully aggregated {len(completed_tasks)} tasks"
        }
        
    except Exception as e:
        logger.error(f"Error aggregating tasks: {e}")
        return {"success": False, "error": str(e)}

@app.get("/workers")
async def get_workers():
    """Get information about connected workers."""
    global client
    return {"workers": client.connected_workers}

@app.get("/test_model/{task_id}")
async def test_model(task_id: str, samples: int = 5):
    """Test a trained model with random input data."""
    global client, tasks
    
    if task_id not in tasks:
        return {"success": False, "error": "Task not found"}
    
    if tasks[task_id]["status"] != "completed":
        return {"success": False, "error": "Task not completed yet"}
    
    try:
        # Get task parameters
        params = tasks[task_id]["parameters"]
        input_size = params.get("input_size", 10)
        
        # Create test data
        test_data = np.random.randn(samples, input_size).astype(np.float32)
        test_tensor = torch.tensor(test_data, dtype=torch.float32)
        
        # Get the model
        if task_id.startswith("aggregated_"):
            # For aggregated models, we need to re-aggregate
            source_tasks = tasks[task_id]["source_tasks"]
            model = await client.aggregate_results(source_tasks)
        else:
            # For single tasks, we create a new aggregation
            model = await client.aggregate_results([task_id])
        
        # Run inference
        with torch.no_grad():
            output = model(test_tensor)
        
        # Convert output to list for JSON serialization
        results = output.numpy().tolist()
        
        return {
            "success": True,
            "input": test_data.tolist(),
            "output": results,
            "input_shape": test_data.shape,
            "output_shape": output.shape
        }
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import socket
    import os
    
    # Get local IP address
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    # Get port from environment variable (for deployment)
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "="*50)
    print("Starting server on all interfaces")
    print("="*50)
    print("\nAccess the dashboard using any of these URLs:")
    print(f"• Local access: http://localhost:{port}")
    print(f"• Local access: http://127.0.0.1:{port}")
    print(f"• Network access: http://{local_ip}:{port}")
    print("\nPress CTRL+C to stop the server")
    print("="*50 + "\n")
    
    # Detect environment (local vs deployment)
    if os.environ.get("RENDER") or os.environ.get("DEPLOYMENT"):
        # Production mode - no reload
        uvicorn.run("web_frontend:app", host="0.0.0.0", port=port)
    else:
        # Development mode with reload
        uvicorn.run("web_frontend:app", host="0.0.0.0", port=port, reload=True)