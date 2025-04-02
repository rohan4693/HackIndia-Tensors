// Global variables
let currentTaskId = null;
let statusChart = null;
let workerChart = null;
let tasksData = [];
let workersData = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Set up navigation
    setupNavigation();
    
    // Load initial data
    refreshTasks();
    refreshWorkers();
    
    // Set up event listeners
    document.getElementById('refresh-tasks').addEventListener('click', refreshTasks);
    document.getElementById('refresh-workers').addEventListener('click', refreshWorkers);
    document.getElementById('task-form').addEventListener('submit', createTask);
    document.getElementById('test-model-btn').addEventListener('click', testModel);
    
    // Initialize charts
    initializeCharts();
    
    // Start periodic refresh
    setInterval(refreshTasks, 5000);
    setInterval(refreshWorkers, 10000);
});

// Set up navigation between sections
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            link.classList.add('active');
            
            // Hide all sections
            sections.forEach(section => section.style.display = 'none');
            
            // Show selected section
            const targetId = link.getAttribute('href').substring(1);
            document.getElementById(targetId).style.display = 'block';
        });
    });
}

// Initialize Chart.js charts
function initializeCharts() {
    // Task status chart
    const statusCtx = document.getElementById('status-chart').getContext('2d');
    statusChart = new Chart(statusCtx, {
        type: 'pie',
        data: {
            labels: ['Pending', 'Running', 'Completed', 'Error'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: ['#ffc107', '#17a2b8', '#28a745', '#dc3545']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    // Worker distribution chart
    const workerCtx = document.getElementById('worker-chart').getContext('2d');
    workerChart = new Chart(workerCtx, {
        type: 'bar',
        data: {
            labels: ['Workers'],
            datasets: [
                {
                    label: 'Available',
                    data: [0],
                    backgroundColor: '#28a745'
                },
                {
                    label: 'Busy',
                    data: [0],
                    backgroundColor: '#dc3545'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    stacked: true
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Refresh tasks from the server
async function refreshTasks() {
    try {
        const response = await fetch('/tasks');
        const data = await response.json();
        
        tasksData = Object.entries(data.tasks || {}).map(([id, task]) => ({
            id,
            ...task
        }));
        
        updateTasksTable();
        updateStatusChart();
        
        // Show status message
        showMessage(`Tasks refreshed: ${tasksData.length} tasks found`, 'info');
    } catch (error) {
        console.error('Error refreshing tasks:', error);
        showMessage('Failed to refresh tasks', 'danger');
    }
}

// Refresh workers from the server
async function refreshWorkers() {
    try {
        const response = await fetch('/workers');
        const data = await response.json();
        
        workersData = Object.entries(data.workers || {}).map(([id, worker]) => ({
            id,
            ...worker
        }));
        
        updateWorkersTable();
        updateWorkerChart();
        
        // Show status message
        showMessage(`Workers refreshed: ${workersData.length} workers found`, 'info');
    } catch (error) {
        console.error('Error refreshing workers:', error);
        showMessage('Failed to refresh workers', 'danger');
    }
}

// Update the tasks table with current data
function updateTasksTable() {
    const tableBody = document.getElementById('tasks-table-body');
    tableBody.innerHTML = '';
    
    if (tasksData.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="6" class="text-center">No tasks found</td>';
        tableBody.appendChild(row);
        return;
    }
    
    tasksData.forEach(task => {
        const row = document.createElement('tr');
        
        // Get model size from parameters
        let modelSize = '';
        if (task.parameters) {
            const p = task.parameters;
            modelSize = `${p.input_size}→${p.hidden_size}→${p.output_size}`;
        } else if (task.type === 'aggregated') {
            modelSize = 'Aggregated Model';
        }
        
        // Get data size
        let dataSize = task.parameters ? `${task.parameters.data_size} samples` : 'N/A';
        
        // Set row color based on status
        let statusClass = '';
        let statusText = task.status || 'unknown';
        
        switch (statusText) {
            case 'pending':
                statusClass = 'table-warning';
                statusText = 'Pending';
                break;
            case 'running':
                statusClass = 'table-info';
                statusText = 'Running';
                break;
            case 'completed':
                statusClass = 'table-success';
                statusText = 'Completed';
                break;
            case 'error':
                statusClass = 'table-danger';
                statusText = 'Error';
                break;
        }
        
        row.className = statusClass;
        
        // Create action buttons
        const viewButton = `<button class="btn btn-sm btn-info view-task" data-task-id="${task.id}">View</button>`;
        const testButton = task.status === 'completed' ? 
            `<button class="btn btn-sm btn-success test-task" data-task-id="${task.id}">Test</button>` : '';
        
        row.innerHTML = `
            <td>${task.id}</td>
            <td>${statusText}</td>
            <td>${modelSize}</td>
            <td>${dataSize}</td>
            <td>${task.created_at || 'N/A'}</td>
            <td>${viewButton} ${testButton}</td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Add event listeners to buttons
    document.querySelectorAll('.view-task').forEach(button => {
        button.addEventListener('click', () => viewTaskDetails(button.dataset.taskId));
    });
    
    document.querySelectorAll('.test-task').forEach(button => {
        button.addEventListener('click', () => testModel(button.dataset.taskId));
    });
}

// Update workers table
function updateWorkersTable() {
    const tableBody = document.getElementById('workers-table-body');
    tableBody.innerHTML = '';
    
    if (workersData.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="5" class="text-center">No workers found</td>';
        tableBody.appendChild(row);
        return;
    }
    
    workersData.forEach(worker => {
        const row = document.createElement('tr');
        
        // Format capabilities
        let capabilities = '';
        if (worker.capabilities) {
            const cap = worker.capabilities;
            capabilities = `
                <span class="badge ${cap.gpu ? 'bg-success' : 'bg-secondary'} me-1">GPU: ${cap.gpu ? 'Yes' : 'No'}</span>
                <span class="badge bg-primary me-1">CPU: ${cap.cpu_cores || 'N/A'}</span>
                <span class="badge bg-info">Memory: ${formatMemory(cap.memory || 0)}</span>
            `;
        }
        
        // Determine status
        const isBusy = worker.active_tasks > 0;
        const statusBadge = isBusy ? 
            '<span class="badge bg-warning">Busy</span>' : 
            '<span class="badge bg-success">Available</span>';
        
        row.innerHTML = `
            <td>${worker.id}</td>
            <td>${capabilities}</td>
            <td>${worker.active_tasks || 0}</td>
            <td>${statusBadge}</td>
            <td>
                <button class="btn btn-sm btn-info worker-details" data-worker-id="${worker.id}">View</button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Add event listeners to buttons
    document.querySelectorAll('.worker-details').forEach(button => {
        button.addEventListener('click', () => {
            alert('Worker details feature coming soon');
        });
    });
}

// Update status chart with current task data
function updateStatusChart() {
    if (!statusChart) return;
    
    // Count tasks by status
    const counts = {
        pending: 0,
        running: 0,
        completed: 0,
        error: 0
    };
    
    tasksData.forEach(task => {
        const status = task.status || 'unknown';
        if (counts.hasOwnProperty(status)) {
            counts[status]++;
        }
    });
    
    // Update chart data
    statusChart.data.datasets[0].data = [
        counts.pending,
        counts.running,
        counts.completed,
        counts.error
    ];
    
    statusChart.update();
}

// Update worker chart
function updateWorkerChart() {
    if (!workerChart) return;
    
    // Count available and busy workers
    let available = 0;
    let busy = 0;
    
    workersData.forEach(worker => {
        if ((worker.active_tasks || 0) > 0) {
            busy++;
        } else {
            available++;
        }
    });
    
    // Update chart data
    workerChart.data.datasets[0].data = [available];
    workerChart.data.datasets[1].data = [busy];
    
    workerChart.update();
}

// Create a new task
async function createTask(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    
    try {
        // Disable submit button
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.innerHTML = 'Creating...';
        
        // Submit the form
        const response = await fetch('/create_task', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMessage(`Task created successfully: ${data.task_id}`, 'success');
            form.reset();
            
            // Refresh tasks
            refreshTasks();
            
            // Return to dashboard
            document.querySelector('.nav-link[href="#dashboard"]').click();
        } else {
            showMessage(`Failed to create task: ${data.error}`, 'danger');
        }
    } catch (error) {
        console.error('Error creating task:', error);
        showMessage('Failed to create task due to an error', 'danger');
    } finally {
        // Re-enable submit button
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = false;
        submitButton.innerHTML = 'Create Task';
    }
}

// View task details
async function viewTaskDetails(taskId) {
    try {
        const response = await fetch(`/task/${taskId}`);
        const data = await response.json();
        
        if (!data.success) {
            showMessage(`Failed to fetch task details: ${data.error}`, 'danger');
            return;
        }
        
        const task = data.task;
        currentTaskId = taskId;
        
        // Populate the modal content
        const modalContent = document.getElementById('task-details-content');
        
        let detailsHtml = `
            <div class="mb-3">
                <h6>Task ID</h6>
                <p>${taskId}</p>
            </div>
            <div class="mb-3">
                <h6>Status</h6>
                <p><span class="badge bg-${getStatusColor(task.status)}">${task.status || 'unknown'}</span></p>
            </div>
            <div class="mb-3">
                <h6>Created At</h6>
                <p>${task.created_at || 'N/A'}</p>
            </div>
            <div class="mb-3">
                <h6>Last Updated</h6>
                <p>${task.last_updated || 'N/A'}</p>
            </div>
        `;
        
        // Add parameters if available
        if (task.parameters) {
            const p = task.parameters;
            detailsHtml += `
                <h5 class="mt-4">Model Parameters</h5>
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-3">
                            <h6>Input Size</h6>
                            <p>${p.input_size}</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-3">
                            <h6>Hidden Size</h6>
                            <p>${p.hidden_size}</p>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-3">
                            <h6>Output Size</h6>
                            <p>${p.output_size}</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-3">
                            <h6>Data Size</h6>
                            <p>${p.data_size} samples</p>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-4">
                        <div class="mb-3">
                            <h6>Batch Size</h6>
                            <p>${p.batch_size}</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="mb-3">
                            <h6>Epochs</h6>
                            <p>${p.epochs}</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="mb-3">
                            <h6>Learning Rate</h6>
                            <p>${p.learning_rate}</p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Add metrics if available
        if (task.metrics) {
            const m = task.metrics;
            detailsHtml += `
                <h5 class="mt-4">Training Metrics</h5>
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-3">
                            <h6>Final Loss</h6>
                            <p>${m.final_loss ? m.final_loss.toFixed(4) : 'N/A'}</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-3">
                            <h6>Epochs Completed</h6>
                            <p>${m.epochs_completed || 'N/A'}</p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Add source tasks if it's an aggregated model
        if (task.type === 'aggregated' && task.source_tasks) {
            detailsHtml += `
                <h5 class="mt-4">Source Tasks</h5>
                <ul>
                    ${task.source_tasks.map(id => `<li>${id}</li>`).join('')}
                </ul>
            `;
        }
        
        modalContent.innerHTML = detailsHtml;
        
        // Show or hide the test button based on status
        const testBtn = document.getElementById('test-model-btn');
        if (task.status === 'completed') {
            testBtn.style.display = 'block';
        } else {
            testBtn.style.display = 'none';
        }
        
        // Show the modal
        const modal = new bootstrap.Modal(document.getElementById('task-details-modal'));
        modal.show();
        
    } catch (error) {
        console.error('Error viewing task details:', error);
        showMessage('Failed to view task details', 'danger');
    }
}

// Test a model
async function testModel(taskId = null) {
    // Use current task ID if none is provided
    if (!taskId) {
        taskId = currentTaskId;
    }
    
    if (!taskId) {
        showMessage('No task selected for testing', 'warning');
        return;
    }
    
    try {
        // Show loading
        showMessage('Testing model...', 'info');
        
        // Make request
        const response = await fetch(`/test_model/${taskId}?samples=5`);
        const data = await response.json();
        
        if (!data.success) {
            showMessage(`Failed to test model: ${data.error}`, 'danger');
            return;
        }
        
        // Format input and output
        const input = data.input;
        const output = data.output;
        
        // Create result display
        const resultsContent = document.getElementById('test-results-content');
        
        let resultsHtml = `
            <h5>Model Test Results</h5>
            <p>Input Shape: ${data.input_shape.join(' × ')}</p>
            <p>Output Shape: ${data.output_shape.join(' × ')}</p>
            
            <div class="table-responsive mt-4">
                <table class="table table-sm table-bordered">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Input (first 3 values)</th>
                            <th>Output (first 3 values)</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        for (let i = 0; i < input.length; i++) {
            // Just show first few values to avoid overwhelming the UI
            const inputSample = input[i].slice(0, 3);
            const outputSample = output[i].slice(0, 3);
            
            resultsHtml += `
                <tr>
                    <td>${i + 1}</td>
                    <td>${inputSample.map(v => v.toFixed(4)).join(', ')}...</td>
                    <td>${outputSample.map(v => v.toFixed(4)).join(', ')}...</td>
                </tr>
            `;
        }
        
        resultsHtml += `
                    </tbody>
                </table>
            </div>
        `;
        
        resultsContent.innerHTML = resultsHtml;
        
        // Close details modal if open
        const detailsModal = bootstrap.Modal.getInstance(document.getElementById('task-details-modal'));
        if (detailsModal) {
            detailsModal.hide();
        }
        
        // Show results modal
        const resultsModal = new bootstrap.Modal(document.getElementById('test-results-modal'));
        resultsModal.show();
        
    } catch (error) {
        console.error('Error testing model:', error);
        showMessage('Failed to test model', 'danger');
    }
}

// Helper function to show a message
function showMessage(message, type = 'info') {
    const alertElement = document.getElementById('status-message');
    alertElement.className = `alert alert-${type}`;
    alertElement.textContent = message;
    
    // Scroll to the message
    alertElement.scrollIntoView({ behavior: 'smooth' });
    
    // Auto-hide after a delay for non-error messages
    if (type !== 'danger') {
        setTimeout(() => {
            alertElement.textContent = 'Decentralized AI Training Dashboard';
            alertElement.className = 'alert alert-info';
        }, 5000);
    }
}

// Helper function to get status color
function getStatusColor(status) {
    switch (status) {
        case 'pending': return 'warning';
        case 'running': return 'info';
        case 'completed': return 'success';
        case 'error': return 'danger';
        default: return 'secondary';
    }
}

// Helper function to format memory
function formatMemory(memory) {
    if (memory < 1024) {
        return `${memory} MB`;
    } else {
        return `${(memory / 1024).toFixed(1)} GB`;
    }
} 