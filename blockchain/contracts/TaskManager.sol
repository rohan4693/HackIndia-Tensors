// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract TaskManager is ReentrancyGuard, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _taskIds;

    struct Task {
        address client;
        string taskHash;
        uint256 reward;
        uint256 deadline;
        bool completed;
        bool verified;
        address worker;
        string resultHash;
    }

    struct Worker {
        uint256 completedTasks;
        uint256 totalEarnings;
        bool isActive;
    }

    mapping(uint256 => Task) public tasks;
    mapping(address => Worker) public workers;
    mapping(address => uint256) public balances;

    event TaskCreated(uint256 taskId, address client, uint256 reward);
    event TaskCompleted(uint256 taskId, address worker);
    event TaskVerified(uint256 taskId);
    event PaymentReleased(uint256 taskId, address worker, uint256 amount);

    constructor() {}

    function createTask(
        string memory _taskHash,
        uint256 _reward,
        uint256 _deadline
    ) external payable returns (uint256) {
        require(msg.value >= _reward, "Insufficient payment");
        _taskIds.increment();
        uint256 newTaskId = _taskIds.current();

        tasks[newTaskId] = Task({
            client: msg.sender,
            taskHash: _taskHash,
            reward: _reward,
            deadline: block.timestamp + _deadline,
            completed: false,
            verified: false,
            worker: address(0),
            resultHash: ""
        });

        emit TaskCreated(newTaskId, msg.sender, _reward);
        return newTaskId;
    }

    function acceptTask(uint256 _taskId) external nonReentrant {
        Task storage task = tasks[_taskId];
        require(!task.completed, "Task already completed");
        require(task.worker == address(0), "Task already assigned");
        require(workers[msg.sender].isActive, "Worker not active");

        task.worker = msg.sender;
    }

    function submitResult(uint256 _taskId, string memory _resultHash) external nonReentrant {
        Task storage task = tasks[_taskId];
        require(task.worker == msg.sender, "Not authorized");
        require(!task.completed, "Task already completed");
        require(block.timestamp <= task.deadline, "Task deadline passed");

        task.resultHash = _resultHash;
        task.completed = true;
        emit TaskCompleted(_taskId, msg.sender);
    }

    function verifyTask(uint256 _taskId) external onlyOwner {
        Task storage task = tasks[_taskId];
        require(task.completed, "Task not completed");
        require(!task.verified, "Task already verified");

        task.verified = true;
        emit TaskVerified(_taskId);

        // Release payment to worker
        uint256 reward = task.reward;
        balances[task.worker] += reward;
        workers[task.worker].completedTasks += 1;
        workers[task.worker].totalEarnings += reward;
        emit PaymentReleased(_taskId, task.worker, reward);
    }

    function withdraw() external nonReentrant {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance to withdraw");
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }

    function registerWorker() external {
        require(!workers[msg.sender].isActive, "Already registered");
        workers[msg.sender] = Worker({
            completedTasks: 0,
            totalEarnings: 0,
            isActive: true
        });
    }
} 