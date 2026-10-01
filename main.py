import asyncio
import random
import time
import uuid
import logging
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# Set up logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass(order=True)
class Task:
    priority: int
    task_id: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)

class VectorMemory:
    """In-memory semantic vector store for multi-agent memory retrieval."""
    def __init__(self, vector_dim: int = 8):
        self.dim = vector_dim
        self.db: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        """Simulate Deterministic Text Vectorization."""
        random.seed(hash(text))
        return [round(random.uniform(-1.0, 1.0), 4) for _ in range(self.dim)]

    def store(self, key: str, text: str):
        self.db[key] = self.embed(text)

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a ** 2 for a in v1) ** 0.5
        mag2 = sum(b ** 2 for b in v2) ** 0.5
        return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0

    def query(self, text: str, top_k: int = 2) -> List[tuple]:
        target = self.embed(text)
        results = []
        for k, v in self.db.items():
            sim = self.cosine_similarity(target, v)
            results.append((k, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

class MessageBus:
    """Pub/Sub event system for multi-agent synchronization."""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, callback: Callable):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, data: Any):
        if topic in self._subscribers:
            for cb in self._subscribers[topic]:
                if asyncio.iscoroutinefunction(cb):
                    await cb(data)
                else:
                    cb(data)

class Agent:
    """Autonomous Worker Agent execution unit."""
    def __init__(self, name: str, role: str, memory: VectorMemory, bus: MessageBus):
        self.name = name
        self.role = role
        self.memory = memory
        self.bus = bus

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        logging.info(f"Agent [{self.name}] started task {task.task_id} with priority {task.priority}")
        
        # Simulate agent execution work & context retrieval
        query_str = task.payload.get("query", "default operational command")
        relevant_memories = self.memory.query(query_str, top_k=1)
        
        await asyncio.sleep(random.uniform(0.3, 0.8)) # Simulated execution delay
        
        # Fail transiently to demonstrate queue fault tolerance
        if random.random() < 0.2 and task.retries < task.max_retries:
            raise RuntimeError("Transient node processing failure.")

        result = {
            "task_id": task.task_id,
            "agent": self.name,
            "role": self.role,
            "status": "SUCCESS",
            "context": relevant_memories,
            "processed_at": time.time()
        }
        
        # Store execution step to shared memory & notify bus
        self.memory.store(task.task_id, str(result))
        await self.bus.publish("agent:completed", result)
        return result

class OrchestratorEngine:
    """Distributed Orchestration Engine with Priority Queue Management."""
    def __init__(self, num_workers: int = 3):
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.memory = VectorMemory()
        self.bus = MessageBus()
        self.workers: List[asyncio.Task] = []
        self.num_workers = num_workers
        self.agents = [
            Agent(f"Agent-0{i+1}", role, self.memory, self.bus)
            for i, role in enumerate(["DataIngestion", "NLP-Processor", "SecurityAuditor"])
        ]
        self.completed_tasks: Dict[str, TaskStatus] = {}

    async def enqueue_task(self, payload: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM):
        task_id = f"TASK-{str(uuid.uuid4())[:8]}"
        # Priority Queue uses lowest number = highest priority; invert TaskPriority
        priority_val = 5 - priority.value
        task = Task(priority=priority_val, task_id=task_id, payload=payload)
        await self.task_queue.put(task)
        logging.info(f"Enqueued {task_id} with priority {priority.name}")
        return task_id

    async def _worker_loop(self, worker_id: int):
        while True:
            task: Task = await self.task_queue.get()
            task.status = TaskStatus.PROCESSING
            agent = self.agents[worker_id % len(self.agents)]
            
            try:
                await agent.execute_task(task)
                task.status = TaskStatus.COMPLETED
                self.completed_tasks[task.task_id] = TaskStatus.COMPLETED
                logging.info(f"Worker-{worker_id} successfully finished task {task.task_id}")
            except Exception as e:
                task.retries += 1
                logging.warning(f"Task {task.task_id} failed on retry {task.retries}/{task.max_retries}: {e}")
                if task.retries <= task.max_retries:
                    task.status = TaskStatus.PENDING
                    await self.task_queue.put(task)
                else:
                    task.status = TaskStatus.FAILED
                    self.completed_tasks[task.task_id] = TaskStatus.FAILED
                    logging.error(f"Task {task.task_id} Permanently FAILED.")
            finally:
                self.task_queue.task_done()

    async def start(self):
        logging.info("Starting Multi-Agent Distributed Engine...")
        
        # Populate initial memory store
        self.memory.store("KNOWLEDGE-01", "Security protocol alpha requires encrypted tokens.")
        self.memory.store("KNOWLEDGE-02", "Data ingestion pipeline accepts JSON and CSV payloads.")
        
        # Subscribe loggers to bus events
        self.bus.subscribe("agent:completed", lambda data: logging.info(f"[EVENT BUS] Task {data['task_id']} broadcast received."))

        # Spawn Worker Pool
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)

    async def shutdown(self):
        await self.task_queue.join()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logging.info("System Engine Shutdown gracefully.")

async def main():
    engine = OrchestratorEngine(num_workers=3)
    await engine.start()

    # Enqueue mock production jobs with varying priorities
    await engine.enqueue_task({"query": "Process system logs for security audit"}, priority=TaskPriority.HIGH)
    await engine.enqueue_task({"query": "Ingest user database CSV files"}, priority=TaskPriority.LOW)
    await engine.enqueue_task({"query": "Emergency patch authorization key"}, priority=TaskPriority.CRITICAL)
    await engine.enqueue_task({"query": "Execute natural language document parsing"}, priority=TaskPriority.MEDIUM)

    # Allow task queue processing to finish
    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
