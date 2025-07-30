import json
import uuid
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict


from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel


class ProgressUpdate(BaseModel):
    """Model for progress updates."""
    task_id: str
    status: str
    percentage: float
    message: str
    timestamp: str
    completed: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskProgress(BaseModel):
    """Model for task progress state."""
    task_id: str
    status: str = "pending"
    percentage: float = 0.0
    message: str = "Task queued"
    created_at: str
    updated_at: str
    completed: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    websocket_clients: list = []


class ProgressTracker:
    """Manages progress tracking for async tasks."""
    
    def __init__(self, cleanup_interval: int = 300):  # Clean up after 5 minutes
        self.tasks: Dict[str, TaskProgress] = {}
        self.websocket_connections: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = cleanup_interval
        self._cleanup_task = None
    
    async def start(self):
        """Start the cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_old_tasks())
    
    async def stop(self):
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_old_tasks(self):
        """Periodically clean up completed tasks."""
        while True:
            try:
                await asyncio.sleep(60) 
                
                now = datetime.now()
                cutoff = now - timedelta(seconds=self.cleanup_interval)
                
                to_remove = []
                for task_id, task in self.tasks.items():
                    if task.completed:
                        updated = datetime.fromisoformat(task.updated_at)
                        if updated < cutoff:
                            to_remove.append(task_id)
                
                for task_id in to_remove:
                    del self.tasks[task_id]
                    if task_id in self.websocket_connections:
                        del self.websocket_connections[task_id]
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup task: {e}")
    
    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        self.tasks[task_id] = TaskProgress(
            task_id=task_id,
            created_at=now,
            updated_at=now
        )
        
        return task_id
    
    async def update_progress(
        self, 
        task_id: str, 
        status: str, 
        percentage: float,
        message: str = None
    ):
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.status = status
        task.percentage = min(max(percentage, 0.0), 1.0)
        task.updated_at = datetime.now().isoformat()
        
        if message:
            task.message = message
        
        update = ProgressUpdate(
            task_id=task_id,
            status=status,
            percentage=task.percentage,
            message=task.message,
            timestamp=task.updated_at,
            completed=task.completed
        )
        
        await self._broadcast_to_task_clients(task_id, update)
    
    async def complete_task(
        self, 
        task_id: str, 
        result: Any = None, 
        error: str = None
    ):
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.completed = True
        task.status = "error" if error else "completed"
        task.percentage = 1.0 if not error else task.percentage
        task.updated_at = datetime.now().isoformat()
        task.result = result
        task.error = error
        
        update = ProgressUpdate(
            task_id=task_id,
            status=task.status,
            percentage=task.percentage,
            message=error or "Task completed successfully",
            timestamp=task.updated_at,
            completed=True,
            result=result,
            error=error
        )
        
        await self._broadcast_to_task_clients(task_id, update)
        
        await asyncio.sleep(1)
        await self._close_task_connections(task_id)
    
    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get current progress for a task."""
        return self.tasks.get(task_id)
    
    async def connect_websocket(self, task_id: str, websocket: WebSocket):
        """Connect a WebSocket client to a task."""
        await websocket.accept()
        
        if task_id not in self.websocket_connections:
            self.websocket_connections[task_id] = []
        
        self.websocket_connections[task_id].append(websocket)
        
        if task_id in self.tasks:
            task = self.tasks[task_id]
            update = ProgressUpdate(
                task_id=task_id,
                status=task.status,
                percentage=task.percentage,
                message=task.message,
                timestamp=task.updated_at,
                completed=task.completed,
                result=task.result,
                error=task.error
            )
            
            try:
                await websocket.send_json(update.dict())
            except WebSocketDisconnect:
                await self.disconnect_websocket(task_id, websocket)
    
    async def disconnect_websocket(self, task_id: str, websocket: WebSocket):
        """Disconnect a WebSocket client from a task."""
        if task_id in self.websocket_connections:
            if websocket in self.websocket_connections[task_id]:
                self.websocket_connections[task_id].remove(websocket)
            
            if not self.websocket_connections[task_id]:
                del self.websocket_connections[task_id]
    
    async def _broadcast_to_task_clients(self, task_id: str, update: ProgressUpdate):
        """Broadcast an update to all clients watching a task."""
        if task_id not in self.websocket_connections:
            return
        
        disconnected = []
        for websocket in self.websocket_connections[task_id]:
            try:
                await websocket.send_json(update.dict())
            except (WebSocketDisconnect, Exception):
                disconnected.append(websocket)
        
        for websocket in disconnected:
            await self.disconnect_websocket(task_id, websocket)
    
    async def _close_task_connections(self, task_id: str):
        if task_id not in self.websocket_connections:
            return
        
        for websocket in self.websocket_connections[task_id][:]:
            try:
                await websocket.close()
            except Exception:
                pass
        
        if task_id in self.websocket_connections:
            del self.websocket_connections[task_id]


progress_tracker = ProgressTracker()