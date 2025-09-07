import { taskStore } from './stores/taskStore.js';

const API_BASE = '/api';

export async function uploadFileAsync(file, normalize = true) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('normalize', normalize.toString());

  try {
    taskStore.update(state => ({ ...state, status: 'uploading' }));
    
    const response = await fetch(`${API_BASE}/extract-funding-async`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      let errorMessage = 'Upload failed';
      try {
        const error = await response.json();
        errorMessage = error.detail || errorMessage;
      } catch (e) {
        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    let data;
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      data = await response.json();
    } else {
      throw new Error("Unexpected response format from server");
    }
    
  
    connectWebSocket(data.task_id);
    
    taskStore.update(state => ({ 
      ...state, 
      status: 'processing',
      taskId: data.task_id 
    }));
    
    return data;
  } catch (error) {
    taskStore.update(state => ({ 
      ...state, 
      status: 'error',
      error: error.message 
    }));
    throw error;
  }
}

function connectWebSocket(taskId) {

  try {
    const wsUrl = `ws://localhost:3000/ws/progress/${taskId}`;
    const ws = new WebSocket(wsUrl);
    
    let connected = false;
    
    ws.onopen = () => {
      connected = true;
      console.log('WebSocket connected for real-time updates');
    };
    
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      
      taskStore.update(state => ({
        ...state,
        percentage: update.percentage,
        message: update.message,
        status: update.completed ? (update.error ? 'error' : 'completed') : 'processing',
        result: update.result || state.result,
        error: update.error || state.error
      }));
      
      if (update.completed) {
        ws.close();
      }
    };
    
    ws.onerror = (error) => {
      if (!connected) {
  
        ws.close();
        startPolling(taskId);
      }
    };
    
    ws.onclose = () => {
      taskStore.update(state => ({ ...state, websocket: null }));
    };
  

    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      } else {
        clearInterval(pingInterval);
      }
    }, 30000);
    
    taskStore.update(state => ({ ...state, websocket: ws }));
  } catch (error) {

    startPolling(taskId);
  }
}

async function startPolling(taskId) {
  const pollInterval = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE}/task/${taskId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch task status');
      }
      
      const data = await response.json();
      
      taskStore.update(state => ({
        ...state,
        percentage: data.percentage,
        message: data.message,
        status: data.completed ? (data.error ? 'error' : 'completed') : 'processing',
        result: data.result || state.result,
        error: data.error || state.error
      }));
      
      if (data.completed) {
        clearInterval(pollInterval);
      }
    } catch (error) {

      if (!error.message.includes('Failed to fetch')) {
        console.error('Polling error:', error);
      }
      clearInterval(pollInterval);
      taskStore.update(state => ({ 
        ...state, 
        status: 'error',
        error: 'Failed to get task status' 
      }));
    }
  }, 1000);
}