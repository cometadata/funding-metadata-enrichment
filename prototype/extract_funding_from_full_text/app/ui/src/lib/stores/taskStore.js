import { writable } from 'svelte/store';

export const taskStore = writable({
  status: 'idle', // idle, uploading, processing, completed, error
  file: null,
  taskId: null,
  percentage: 0,
  message: '',
  result: null,
  error: null,
  websocket: null
});

export function resetTask() {
  taskStore.update(state => {
    if (state.websocket) {
      state.websocket.close();
    }
    return {
      status: 'idle',
      file: null,
      taskId: null,
      percentage: 0,
      message: '',
      result: null,
      error: null,
      websocket: null
    };
  });
}