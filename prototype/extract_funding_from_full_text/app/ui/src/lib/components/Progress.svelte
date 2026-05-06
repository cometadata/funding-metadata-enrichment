<script>
  import { fade } from 'svelte/transition';
  
  export let percentage = 0;
  export let message = '';
</script>

<div class="w-full max-w-2xl mx-auto" in:fade={{ duration: 300 }}>
  <div class="mb-4">
    <div class="flex justify-between mb-2">
      <span class="text-sm font-medium text-gray-700">Processing</span>
      <span class="text-sm text-gray-500">{Math.round(percentage * 100)}%</span>
    </div>
    
    <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
      <div 
        class="bg-blue-600 h-full rounded-full transition-all duration-300 ease-out"
        style="width: {percentage * 100}%"
      >
        <div class="h-full bg-gradient-to-r from-blue-500 to-blue-600 animate-pulse"></div>
      </div>
    </div>
  </div>
  
  {#if message}
    <p class="text-sm text-gray-600 text-center animate-pulse" in:fade={{ duration: 200 }}>
      {message}
    </p>
  {/if}
</div>