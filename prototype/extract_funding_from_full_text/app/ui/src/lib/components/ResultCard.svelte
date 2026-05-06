<script>
  import { fade, slide } from 'svelte/transition';
  
  export let statement;
  export let index = 0;
</script>

<div 
  class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
  in:fade={{ duration: 300, delay: index * 50 }}
>
  <div class="space-y-3">
    <p class="text-gray-800 leading-relaxed">
      {statement.statement}
    </p>
    
    <div class="flex items-center gap-4 text-sm text-gray-500">
      <div class="flex items-center gap-1">
        <span class="font-medium">Score:</span>
        <span class="text-blue-600 font-mono">{statement.score.toFixed(2)}</span>
      </div>
      
      <div class="flex items-center gap-1">
        <span class="font-medium">Query:</span>
        <span class="text-purple-600">{statement.query}</span>
      </div>
      
      {#if statement.normalized}
        <span class="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
          Normalized
        </span>
      {/if}
    </div>
  </div>
</div>