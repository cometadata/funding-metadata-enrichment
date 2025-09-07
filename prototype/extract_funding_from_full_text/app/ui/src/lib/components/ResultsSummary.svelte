<script>
  import { fade } from 'svelte/transition';
  import { CheckCircle2 } from 'lucide-svelte';
  
  export let summary;
  export let metadata;
</script>

<div class="bg-green-50 border border-green-200 rounded-lg p-6 mb-6" in:fade={{ duration: 300 }}>
  <div class="flex items-start gap-3">
    <CheckCircle2 class="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
    
    <div class="flex-1">
      <h2 class="text-lg font-semibold text-gray-900 mb-2">
        Found {summary.unique_statements} unique funding statement{summary.unique_statements !== 1 ? 's' : ''} in {metadata.processing_time.toFixed(1)} seconds
      </h2>
      
      <div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
        <div>
          <span class="text-gray-600">Total statements:</span>
          <span class="font-medium ml-1">{summary.total_statements}</span>
        </div>
        
        <div>
          <span class="text-gray-600">Paragraphs analyzed:</span>
          <span class="font-medium ml-1">{metadata.num_paragraphs}</span>
        </div>
        
        <div>
          <span class="text-gray-600">Processing time:</span>
          <span class="font-medium ml-1">{metadata.processing_time.toFixed(2)}s</span>
        </div>
      </div>
      
      {#if Object.keys(summary.statements_by_query).length > 0}
        <div class="mt-3 pt-3 border-t border-green-200">
          <p class="text-sm text-gray-600 mb-1">Statements by query:</p>
          <div class="flex flex-wrap gap-2">
            {#each Object.entries(summary.statements_by_query) as [query, count]}
              <span class="px-2 py-1 bg-white text-gray-700 text-xs rounded border border-gray-200">
                {query}: {count}
              </span>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>