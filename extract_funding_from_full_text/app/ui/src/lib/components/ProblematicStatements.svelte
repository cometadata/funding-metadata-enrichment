<script>
  import { fade } from 'svelte/transition';
  import { AlertTriangle } from 'lucide-svelte';
  
  export let statements = [];
</script>

{#if statements.length > 0}
  <div class="mt-8" in:fade={{ duration: 300 }}>
    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
      <div class="flex items-start gap-3">
        <AlertTriangle class="w-6 h-6 text-yellow-600 flex-shrink-0 mt-0.5" />
        
        <div class="flex-1">
          <h3 class="text-lg font-semibold text-gray-900 mb-3">
            Statements with Potential Formatting Issues
          </h3>
          
          <div class="space-y-3">
            {#each statements as statement, i}
              <div class="bg-white rounded p-4 border border-yellow-200">
                <p class="text-sm text-gray-700 font-mono whitespace-pre-wrap">{statement}</p>
              </div>
            {/each}
          </div>
          
          <p class="text-sm text-gray-600 mt-3">
            These statements may contain table formatting or other structural issues that prevented proper normalization.
          </p>
        </div>
      </div>
    </div>
  </div>
{/if}