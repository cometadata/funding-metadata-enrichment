<script>
  import { taskStore, resetTask } from './lib/stores/taskStore.js';
  import { uploadFileAsync } from './lib/api.js';
  import { fade, fly } from 'svelte/transition';
  
  import Uploader from './lib/components/Uploader.svelte';
  import Progress from './lib/components/Progress.svelte';
  import ResultCard from './lib/components/ResultCard.svelte';
  import ResultsSummary from './lib/components/ResultsSummary.svelte';
  import ProblematicStatements from './lib/components/ProblematicStatements.svelte';
  
  import { AlertCircle, FileSearch } from 'lucide-svelte';
  
  async function handleUpload(event) {
    const { file, normalize } = event.detail;
    
    $taskStore.file = file;
    
    try {
      await uploadFileAsync(file, normalize);
    } catch (error) {
      console.error('Upload error:', error);
    }
  }
  
  function handleReset() {
    resetTask();
  }
</script>

<main class="min-h-screen bg-gray-50">
  <div class="container mx-auto px-4 py-8 max-w-4xl">
    <header class="text-center mb-10">
      <div class="flex items-center justify-center gap-3 mb-4">
        <FileSearch class="w-10 h-10 text-blue-600" />
        <h1 class="text-3xl font-bold text-gray-900">COMET Funding Statement Extractor</h1>
      </div>
      <p class="text-gray-600">Upload a Markdown file to extract funding acknowledgements</p>
    </header>
    
    {#if $taskStore.status === 'idle'}
      <div in:fade={{ duration: 300 }}>
        <Uploader on:submit={handleUpload} />
      </div>
    {:else if $taskStore.status === 'uploading' || $taskStore.status === 'processing'}
      <div in:fly={{ y: 20, duration: 300 }}>
        <Progress 
          percentage={$taskStore.percentage} 
          message={$taskStore.message}
        />
      </div>
    {:else if $taskStore.status === 'completed' && $taskStore.result}
      <div in:fly={{ y: 20, duration: 300 }}>
        <ResultsSummary 
          summary={$taskStore.result.summary}
          metadata={$taskStore.result.metadata}
        />
        
        {#if $taskStore.result.funding_statements.length > 0}
          <div class="space-y-4 mb-8">
            {#each $taskStore.result.funding_statements as statement, i}
              <ResultCard {statement} index={i} />
            {/each}
          </div>
        {:else}
          <div class="bg-gray-100 rounded-lg p-8 text-center mb-8">
            <p class="text-gray-600">No funding statements found in the document.</p>
          </div>
        {/if}
        
        {#if $taskStore.result.problematic_statements}
          <ProblematicStatements statements={$taskStore.result.problematic_statements} />
        {/if}
        
        <div class="mt-8 text-center">
          <button
            on:click={handleReset}
            class="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium
                   hover:bg-blue-700 transition-colors"
          >
            Process Another File
          </button>
        </div>
      </div>
    {:else if $taskStore.status === 'error'}
      <div class="bg-red-50 border border-red-200 rounded-lg p-6" in:fade={{ duration: 300 }}>
        <div class="flex items-start gap-3">
          <AlertCircle class="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 class="text-lg font-semibold text-gray-900 mb-2">Processing Error</h3>
            <p class="text-gray-700">{$taskStore.error || 'An unexpected error occurred'}</p>
            
            <button
              on:click={handleReset}
              class="mt-4 px-4 py-2 bg-red-600 text-white rounded font-medium
                     hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    {/if}
  </div>
</main>
