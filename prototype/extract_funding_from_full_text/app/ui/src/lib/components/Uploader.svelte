<script>
  import { createEventDispatcher } from 'svelte';
  import Dropzone from 'svelte-file-dropzone';
  import { FileText, X } from 'lucide-svelte';
  
  export let normalize = true;
  
  const dispatch = createEventDispatcher();
  
  let files = [];
  
  function handleFilesSelect(e) {
    const { acceptedFiles } = e.detail;
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.name.endsWith('.md')) {
        files = [file];
      } else {
        alert('Please select a Markdown (.md) file');
      }
    }
  }
  
  function removeFile() {
    files = [];
  }
  
  function handleSubmit() {
    if (files.length > 0) {
      dispatch('submit', { file: files[0], normalize });
    }
  }
</script>

<div class="w-full max-w-2xl mx-auto">
  <Dropzone
    on:drop={handleFilesSelect}
    accept=".md"
    multiple={false}
    containerClasses="w-full"
  >
    <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
      {#if files.length === 0}
        <FileText class="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p class="text-lg mb-2">Drag & drop a .md file here</p>
        <p class="text-sm text-gray-500">or click to select</p>
      {:else}
        <div class="flex items-center justify-center gap-2">
          <FileText class="w-6 h-6 text-blue-500" />
          <span class="text-lg">{files[0].name}</span>
          <button
            on:click|stopPropagation={removeFile}
            class="p-1 hover:bg-gray-100 rounded"
            type="button"
          >
            <X class="w-5 h-5 text-gray-500" />
          </button>
        </div>
      {/if}
    </div>
  </Dropzone>
  
  <div class="mt-6 space-y-4">
    <label class="flex items-center gap-2">
      <input
        type="checkbox"
        bind:checked={normalize}
        class="w-4 h-4 text-blue-600 rounded"
      />
      <span class="text-sm">Normalize statements</span>
    </label>
    
    <button
      on:click={handleSubmit}
      disabled={files.length === 0}
      class="w-full py-3 px-4 bg-blue-600 text-white rounded-lg font-medium
             hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
             transition-colors"
    >
      Extract Statements
    </button>
  </div>
</div>