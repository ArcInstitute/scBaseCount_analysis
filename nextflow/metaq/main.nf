#!/usr/bin/env nextflow

// Define parameters
params.input_dir = "${baseDir}/GeneFull_Ex50pAS/GeneFull_Ex50pAS"
params.output_dir = "${baseDir}/metaq"
params.frequency = 0.25
params.cpus = 4
params.memory = '40G'
params.queue = 'gpu_batch_high_mem,gpu_high_mem,gpu_batch,preemptible,gpu'  // Default queue/partition

// Print parameter information
log.info """\
         METAQ H5AD PROCESSING PIPELINE
         ===========================
         Input directory : ${params.input_dir}
         Output directory: ${params.output_dir}
         Meta Frequency  : ${params.frequency}
         CPUs            : ${params.cpus}
         Memory          : ${params.memory}
         Queue/Partition : ${params.queue}
         """
         .stripIndent()

// Process to find all h5ad files and create a channel
process findH5adFiles {

    executor 'local'
    cpus 1

    output:
    path 'file_list.txt', emit: file_list

    script:
    """
    find ${params.input_dir} -name "*.h5ad" > file_list.txt
    """
}

// Process each h5ad file
process processH5ad {
    tag { h5ad_file }
    
    // Resource requirements for the HPC
    cpus params.cpus
    memory params.memory
    queue params.queue
    clusterOptions '--gpus=1'
    
    // Error strategy - retry on error up to 3 times
    errorStrategy { task.attempt <= 3 ? 'retry' : 'finish' }
    maxRetries 3
    
    input:
    val h5ad_file
    
    output:
    val "${h5ad_file}", emit: processed_file
    
    script:
    // Extract species from the file path
    // Assuming path structure like /path/to/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/file.h5ad
    def species = h5ad_file.toString().split('/')[-2]
    
    // Extract basename
    def basename = h5ad_file.toString().split('/')[-1]
    
    // Create output directory if it doesn't exist
    """
    mkdir -p ${params.output_dir}/${species}
    
    # Run the metaq command
    run_metaq -i ${h5ad_file} -o ${params.output_dir}/${species}/${basename} -f ${params.frequency}
    """
}

// Main workflow
workflow {
    // Find all h5ad files
    findH5adFiles()
    
    // Convert file list to a channel of individual file paths
    h5ad_files = findH5adFiles.out.file_list
        .splitText()
        .map { it.trim() }
        .filter { it.endsWith('.h5ad') }
    
    // Process each h5ad file
    processH5ad(h5ad_files)
    
    // Summary of processed files
    processH5ad.out.processed_file
        .collect()
        .view { processed_files ->
            "\nProcessing complete. Processed ${processed_files.size()} files.\n"
        }
}
