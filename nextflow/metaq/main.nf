#!/usr/bin/env nextflow

// Define parameters
params.base_dir = "/large_storage/ctc/public/scBasecamp"
params.input_dir = "${params.base_dir}/GeneFull_Ex50pAS/GeneFull_Ex50pAS"
params.output_dir = "${params.base_dir}/metaq"
params.frequency = 0.2
params.min_genes = 300
params.min_umis = 500
params.cpus = 4
params.memory = '40G'
params.queue = 'gpu_batch_high_mem,gpu_high_mem,gpu_batch,preemptible,gpu'  // Default queue/partition
params.tsv_file = "${params.base_dir}/meta_analysis/passing_datasets.tsv"

// Print parameter information
log.info """\
         METAQ H5AD PROCESSING PIPELINE
         ===========================
         Input directory : ${params.input_dir}
         Output directory: ${params.output_dir}
         TSV File        : ${params.tsv_file}
         Meta Frequency  : ${params.frequency}
         Minimum Genes   : ${params.min_genes}
         Minimum UMIs    : ${params.min_umis}
         CPUs            : ${params.cpus}
         Memory          : ${params.memory}
         Queue/Partition : ${params.queue}
         """
         .stripIndent()

// Process to read the TSV file and create a channel of datasets to process
process readTsvFile {
    executor 'local'
    cpus 1
    
    output:
    path 'datasets_to_process.txt', emit: datasets_list
    
    script:
    """
    #!/usr/bin/env bash
    # Read the TSV file and create a list of SRX accessions with their species
    tail -n +2 ${params.tsv_file} | awk '{print \$1"\\t"\$2}' > datasets_to_process.txt
    """
}

// Process each dataset in the TSV
process processH5ad {
    tag { "${species}/${accession}" }

    // Resource requirements for the HPC
    cpus params.cpus
    memory params.memory
    queue params.queue
    clusterOptions '--gpus=1'

    // Error strategy - retry on error up to 3 times
    errorStrategy { task.attempt <= 3 ? 'retry' : 'finish' }
    maxRetries 3

    input:
    tuple val(species), val(accession)

    output:
    tuple val(species), val(accession), emit: processed_dataset

    script:
    // Construct the h5ad file path based on species and accession
    def h5ad_file = "${params.input_dir}/${species}/${accession}.h5ad"
    
    """
    # Create output directory if it doesn't exist
    mkdir -p ${params.output_dir}/${species}
    
    # Run the metaq command
    run_metaq \
        -i ${h5ad_file} \
        -o ${params.output_dir}/${species}/${accession}.h5ad \
        --min_genes ${params.min_genes} \
        --min_umi ${params.min_umis} \
        --frac_metacells ${params.frequency};
    """
}

// Main workflow
workflow {
    // Read the TSV file
    readTsvFile()
    
    // Convert TSV data to a channel of tuples (species, accession)
    datasets = readTsvFile.out.datasets_list
        .splitText()
        .map { line -> 
            def fields = line.trim().split('\t')
            return tuple(fields[0], fields[1])
        }
        .take(4)
    
    // Process each dataset
    processH5ad(datasets)
    
    // Summary of processed files
    processH5ad.out.processed_dataset
        .collect()
        .view { processed_datasets ->
            "\nProcessing complete. Processed ${processed_datasets.size()} datasets.\n"
        }
}