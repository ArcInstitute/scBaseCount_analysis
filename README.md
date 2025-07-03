Analysis Code for scBaseCount
=============================

* [paper](https://www.biorxiv.org/content/10.1101/2025.02.27.640494v2)
* [website](https://arcinstitute.org/tools/virtualcellatlas)
* [figures](https://www.dropbox.com/scl/fo/60fdemdmbgoowo3equfb4/AP-8vF9IN2cD0oCwyVFnqGw?dl=0&e=1&rlkey=a9bunjscbdmgr1sslg2vaubvs)


# scBaseCount Run stats 

## `prod3` 

* First scBaseCount release
* Jan to March 2025 
* [paper](https://www.biorxiv.org/content/10.1101/2025.02.27.640494v2)

## `prod3`  + `prod4`

* April to June 2025
* Combined `prod3` and `prod4`

## Chris' datasets

* Datasets with SRA experiment accession:
  * `/processed_datasets/scRecount/cellxgene/counted_SRXs/`
* Datasets lacking an SRA experiment accession:
  * `/processed_datasets/scRecount/cellxgene/`
    * `marmoset_brain_cell_atlas`, `AIDB`, `brain_cell_atlas`, `neocortex_atlas`


### AIDB:
* technology: 10x 5' v2
* Organism: Home sapiens
* CZI_collection_name: 'Asian immune diversity atlas'
* tissue: Blood

### neocortex_atlas:
* technology: 10x 3' v2
* Organism: Human
* CZI collection name: not a specific collection, but comes from BRAIN Initiative
* tissue: neocortex

### marmoset_brain_cell_atlas:
* technology: 10x3' v2
* Organism: Callithrix jacchus
* CZI collection name: A marmoset brain cell census reveals influence of developmental origin and functional class on neuronal identity
* tissue: brain


### SRAgent stats

* No. of SRA experiments accessed: 193806
* No. of 10X experiments: 98305
  * 3' => 68936
  * 5' => 15475

### scRecounter stats

* No. of organisms: 27
* No. of CellXGene tissues: 72
* No. of SRA experiments: 59555
* No. of SRA projects: TODO (`srx-to-project.py`)
* No. of reads mapped via STAR: 13.3 trillion


# Log

## June 26, 2025

### Nick

To "lock" in the scBaseCount dataset for the paper submission, I've stoped the scRecounter data processing jobs.
There are 4 jobs remaining, but they will likely all be finished within a couple of hours.

The current cell count is ~491M across 26 organisms.
There are 2 runs of the scRecounter pipeline:
  * `prod3` => what is shown in the current biorxiv paper
    * `gs://arc-ctc-screcounter/prod3/`
  * `prod4` => what has been done since (with updates to the pipeline such as Noam's xsra)
    * `gs://arc-ctc-screcounter/prod4/`
prod3 has been downloaded to chimera. It appears that my original download has been deleted (probably when space was very limited), but there appears to be at least a subset of files at /large_storage/ctc/datasets/scBasecamp
There is also some analysis files own by Rajesh at /large_storage/ctc/projects/vci/scbasecamp
The postgresql database with all metadata is located on GCP SQL. To access: see the SRAgent codebase, and ask me about the database credentials.

### Chris

`/large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS` now has all of the scBaseCamp h5ad files.








