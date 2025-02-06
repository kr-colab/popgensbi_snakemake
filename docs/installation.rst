Installation
=============

To install PopGenSBI, follow these steps:

1. **Clone the repository**:
   Open a terminal and run the following command to clone the repository:
   ```
   git clone https://github.com/kr-colab/popgensbi_snakemake.git
   ```

2. **Navigate to the project directory**:
   Change into the project directory:
   ```
   cd popgensbi_snakemake
   ```

3. **Create a conda environment**:
   Create a new conda environment using the provided environment file:
   ```
   conda env create -f workflow/environment.yaml
   ```

4. **Activate the environment**:
   Activate the newly created environment:
   ```
   conda activate popgensbi_env
   ```

5. **Development Dependencies** (if necessary):
   If you want to develop PopGenSBI, we have a separate set of dependencies
   in a conda environment. To install these dependencies, run the following command:
   ```
   conda env create -f requirements-dev.yaml
   ```


Now you are ready to use PopGenSBI for your simulation-based inference tasks in population genetics!
