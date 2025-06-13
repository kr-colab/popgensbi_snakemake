Simulators
==========

The simulator module provides classes for generating population genetic simulations under various demographic models. These simulators produce tree sequences that serve as the foundation for the simulation-based inference workflow.

Overview
--------

Simulators in popgensbi are responsible for:

- Generating tree sequences under specified demographic models
- Sampling parameters from prior distributions
- Simulating both ancestry and mutations
- Ensuring consistent random seeding for reproducibility

All simulators inherit from ``BaseSimulator`` and follow a consistent interface, making them interchangeable within the workflow.

Base Simulator
--------------

.. currentmodule:: workflow.scripts.ts_simulators

.. autoclass:: BaseSimulator
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all simulators. Handles configuration parsing and default parameter assignment.

   **Parameters:**
   
   - **config** (*dict*) -- Configuration dictionary with simulator parameters
   - **default** (*dict*) -- Default parameter values

Available Simulators
--------------------

YRI_CEU
~~~~~~~

.. autoclass:: YRI_CEU
   :members:
   :undoc-members:
   :show-inheritance:

   Simulates the demographic model from the dadi manual for YRI (Yoruba) and CEU (European) populations.

   **Model Description:**
   
   1. Ancestral population of size N_A
   2. Split into YRI and CEU populations at time Tp + T
   3. CEU undergoes bottleneck (N_CEU_initial) and exponential growth to N_CEU_final
   4. YRI maintains constant size N_YRI
   5. Continuous symmetric migration M between populations

   **Fixed Parameters:**
   
   - ``samples``: {"YRI": 10, "CEU": 10}
   - ``sequence_length``: 10e6
   - ``recombination_rate``: 1.5e-8
   - ``mutation_rate``: 1.5e-8

   **Inferred Parameters (uniform priors):**
   
   - ``N_A``: [1e2, 1e5] - Ancestral population size
   - ``N_YRI``: [1e2, 1e5] - YRI population size
   - ``N_CEU_initial``: [1e2, 1e5] - CEU bottleneck size
   - ``N_CEU_final``: [1e2, 1e5] - CEU final size
   - ``M``: [0, 5e-4] - Migration rate
   - ``Tp``: [0, 6e4] - Time before split
   - ``T``: [0, 6e4] - Time of split

   **Example Configuration:**

   .. code-block:: yaml

      simulator:
        class_name: YRI_CEU
        parameters:
          samples:
            YRI: 20
            CEU: 20
          sequence_length: 5e6

AraTha_2epoch
~~~~~~~~~~~~~

.. autoclass:: AraTha_2epoch
   :members:
   :undoc-members:
   :show-inheritance:

   Simulates the African2Epoch_1H18 model from stdpopsim for *Arabidopsis thaliana*.

   **Model Description:**
   
   A single population model with a size change event:
   
   1. Ancestral population of size N_A
   2. Size change to nu * N_A at time T * 2 * N_A generations ago

   **Fixed Parameters:**
   
   - ``samples``: {"SouthMiddleAtlas": 10}
   - ``sequence_length``: 10e6
   - ``recombination_rate``: 1.5e-8
   - ``mutation_rate``: 1.5e-8

   **Inferred Parameters (uniform priors):**
   
   - ``nu``: [0.01, 1] - Ratio of current to ancestral population size
   - ``T``: [0.01, 1.5] - Time of size change (scaled by 2*N_A)

   **Example Configuration:**

   .. code-block:: yaml

      simulator:
        class_name: AraTha_2epoch
        parameters:
          samples:
            SouthMiddleAtlas: 15
          sequence_length: 1e7

VariablePopulationSize
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VariablePopulationSize
   :members:
   :undoc-members:
   :show-inheritance:

   Simulates a single population with multiple size changes across time windows.

   **Model Description:**
   
   1. Single population with size changes at fixed time intervals
   2. Time points are exponentially spaced using the formula: ((exp(log(1 + rate*max_time) * i/(n-1)) - 1) / rate)
   3. Population sizes are sampled independently for each time window
   4. Ensures minimum number of SNPs (400) after MAF filtering

   **Fixed Parameters:**
   
   - ``samples``: {"pop": 10}
   - ``sequence_length``: 10e6
   - ``mutation_rate``: 1.5e-8
   - ``num_time_windows``: 3
   - ``max_time``: 100000
   - ``time_rate``: 0.1

   **Inferred Parameters (uniform priors):**
   
   - ``N_0, N_1, ..., N_{n-1}``: [1e2, 1e5] - Population sizes (log10 space)
   - ``recomb_rate``: [1e-9, 1e-7] - Recombination rate

   **Example Configuration:**

   .. code-block:: yaml

      simulator:
        class_name: VariablePopulationSize
        parameters:
          num_time_windows: 5
          pop_sizes: [1e3, 1e6]
          max_time: 50000

recombination_rate
~~~~~~~~~~~~~~~~~~

.. autoclass:: recombination_rate
   :members:
   :undoc-members:
   :show-inheritance:

   Simple constant-size population model with variable recombination rate.

   **Model Description:**
   
   Single population of constant size with recombination rate as the only inferred parameter.

   **Fixed Parameters:**
   
   - ``samples``: {0: 10}
   - ``sequence_length``: 1e6
   - ``mutation_rate``: 1.5e-8
   - ``pop_size``: 1e4

   **Inferred Parameters (uniform priors):**
   
   - ``recombination_rate``: [0, 1e-8]

   **Example Configuration:**

   .. code-block:: yaml

      simulator:
        class_name: recombination_rate
        parameters:
          pop_size: 5e4
          sequence_length: 5e6

Usage in Configuration Files
----------------------------

Simulators are specified in the workflow configuration YAML files:

.. code-block:: yaml

   simulator:
     class_name: YRI_CEU  # Name of the simulator class
     parameters:          # Override default parameters
       sequence_length: 5e6
       samples:
         YRI: 20
         CEU: 20

The ``class_name`` must match one of the available simulator classes. Parameters specified in the configuration will override the defaults.

Creating Custom Simulators
--------------------------

To create a custom simulator:

1. Inherit from ``BaseSimulator``
2. Define ``default_config`` with fixed and random parameters
3. Set up the parameter prior in ``__init__``
4. Implement ``__call__`` to return (tree_sequence, parameters)

Example:

.. code-block:: python

   class MySimulator(BaseSimulator):
       default_config = {
           # Fixed parameters
           "samples": {"pop1": 10},
           "sequence_length": 1e6,
           # Random parameters (ranges)
           "pop_size": [1e3, 1e5],
       }
       
       def __init__(self, config: dict):
           super().__init__(config, self.default_config)
           self.parameters = ["pop_size"]
           self.prior = BoxUniform(
               low=torch.tensor([self.pop_size[0]]),
               high=torch.tensor([self.pop_size[1]])
           )
       
       def __call__(self, seed: int = None):
           torch.manual_seed(seed)
           theta = self.prior.sample().numpy()
           pop_size = theta[0]
           
           # Simulate tree sequence
           ts = msprime.sim_ancestry(...)
           ts = msprime.sim_mutations(...)
           
           return ts, theta

Technical Notes
---------------

- All simulators use msprime for coalescent simulations
- Tree sequences include both topology and mutations
- Random seeds ensure reproducibility across runs
- Parameters are sampled from uniform priors (BoxUniform)
- The returned parameter vector matches the order in ``self.parameters``