# Overview
Each limb of the open3dhumanoid is a 3d printed link which houses either 1 or 2 joints. As per the 3d printed nature of 
the humanoid platform, the strength of the limbs are of paramount concern. With respects to our bio-inspired academic origins,
we have settled with the idea of using a porous bone infill (PBITO) topological optimisation. For a given voxel structure, the
PBITO is described as:

$do later$

From this equation, three parameters are under study: $\alpha, p, R$. To ensure scientific validity in all experiments to verify findings of this process,
all other variables will be control variables.
# GA Optimisation Process
The three parameters under study will be optimised with respect to a given limb. Consequently, there will be 33 versions of each parameter. 
To achieve this optimisation, a GA is employed. Each parameter forms a gene and a combination of the 3 parameters forms a genotype. A population of 500, per generation, is used to ensure
sufficient exploration of the fitness landscape.

## Fitness Function
The fitness of a given genotype, $\theta$, is equal to:
$$
f_{\theta}(m) = -\bigg(\phi_{\text{solid}}(\theta) + \lambda \cdot \text{max}\bigg(0, \frac{SF_{\text{target}}}{SF(\theta)} - 1\bigg)^2\bigg)
$$
Where:
- $SF$ denotes the safety factor:
- $\phi_{\text{solid}}(\theta)$ is the solid fraction the genotype $\theta$ resulted in during voxel optimisation.

# Simulation

To be determined!

# Results
To be determined!

# References