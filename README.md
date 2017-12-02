# TSP-Hopfield-Tank

## Contents
* Hopfield-Tank network
* 'Heatmap image' generation for given step
* 'Heatmap over time' video generation (very usefull for debuging)
![Heatmaps over time](https://github.com/MacStan/TSP-Hopfield-Tank/blob/master/res/output.gif)

## Deployment
### Running
Programs supports commandline interface which allows to run network once, or multiple times in a row.
Properties of the network can be configured with arguments

```
  -h, --help            show this help message and exit
  --steps [STEPS]       Number of steps to take.
  --freq [FREQ]         Frequency of taking snapshots.
  --seeds [SEEDS [SEEDS ...]]
                        Seed for random. Defines whole run.
  --size-adjs [SIZE_ADJS [SIZE_ADJS ...]]
                        specifies value of size adjustment
  --tag [TAG]           tag added to name
```
### Dependancies
* Matplotlib 
* numpy
* ffmpeg 


## Theoretical remarks
Implementation of Hopfield-Tank model for TSP. Project for my University Course.

**Current version requires careful tuning of parameters to create feasible solutions.**
Look up page 16, second paragraph in following paper:
http://www.iro.umontreal.ca/~dift6751/paper_potvin_nn_tsp.pdf

Based on following article by John J Hopfield and D W Tank:
https://www.researchgate.net/publication/19135224_Neural_Computation_of_Decisions_in_Optimization_Problems
Mainly equations and constants from 4'th paragraph were used.

## Contributors
* Maciej Staniuk
* Mateusz Albecki




