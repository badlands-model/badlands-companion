# pyBadlands-Companion

## Installation

The companion contains a series of IPython notebooks to perform post and pre-processing tasks useful to run and analyse Badlands model.

The easiest way to install and run the **Companion** is by loading the associated [Docker container](http://hub.docker.com/u/badlandsmodel/dashboard/) using [Kitematic](https://docs.docker.com/kitematic/userguide/). Once **Kitematic** is installed on your computer, open it and look for **badlands-companion** via the *search* menu.  

## Structure

Each post and pre-processing notebook is associated with a Python file. We have chosen this structure to give you the transparency and opportunity to:
* clearly understand the creation and format of Badlands input file,
* perform some basic quantitative analyses of Badlands output file,
* easily design your own notebook and further improve this workflow.
 
If you have any suggestions or if you would like to share your own Badlands template with others, feel free to contact us, we will be happy to integrate your own workflow here.

## Pre-processing

The pre-processing notebooks will help you to create:
+ surface grids for generic (simple geometrical model) and real (based on etopo1) topographic/bathymetric dataset,
+ sea level fluctuations file (using Haq curve or building your own one) to look at the impact of sea-level change on landscape dynamics,
+ horizontal (uplift/subsidence) displacement maps to study landscape response to tectonic forces. 
+ regridding script for refining intial input file
 
## Post-processing

### Morpho & Hydrometrics analyses

**Morphometrics** refers to quantitative description and analysis of the produced Badlands landforms which could be applied to a particular kind of landform or to drainage basins and large regions. The following suite of geomorphic attributes could be extracted:
- gradient: magnitude of maximum gradient
- horizontal curvature describes convergent or divergent fluxes
- vertical curvature: positive values describe convex profile curvature, negative values concave profile.
- aspect: direction of maximum gradient
- discharge: it relates to the drainage area

**Hydrometrics** refers to quantitative description and analysis of water surface. We will show how you can extract a particular catchment from a given model and compute for this particular catchment a series of paramters such as:
- river profile evolution based on main stream elevation and distance to outlet,
- Peclet number distribution which evaluates the dominant processes shaping the landscape,
- Chi parameter that characterizes rivers system evolution based on terrain steepness and the arrangement of tributaries,
- discharge profiles along the main catchment river

### Stratigraphic analyses

TODO
