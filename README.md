[<img src="http://image.jeuxvideo.com/medias-md/149207/1492068501-5365-card.jpg" width="500px" alt="logo" />](https://github.com/gjeusel/opt_graph_EEL857)

# EEL857 Optimização em Grafos : [Luigi](http://www.cos.ufrj.br/~luidi/)

----
## Purpose :
Studying graph optimization algorithms for NP-Hard problem.

## Problem choosed :
Shortest path between spawns of Pokemon in San Fransisco.
Only rare pokemons will be considered.

Exemple : Dragonair, Charmeleon, Porygon

Database used : [Kaggle](https://www.kaggle.com/kveykva/sf-bay-area-pokemon-go-spawns)

----
## Usage :
python opt_graph.py --help

See From line 217 to 246 for graph algos.

<!-------->
<!--## Vizualization :-->
<!--1. Vizualization using **agraph**, see graphWrapper.\_\_str\_\_ method:-->
<!--<img src=results/agraph_example.png width="500px" alt="agraph_ex" />-->

<!--2. Vizualization opening **html** file in browser (always generated):-->
<!--<img src=results/screen_gmap_dragonair_example.png width="500px" alt="gmap_ex" />-->

----
## Results :
1. For Graph Order = 5 (i.e. only Dragonair) :
<img src=results/table_score_order5.png width="800px" alt="table_score_dragonair" />
<img src=results/screen_gmap_dragonair.png width="500px" alt="gmap_dragonair" />
2. For Graph Order = 10 (i.e. Dragonair & Porygon) :
<img src=results/table_score_order10.png width="800px" alt="table_score_dragonair-porygon" />
<img src=results/screen_gmap_dragonair-porygon.png width="500px" alt="gmap_dragonair-porygon" />
<!--3. For Graph Order = 12 (i.e. Dragonair & Charmeleon) :-->
