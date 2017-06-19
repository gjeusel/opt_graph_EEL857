[<img src="http://image.jeuxvideo.com/medias-md/149207/1492068501-5365-card.jpg" width="400px" alt="logo" />](https://github.com/gjeusel/opt_graph_EEL857)

# EEL857 Optimização em Grafos : [Luigi](http://www.cos.ufrj.br/~luidi/)

Studying graph optimization algorithms for **NP-Hard** problem.

**Usage** :
> python opt_graph.py --help

examples :
> python opt_graph.py --poks_hunted 'Charmeleon, Dragonair'\
> python opt_graph.py --adress [37.877875, -122.305926]

----

## Problem choosed :
**Traveling Salesman** with spawns of rarest Pokemon in San Fransisco.\
Exemple : Dragonair, Charmeleon, Porygon

Database used : [Kaggle SF pokemon GO spawns](https://www.kaggle.com/kveykva/sf-bay-area-pokemon-go-spawns)\
schema :\
| s2_id | s2_token | num | name | lat | lng | encounter_ms | disppear_ms |
|-------|----------|-----|------|-----|-----|--------------|-------------|
-9182942301737976000|808fa2a1473|148|Dragonair|37.5320567008|-122.19265261|1469526527404|1469525805045
-9182942301737976000|808fa2a1473|148|Dragonair|37.5320567008|-122.19265261|1469526527404|1469525805453
-9182960995583132000|808f91a0917|5|Charmeleon|37.6764960101|-122.110238578|1469527541628|1469527278982
-9182960995583132000|808f91a0917|5|Charmeleon|37.6764960101|-122.110238578|1469527541628|1469527279064
-9182960995583132000|808f91a0917|5|Charmeleon|37.6764960101|-122.110238578|1469527541628|1469527279228
...

- s2_id and s2_token reference Google's S2 spatial area library.
- num represents pokemon pokedex id
- encounter_ms represents time of scan
- disappear_ms represents time this encountered mon will despawn

**Simplification** : graphs that will be generated with ['lat', 'lng'] will be considered complete.\
*As the crow flies* study.\
A possible future work should be to request existings bicycle, cars [,...] paths to Google API and use them to build the graph


----

## Algorithms :

----
### [Brute Force](https://github.com/gjeusel/opt_graph_EEL857/blob/master/opt_graph.py#L274)

----
### Backtracks
- [Defined with loops](https://github.com/gjeusel/opt_graph_EEL857/blob/master/opt_graph.py#L391)
- [Defined by recurrence](https://github.com/gjeusel/opt_graph_EEL857/blob/master/opt_graph.py#L316)

----
### [Heuristic greedy with shortest edge add](https://github.com/gjeusel/opt_graph_EEL857/blob/master/opt_graph.py#L468)

----
### [Heuristic neighboors](https://github.com/gjeusel/opt_graph_EEL857/blob/master/opt_graph.py#L493)
Improvement of previous heuristc with neighboors permutation checks.

----

# Results :

## Dragonair
<img src=results/table_score_n_poks_5.png width="800px" alt="table_score_dragonair" />
<img src=results/screen_gmap_dragonair.png width="500px" alt="gmap_dragonair" />

## Porygon
<img src=results/table_score_n_poks_6.png width="800px" alt="table_score_porygon" />
<img src=results/screen_gmap_porygon.png width="500px" alt="gmap_porygon" />

## Charmeleon
<img src=results/table_score_n_poks_8.png width="800px" alt="table_score_charmeleon" />
<img src=results/screen_gmap_charmeleon.png width="500px" alt="gmap_charmeleon" />

## Dragonair and Porygon
<img src=results/table_score_n_poks_10.png width="800px" alt="table_score_dragonair-porygon" />
<img src=results/screen_gmap_dragonair-porygon.png width="500px" alt="gmap_dragonair-porygon" />

## Dragonair and Charmeleon
<img src=results/table_score_n_poks_12.png width="800px" alt="table_score_dragonair-charmeleon" />
<img src=results/screen_gmap_dragonair-charmeleon.png width="500px" alt="gmap_dragonair-charmeleon" />
