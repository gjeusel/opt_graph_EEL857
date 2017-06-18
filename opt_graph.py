#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re
reload(sys)
sys.setdefaultencoding('utf8') # problem with encoding

import argparse
import subprocess

import matplotlib
matplotlib.use("Qt4Agg") # enable plt.show() to display
import matplotlib.pyplot as plt

import logging as log
import errno # cf error raised bu os.makedirs

import pandas as pd
import seaborn as sns

import math
import itertools
import numpy as np

import networkx as nx
# import pygraphviz as pgv

import time

##############################################
########      Global variables :      ########

# Paths
script_path = os.path.abspath(sys.argv[0])
working_dir_path = os.path.dirname(script_path)
default_csv_dir = working_dir_path+"/data/"

# Cmap
from matplotlib.colors import ListedColormap
cmap_OrRd = ListedColormap(sns.color_palette("OrRd", 10).as_hex())
cmap_RdYlBu = ListedColormap(sns.color_palette("RdYlBu", 10).as_hex())

# Some colors
color_green = sns.color_palette('GnBu', 10)[3]
color_blue = sns.color_palette("PuBu", 10)[7]
color_purple = sns.color_palette("PuBu", 10)[2]
color_red = sns.color_palette("OrRd", 10)[6]
##############################################


def compute_dist(lat1, lng1, lat2, lng2):
    """ Compute distance in km from lats and lngs. """
#{{{
    # cf http://www.movable-type.co.uk/scripts/latlong.html
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.pow(math.sin(delta_phi/2),2) \
        + math.cos(phi1) * math.cos(phi2) * math.pow(math.sin(delta_lambda/2),2)

    c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a))
    R = 6371 #[km]
    distance = R*c

    return distance
#}}}

# Function to convert Dataframe in nice colored table :
def render_mpl_table(data, col_width=4.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0.1, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """ <pandas.DataFrame> to nice table. """
#{{{
    import six
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, rowLabels=data.index, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return fig, ax
#}}}


class wrapperDataFrame:
    """
    - df : <pandas.DataFrame> of all pokemons in the csv
        schema : ['s2_id', 's2_token', 'num', name', 'lat', 'lng',
                      'encounter_ms', 'disppear_ms']

    - pok_nml_few : <list of string> containing the pokemon names of interest
    - df_few : <pandas.DataFrame> with only pokemons registers of interest

    - df_counts (optionally computed) :
        <pandas.DataFrame> of pokemons count
        schema : ['Pokemon',  'Count']

    - df_rarest (optionally computed) :
        <pandas.DataFrame> reduced to a % of the rarest
    """

    # s2_id and s2_token reference Google's S2 spatial area library.
    # num represents pokemon pokedex id
    # encounter_ms represents time of scan
    # disappear_ms represents time this encountered mon will despawn

#{{{ Methods of wrapperDataFrame

    def __init__(self, df_path=default_csv_dir+"pokemon-spawns.csv",
            pok_nml_few=["Dragonair"]): #constructor

        print "Reading csv ..."
        self.df = pd.read_csv(df_path)

        self.pok_nml_few = pok_nml_few
        print "Constructing df_few with : ", self.pok_nml_few, "..."
        self.construct_df_few()

        print "Removing spawns out of San Fransisco ..."
        self.clean_outofSF()

        print "Removing spawns in double ..."
        self.clean_spawns_pos_doubles()

    def __str__(self):
        # print ".df.head() = \n",self.df.loc[:,"num":"lng"].head() , "\n"
        # print ".df_counts.tail(10) = \n", self.df_counts.tail(10) , "\n"
        # print ".df_rarest.head() = \n", self.df_rarest.loc[:,"num":"lng"].head() , "\n"
        print ".df_few = \n", self.df_few.loc[:,"num":"lng"]
        return("")

    def construct_df_few(self):
#{{{
        self.df_few = pd.DataFrame()
        for pok_name in self.pok_nml_few:
            self.df_few = pd.concat( [self.df_few,
                    self.df.loc[self.df.loc[:,"name"] == pok_name] ] )
#}}}

    def clean_spawns_pos_doubles(self):
        """ Remove registers with the same lat AND lnd"""
#{{{
        i = 0 ; i_end = self.df_few.shape[0] ;
        k_end = self.df_few.shape[0] ;
        while (i < i_end):
            k = i+1
            while (k<k_end):
                # print "i=",i, "   ;  k=", k
                bool_lat = self.df_few.iloc[i].loc["lat"] == self.df_few.iloc[k].loc["lat"]
                bool_lng = self.df_few.iloc[i].loc["lng"] == self.df_few.iloc[k].loc["lng"]
                if (bool_lat and bool_lng):
                    self.df_few = self.df_few.drop(self.df_few.index[[k]])
                    i_end = i_end - 1
                    k_end = k_end - 1
                else:
                    k = k+1
            i = i+1
#}}}

    def clean_outofSF(self):
        """ Keep only registers with 36<lat<38 and -125<lng<-120 """
#{{{
        i = 0 ; i_end = self.df_few.shape[0] ;
        while (i < i_end):
            # print "i=",i"
            bool_lat_l = (36 < self.df_few.iloc[i].loc["lat"])
            bool_lat_r = (self.df_few.iloc[i].loc["lat"] < 38)

            bool_lng_l = (-125 < self.df_few.iloc[i].loc["lng"])
            bool_lng_r = (self.df_few.iloc[i].loc["lng"] < -120)

            bool_lat = bool_lat_l and bool_lat_r
            bool_lng = bool_lng_l and bool_lng_r

            if not(bool_lat and bool_lng):
                self.df_few = self.df_few.drop(self.df_few.index[[i]])
                i_end = i_end - 1
            else:
                i = i+1
#}}}

    def add_adress(self, lat=37.754242, lng=-122.383602):
        """ Add register at iloc 0 with your adress.
            default : 24th St, San Francisco, CA 94107, États-Unis
        """
#{{{
        s2 = pd.Series(['0', 'my_adress', lat, lng], index=['num', 'name', 'lat', 'lng'])
        self.df_few.loc[-1] = s2
        self.df_few = self.df_few.sort_index()
        self.df_few = self.df_few.reset_index()
#}}}


    def construct_df_rarest(self, threshold=10):
        """ Compute df_rarest with the threshold percents of the rarest. """
#{{{

        print "Couting spawns ..."
        self.df_counts = (self.df.groupby("name").size().to_frame()
                       .reset_index(level=0)
                       .rename(columns={0: "Count", "name": "Pokemon"})
                       .sort_values(by="Count", ascending=False))

        total_count = sum(self.df_counts.Count)
        n_last_lines = int(self.df_counts.size*threshold/100)
        counts_reduced = self.df_counts.tail(n_last_lines)

        print "Constructing df_rarest ..."
        self.df_rarest = self.df.loc[self.df["name"].isin(counts_reduced["Pokemon"])]
#}}}

    def plot_spawn_counts(self, ax):
        """ barplot of df_rarest """
#{{{
        # self argument needed, cf :
        # http://sametmax.com/quelques-erreurs-tordues-et-leurs-solutions-en-python/
        ax = sns.barplot(x="Pokemon", y="Count", data=self.df_counts, palette="GnBu_d")
        ax.set_xlabel("Pokemon")
        ax.set_xticklabels(self.df_counts["Pokemon"], rotation=90)
        ax.set_ylabel("Number of Spawns")
        return(ax)
#}}}

    def write_rarest_csv(self, path="./data/", threshold=10):
        """ df_rarest to csv with normalized filename. """
#{{{
        full_path =  path + "pokemon-spawns-" + str(int(threshold*100)) + "%-rarest.csv"
        print "Writting" + full_path + " ..."
        self.df_rarest.to_csv(full_path, index=False)
#}}}

#}}}



################################################################
#
#               Shortest Path algorithms :
#
################################################################


def verify_path(G, list_of_nodes):
    """ Function to verify if a path exists in the ordered list_of_nodes.
        Remark : it will always be the case with complete graph.
        return : bool
    """
#{{{
    path_found=True
    for i in range(0,len(list_of_nodes)-1):
        if not list_of_nodes[i+1] in G.neighbors(list_of_nodes[i]):
            path_found=False
    return path_found
#}}}

def brute_force(G):
    """ brute_force compute all combination of nodes path and get
        the shortest.
        Verification if the path exists from a list of nodes is used.
    """
#{{{
    min_dist = np.inf
    opt_list_of_nodes = []
    num_nodes = G.order() -1

    # Creating all permutations of [0, 1, 2, ..., n]
    combinations = itertools.permutations(np.arange(G.order()))

    for comb in combinations:
        list_of_nodes = np.array(list(comb)) # change type to array
        path_found = verify_path(G, list_of_nodes)
        if path_found is False:
            break
        # print "list_of_nodes = ", list_of_nodes

        dist_tmp = 0
        for i in range(0, num_nodes):
            dist_tmp = dist_tmp + G[list_of_nodes[i]][list_of_nodes[i+1]]['weight']

        # Add distance to make a cicle :
        dist_tmp = dist_tmp + G[list_of_nodes[num_nodes]][list_of_nodes[0]]['weight']

        if (dist_tmp < min_dist):
            opt_list_of_nodes = np.append(list_of_nodes, 0)
            min_dist = dist_tmp

    return min_dist, opt_list_of_nodes
#}}}

def swap(array, n1, n2):
    tmp = array[n1]
    array[n1] = array[n2]
    array[n2] = tmp
    return array

def backtrack_defby_rec(G, list_of_nodes, i_node=0, dist_tmp=0,
        min_dist=np.inf, opt_list_of_nodes=[]):
    """ backtrack_defby_rec : backtrack using verify_path function
        for solution viability, and already computed minimal dist
        to check if valid solution.

        It is a function defined by recurrency, so be carefull with
        variables scoops.
    """

#{{{
    num_nodes = G.order()-1

    # print "----------------------------------------"
    # print "backtrack_defby_rec have been called with :"
    # print "i_node = ", i_node
    # print "min_dist = ", min_dist
    # print "dist_tmp = ", dist_tmp
    # print "list_of_nodes = ", list_of_nodes
    # print "opt_list_of_nodes = ", opt_list_of_nodes

    if(i_node == num_nodes):
        min_dist = dist_tmp + G[list_of_nodes[num_nodes]][list_of_nodes[0]]['weight']
        opt_list_of_nodes = np.append(list_of_nodes, 0)
    else:
        for i in range(i_node+1, num_nodes+1):
            list_of_nodes = swap(list_of_nodes, i_node+1, i)

            # Not necessary for complete graph :
            path_found = verify_path(G, list_of_nodes)
            if path_found is False:
                break

            # Won't work :
            # dist_tmp = dist_tmp + G[list_of_nodes[i_node]][list_of_nodes[i_node+1]]['weight']
            # problem with shared memory between calls

            new_dist = dist_tmp + G[list_of_nodes[i_node]][list_of_nodes[i_node+1]]['weight']

            if (new_dist < min_dist):
                min_dist_returned, opt_list_of_nodes_returned = \
                        backtrack_defby_rec(G, list_of_nodes=list_of_nodes,
                            i_node=i_node+1,
                            dist_tmp=new_dist,
                            min_dist=min_dist,
                            opt_list_of_nodes=opt_list_of_nodes)

                if(min_dist_returned < min_dist):
                    min_dist = min_dist_returned
                    opt_list_of_nodes = opt_list_of_nodes_returned

            list_of_nodes = swap(list_of_nodes, i_node+1, i)

    return(min_dist, opt_list_of_nodes)
#}}}

def backtrack(G):
    """ backtrack : backtrack using verify_path function
        for solution viability, and already computed minimal dist
        to check if valid solution.
    """
    min_dist = np.inf
    opt_list_of_nodes = []
    num_nodes = G.order() -1

    # Creating all permutations of [0, 1, 2, ..., n]
    combinations = itertools.permutations(np.arange(G.order()))

    for comb in combinations:
        list_of_nodes = np.array(list(comb)) # change type to array
        path_found = verify_path(G, list_of_nodes)
        if path_found is False:
            break
        # print "list_of_nodes = ", list_of_nodes

        dist_tmp = 0
        loop_broken = False
        for i in range(0, num_nodes):
            dist_tmp = dist_tmp + G[list_of_nodes[i]][list_of_nodes[i+1]]['weight']
            if (dist_tmp > min_dist):
                loop_broken = True
                break

        # Add distance to make a cicle :
        dist_tmp = dist_tmp + G[list_of_nodes[num_nodes]][list_of_nodes[0]]['weight']

        if not(loop_broken) and (dist_tmp < min_dist) :
            min_dist = dist_tmp
            opt_list_of_nodes = np.append(list_of_nodes, 0)

    return min_dist, opt_list_of_nodes


def smallest_edge(G, idx_nodes_used, idx_node):
    # idx_node is the index of the node considered in left_in_LoN

    num_nodes = G.order()
    min_dist = np.inf

    for k in range(0, num_nodes):
        if not(k in idx_nodes_used):
            dist_tmp = G[idx_node][k]['weight']
            if dist_tmp < min_dist:
                min_dist = dist_tmp
                idx_next_node = k

    return idx_next_node, min_dist


# Strategy used : always choose the shortest edge
def heuristic(G, idx_first_node = 0):
    num_nodes = G.order()-1
    dist_path = 0

    idx_nodes_used = [idx_first_node]

    idx_current_node = idx_first_node
    while (len(idx_nodes_used)-1 < num_nodes):

        # finding smallest edge :
        idx_next_node, min_dist_edge = smallest_edge(G,
                idx_nodes_used = idx_nodes_used,
                idx_node = idx_current_node)

        # Update values :
        dist_path = dist_path + min_dist_edge
        idx_nodes_used.append(idx_next_node)
        idx_current_node = idx_next_node

    # Add distance to make a cicle :
    dist_path = dist_path + G[idx_first_node][idx_next_node]['weight']
    idx_nodes_used.append(idx_first_node)

    return dist_path, idx_nodes_used


# Class wrappers for resolution methods :
class graphWrapper:
    """
    - G : NetworkX graph
        node_scheme = ['num', 'name', 'lat', 'lng']
        edge : weight=dist, label=dist+"km"

    - df_scores : <pandas.DataFrame> of execution time and results
                  obtained by shortest path algos
        schema = [ 'Algo', 'Execution Time [s]', 'Shortest Path [km]',
                  'List of nodes ordered']

    """

#{{{ Methods of graphWrapper

    def __init__(self, df): #constructor
        print "Constructing NetworkX Graph ..."
        self.df_to_nx_complete(df)

        # Initialize df_scores :
        self.df_scores = pd.DataFrame(columns=['Execution Time [s]',
                'Shortest Path [km]', 'List of nodes ordered'])

    def df_to_nx_complete(self, df):
        """ Construct <networkx.classes.graph.Graph> from
            <pandas.DataFrame> as a complete graph ('as the crow flies').
        """
#{{{
        n = df.shape[0]
        self.G = nx.complete_graph(0)

        for i in range(0,n):
            self.G.add_node(i, \
                       num=df.iloc[i].loc["num"], \
                       name=df.iloc[i].loc["name"], \
                       lat=df.iloc[i].loc["lat"], \
                       lng=df.iloc[i].loc["lng"] \
                      )

        for i in range(0,n-1):
            for j in range(i+1,n):
                dist = compute_dist(self.G.node[i]["lat"], self.G.node[i]["lng"], \
                                    self.G.node[j]["lat"], self.G.node[j]["lng"])
                dist_trunc_str = str(int(dist)) + "km"
                self.G.add_edge(i, j, weight=dist, label=dist_trunc_str)
#}}}

    def __str__(self):
        """ print method of this class result in the bash cmd display of
            an agraph saved as png. """
#{{{
        # converting to Agraph :
        K_agraph = nx.nx_agraph.to_agraph(self.G)

        # Modifying attributes :
        palette = sns.color_palette("RdBu", n_colors=7)

        K_agraph.graph_attr['label']='San Fransisco Dragonair Pop'
        K_agraph.graph_attr['fontSize']='12'
        K_agraph.graph_attr['fontcolor']='black'

        # K_agraph.graph_attr['size']='1120,1120'

        # K_agraph.graph_attr.update(colorscheme=palette, ranksep='0.1')
        K_agraph.node_attr.update(color='red')
        K_agraph.edge_attr.update(color='blue')

        # Displaying by saving first and delete at the end
        K_agraph.write("tmp.dot")
        K_agraph.draw('tmp.png', prog="circo")

        command = "display -geometry 1200x720  tmp.png"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print "Displaying graph process returncode = ", process.returncode

        # command = "rm tmp.png tmp.dot"
        # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        # process.wait()
#}}}

    def wrapp_shortest_path(self, algo):
        """ add a register to df_scores"""
#{{{
        algo_str = re.search('<function (.+?) at', str(algo)).group(1)
        print "Computing Shortest Path with " + algo_str + " ..."

        start_time = time.time()
        if algo_str == 'backtrack_defby_rec':
            list_of_nodes = np.arange(self.G.order())
            min_dist, opt_LoN = algo(self.G, list_of_nodes)
        else:
            min_dist, opt_LoN = algo(self.G)
        end_time = time.time()
        total_time = end_time - start_time
        self.df_scores.loc[algo_str] = [total_time, min_dist, opt_LoN]
#}}}

    def compute_shortest_path(self, algo_nml=
            [brute_force, backtrack, backtrack_defby_rec, heuristic]):
        """ Compute Shortest Paths using algos in algo_nml. """

        for e in algo_nml:
            self.wrapp_shortest_path(e)

    def display_scores(self):
        fig, ax = render_mpl_table(self.df_scores, header_color=color_blue,
                              row_colors=['w', 'w'], edge_color='w')
        return fig, ax
#}}}


# Experimentation of vizualization using GoogleMap API :
def write_pok_gmap_loc(dfs, pathname="/home/gjeusel/projects/opt_graph/"): #{{{

    fout = pathname + '-'.join(dfs.pok_nml_few).lower() + "-locations.txt"
    print "Writting ", fout, " ..."
    f = open(fout, 'w')

    for i in range(dfs.df_few.shape[0]):
        line = dfs.df_few.iloc[i]
        tmp = str(line["lat"]) + "," + str(line["lng"]) + "\n"
        # print tmp
        f.write(tmp)

    f.close()

    fhtml = pathname + "gmap_" + '-'.join(dfs.pok_nml_few).lower() + ".html"
    print "Writting ", fhtml, " ...\n"
    f = open(fhtml, 'w')

    f.write("""<!DOCTYPE html>
<html>

  <head>
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
    <meta charset="utf-8">
    <title>San Fransisco Rare Pokemons Hunt</title>
    <style>
      /* Always set the map height explicitly to define the size of the div
       * element that contains the map. */
      #map {
         height: 100%;
         width: 70%;
      }
      /* Optional: Makes the sample page fill the window. */
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
      #right-panel {
              font-family: 'Roboto','sans-serif';
              line-height: 30px;
              padding-left: 10px;
      }

      #right-panel select, #right-panel input {
              font-size: 20px;
      }

      #right-panel select {
               width: 100%;
      }
      #right-panel i {
               font-size: 20px;
      }
      html, body {
              height: 100%;
              margin: 0;
              padding: 0;
      }
      #right-panel {
              float: right;
              width: 28%;
              padding-left: 2%;
      }
      #output {
              font-size: 15x;
      }

    </style>
  </head>

  <body>

    <div id="right-panel">
      <div id="inputs">
        <pre>

""")

    pok_list_str = "" + "var pok_list = [\n"
    for i in range(0,dfs.df_few.shape[0]):
        tmp_str = "  ['" + str(dfs.df_few.iloc[i].loc["name"]) + "', " \
                + str(dfs.df_few.iloc[i].loc["lat"]) + ", " \
                + str(dfs.df_few.iloc[i].loc["lng"]) + ", " \
                + str(i) + "],\n"
        pok_list_str = pok_list_str + tmp_str
    pok_list_str = pok_list_str + "        ];\n"

    f.write(pok_list_str)

    f.write("""
        </pre>
      </div>
      <div>
        <strong>Results</strong>
      </div>
      <div id="output"></div>
    </div>


    <div id="map"></div>
    <script>

    function initMap() {
        var map = new google.maps.Map(document.getElementById('map'), {
        zoom: 10,
        center: {lat: 37.615223, lng: -122.389977},
        mapTypeId: 'terrain'
        });

        var dragonairImage = {
           url: 'http://pre00.deviantart.net/73f7/th/pre/i/2013/024/f/5/dragonair_by_darkheroic-d5sizqi.png',
           size: new google.maps.Size(70,70),
           origin: new google.maps.Point(0, 0),
           anchor: new google.maps.Point(0, 0),
           scaledSize: new google.maps.Size(60, 60),
           labelOrigin: new google.maps.Point(9, 8)
        };

        var homeImage = {
           url: 'http://www.icone-png.com/png/54/53529.png',
           size: new google.maps.Size(30,30),
           origin: new google.maps.Point(0, 0),
           anchor: new google.maps.Point(0, 0),
           scaledSize: new google.maps.Size(30, 30),
           labelOrigin: new google.maps.Point(-5, 8)
        };

        var charmeleonImage = {
           url: 'http://pokemonbr.net/wp-content/uploads/2016/08/charmeleon.png',
           size: new google.maps.Size(50,50),
           origin: new google.maps.Point(0, 0),
           anchor: new google.maps.Point(0, 0),
           scaledSize: new google.maps.Size(50, 50),
           labelOrigin: new google.maps.Point(-5, 8)
        };

        var porygonImage = {
           url:'http://vignette2.wikia.nocookie.net/pokemon/images/3/3b/137Porygon_AG_anime.png/revision/latest?cb=20141006025936',
           size: new google.maps.Size(50,50),
           origin: new google.maps.Point(0, 0),
           anchor: new google.maps.Point(0, 0),
           scaledSize: new google.maps.Size(50, 50),
           labelOrigin: new google.maps.Point(-5, 8)
        };


        // Shapes define the clickable region of the icon. The type defines an HTML
        // <area> element 'poly' which traces out a polygon as a series of X,Y points.
        // The final coordinate closes the poly by connecting to the first coordinate.
        var shape = {
            coords: [0, 0, 0, 50, 50, 50, 50, 0],
            type: 'poly'
        };

""")

    pok_list_str = "        " + "var pok_list = [\n"
    for i in range(0,dfs.df_few.shape[0]):
        tmp_str = "            ['" + str(dfs.df_few.iloc[i].loc["name"]) + "', " \
                + str(dfs.df_few.iloc[i].loc["lat"]) + ", " \
                + str(dfs.df_few.iloc[i].loc["lng"]) + ", " \
                + str(i) + "],\n"
        pok_list_str = pok_list_str + tmp_str
    pok_list_str = pok_list_str + "        ];\n"

    f.write(pok_list_str)

    f.write("""

        // Markers :
        for (var i = 0; i < pok_list.length; i++) {
            var mark = pok_list[i];
            if(mark[0]=="Dragonair"){
               var icon = dragonairImage;
            }
            if(mark[0]=="Charmeleon"){
                var icon = charmeleonImage;
            }
            if(mark[0]=="Porygon"){
                var icon = porygonImage;
            }
            if(mark[0]=="my_adress"){
                var icon = homeImage;
            }
            var marker = new google.maps.Marker({
                position: {lat: mark[1], lng: mark[2]},
                map: map,
                icon: icon,
                shape: shape,
                title: mark[0] + " : " + mark[3],
                zIndex: mark[3],
                label: {
                    text: i.toString(),
                    fontWeight: 'bold',
                    fontSize: '40px',
                    fontFamily: '"Courier New", Courier,Monospace',
                    color: 'black'
                }
            });
        }


        var bounds = new google.maps.LatLngBounds; // automate bounds
        var dist = '';

        var outputDiv = document.getElementById('output');
        outputDiv.innerHTML = 'From ----> To ----> distance <br>';

        // Distances :

        function calcDistance(origin1,destinationB,ref_Callback_calcDistance, k, n){
        var service = new google.maps.DistanceMatrixService();
        var temp_duration = 0;
        var temp_distance = 0;
        var testres;
        service.getDistanceMatrix(
                {
                origins: [origin1],
                destinations: [destinationB],
                travelMode: google.maps.TravelMode.DRIVING,
                unitSystem: google.maps.UnitSystem.METRIC,
                avoidHighways: false,
                avoidTolls: false

                }, function(response, status) {
                if (status !== google.maps.DistanceMatrixStatus.OK) {
                    alert('Error was: ' + status);
                    testres= {"duration":0,"distance":0};

                } else {
                    var originList = response.originAddresses;
                    var destinationList = response.destinationAddresses;
                    var showGeocodedAddressOnMap = function (asDestination) {
                        testres = function (results, status) {
                          if (status === 'OK') {
                            map.fitBounds(bounds.extend(results[0].geometry.location));
                          } else {
                            alert('Geocode was not successful due to: ' + status);
                          }
                        };
                    };

                    for (var i = 0; i < originList.length; i++) {
                        var results = response.rows[i].elements;
                        for (var j = 0; j < results.length; j++) {
                            temp_duration+=results[j].duration.text;
                            temp_distance+=results[j].distance.text;
                        }
                    }
                        testres=[temp_duration,temp_distance];

                        if(typeof ref_Callback_calcDistance === 'function'){
                             //calling the callback function
                            ref_Callback_calcDistance(testres, k, n)

                        }
                    }
                }
                );
        }

        function Callback_calcDistance(testres, k, n) {
            dist = testres[1];
            outputDiv.innerHTML += k + ' ----> ' + n + ' ----> ' + dist + '<br>'

            console.log(testres[1]);
        }

        for (var k = 0; k < pok_list.length; k++) {
            var origin = new google.maps.LatLng(pok_list[k][1], pok_list[k][2]);

            for (var n = 0; n < pok_list.length; n++) {
                if (n !== k) {
                    var dest = new google.maps.LatLng(pok_list[n][1],pok_list[n][2]);

                    //calling the calcDistance function and passing callback function reference
                    calcDistance(origin, dest, Callback_calcDistance, k,n);
                }
            }
        }

    }

    </script>

    <script async defer
      src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCgu_eNgt-Hiu0HAnZwkIWYcnUoLsGSqVs&callback=initMap">
    </script>

  </body>
</html>
""")

    f.close()
#}}}


def setup_argparser():
    """ Define and return the command argument parser. """
#{{{
    parser = argparse.ArgumentParser(description='''Graph Optimizatino Study.''')

    parser.add_argument('--show_counts', action='store_true', default=False, dest='show_counts',
            help='whether to run counts analysis or not')

    parser.add_argument('--adress', action='store', nargs=2, type=float,
            default=[37.877875, -122.305926], dest='adress', metavar=['lat','lnt'],
            help='which latitude and longitude for Home Adress ')

    parser.add_argument('--poks_hunted', default="Dragonair", dest='poks_hunted',
            type=str, metavar="'Dragonair, Porygon, ...'",
            help="which pokemons to hunt in comma separated list ")

    return parser
#}}}

def main():

    parser = setup_argparser()

    try:
        args = parser.parse_args()

    except argparse.ArgumentError as exc:
        log.exception('Error parsing options.')
        parser.error(str(exc.message))
        raise

    if (args.show_counts): # will only compute count barplot
        dfs = wrapperDataFrame()
        dfs.construct_df_rarest()
        fig, ax = plt.subplots(figsize=(30, 30))
        dfs.plot_spawn_counts(ax)
        plt.show() # interactive plot
        return

    # Convert from string to list of string
    poks_hunted_list = args.poks_hunted.split(",")
    poks_hunted_list = map(str.strip, poks_hunted_list)
    dfs = wrapperDataFrame(pok_nml_few = poks_hunted_list)

    # dfs.add_adress(lat=37.877875, lng=-122.305926)
    dfs.add_adress(lat=args.adress[0], lng=args.adress[1])

    # Generating html file :
    write_pok_gmap_loc(dfs)

    print dfs
    n_poks = dfs.df_few.shape[0]

    Gwrap = graphWrapper(dfs.df_few)
    # print Gwrap

    print "--------------------------------------------"
    print "For Graph of n=", n_poks

    # Gwrap.brute_force()
    # Gwrap.backtrack()
    # Gwrap.backtrack_defby_rec()
    # Gwrap.heuristic()
    # Gwrap.wrapp_shortest_path(brute_force)
    Gwrap.compute_shortest_path()

    print Gwrap.df_scores
    fig, ax = Gwrap.display_scores()
    fig.suptitle("Graph Order = " + str(n_poks))


    from IPython import embed; embed() # Enter Ipython
    # plt.show() # interactive plot

if __name__ == '__main__':
    main()
