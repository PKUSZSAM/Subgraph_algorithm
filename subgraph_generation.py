import os
import re
import sys
import time
import math
import operator
import pymatgen as pm
from pymatgen.core import Structure as Struct
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.analysis.local_env import CrystalNN
import timeout_decorator as Timeout
import multiprocessing
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.core.periodic_table import Element
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import argparse

class Structure:
    '''
    this class is for storage structure information
    param:
    atom_position: a list for storing all atom position with fractional coordinates
    '''

    element_table = ['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                     'Ar',
                     'K', 'Ca',
                     'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                     'Rb', 'Sr', 'Y',
                     'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                     'Ba', 'La',
                     'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                     'W', 'Re',
                     'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                     'U', 'Np',
                     'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

    layer = 3
    atom_num = 0
    expand_cell_ratio = 1 #for some small system, it may need to be 2
    lattice = [[], [], []]
    atom_position = []
    cartesian_position = [[0.0, 0.0, 0.0] for i in range(atom_num)]
    expand_position = [[0.0, 0.0, 0.0] for i in
                       range(atom_num * ((2*expand_cell_ratio+1) ** 3))]  # now it is cartesian_position
    expand_index = []
    ex_eles_list = []
    eles = []  # for store all elements types
    eles_list = []  # for store all atoms' elements
    formula_dict = {}
    formula = ''
    is_order = True
    is_order_trans = False
    is_time_error = False

    def __init__(self, path, layer=3, type='CIF'):
        self.layer=layer
        if type == 'CIF':
            self.CIFloader(path)
        if type == 'POSCAR':
            self.POSCARloader(path)
        if type == 'CONFIG':
            self.CONFIGloader(path)

    def CIFloader(self, path):  # now is only load reduced formula
        self.structure = pm.Structure.from_file(path)
        self.is_order = self.structure.is_ordered
        if self.is_order == True:
            self.formula_dict = self.structure.composition.to_reduced_dict
            for i in range(len(self.formula_dict)):
                self.formula += list(self.formula_dict.keys())[i] + str(int(list(self.formula_dict.values())[i]))
        else:
            print('{} is scores to occupy,try to transpose'.format(path.split('/')[-1]))
            try:
                self.order_trans()
            except:
                self.is_time_error = True
            if self.is_order_trans == True:
                print('trans successful')
            else:
                print('trans fail')
        return

    def CONFIGloader(self, filepath):
        tmp = 0
        with open(filepath, 'r') as file:
            file_tmp = file.readlines()
            self.atom_num = int(file_tmp[0])
            self.atom_position = [[] for i in range(self.atom_num)]
            for i in range(2, 5):
                for ii in range(0, 3):
                    self.lattice[tmp].append(float(file_tmp[i].split()[ii]))
                tmp = tmp + 1
            tmp = 0
            for i in range(6, len(file_tmp)):
                self.eles_list.append(int(file_tmp[i].split()[0]))
                for ii in range(1, 4):
                    self.atom_position[tmp].append(float(file_tmp[i].split()[ii]))
                tmp = tmp + 1

        for i in range(len(self.eles_list)):
            if (self.eles.count(self.eles_list[i]) == 0):
                self.eles.append(self.eles_list[i])

        self.pm_struct = Struct.from_file(filepath)
        return

    def POSCARloader(self, path):
        return

    def log(self, path):
        if self.is_order_trans == True or self.is_order == True:
            with open('formula_list.txt', "a+") as f:
                f.write('{}\n'.format(self.formula))

        if self.is_time_error == True:
            with open('time_error.txt', "a+") as f:
                f.write('{}\n'.format(path))

        with open('file_list.txt', "a+") as f:
            f.write('{}\n'.format(path))

        return

    def store_by_formula(self, filename, path):
        if self.is_order == True or self.is_order_trans == True:
            file_from = 'ICSD_'
            dir_name = path + '/' + self.formula
            file_name = dir_name + '/' + file_from + filename
            if os.path.exists(dir_name) == False:
                os.makedirs(dir_name)
            structure_output = pm.io.cif.CifWriter(self.structure)
            structure_output.write_file(file_name)
        return

    @Timeout.timeout(600)
    def order_trans(self):
        # time.sleep(6)
        trans = OrderDisorderedStructureTransformation()
        try:
            trans.apply_transformation(self.structure, return_ranked_list=False)
            self.structure = trans.lowest_energy_structure
            self.formula_dict = self.structure.composition.to_reduced_dict
            for i in range(len(self.formula_dict)):
                self.formula += list(self.formula_dict.keys())[i] + str(int(list(self.formula_dict.values())[i]))
            self.is_order_trans = True
        except:
            self.is_order_trans = False

    def expand_cell(self, multi):
        num = 0
        expand_position = [[], [], []]
        expand_index = []
        self.ex_eles_list = []
        for i in range(self.atom_num):
            for ii in range(-multi, multi + 1):
                for jj in range(-multi, multi + 1):
                    for zz in range(-multi, multi + 1):
                        if ii == 0 and jj == 0 and zz == 0:
                            expand_index.append(num)
                        for k in range(3):
                            expand_position[k].append(
                                self.cartesian_position[i][k] + ii * self.lattice[0][k] + jj * self.lattice[1][k] + zz *
                                self.lattice[2][k])
                        self.ex_eles_list.append(self.eles_list[i])
                        num = num + 1
        self.expand_position = np.array(expand_position).transpose().reshape(-1, 3)
        self.expand_index = np.array(expand_index)
        return

    def calc_cartesian(self):
        self.cartesian_position = [[] for i in range(self.atom_num)]
        for i in range(self.atom_num):
            for j in range(3):
                self.cartesian_position[i].append(
                    float(self.atom_position[i][0]) * float(self.lattice[0][j]) + float(
                        self.atom_position[i][1]) * float(self.lattice[1][j]) + float(
                        self.atom_position[i][2]) * float(self.lattice[2][j]))
        return


    def get_covalent_neighbor(self,atom_index,layer,Graph, factor):

        print("now, beging the atom index {} layer {} neighbor search".format(atom_index, layer))
        print("the graph nodes are {}".format(list(Graph.nodes)))

        flag = False
        nbrs = []
        edges = []
        tmp = 0
        self.nbr_num = []
        nbrs.append([])
        nbrs[0].append(atom_index)
        nbrs[0].append({"element": self.ex_eles_list[atom_index]})
        tmp += 1

        for i in range(len(self.expand_position)):
            if atom_index==i:
                continue
            if (calc_dist(self.expand_position[atom_index], self.expand_position[i]) < self.cov_radii[index(self.eles,self.ex_eles_list[atom_index])][index(self.eles,self.ex_eles_list[i])] * factor):
                nbrs.append([])
                edges.append([])
                nbrs[tmp].append(i)
                nbrs[tmp].append({"element": self.ex_eles_list[i]})

                edges[tmp - 1].append(atom_index)
                edges[tmp - 1].append(i)
                edges[tmp - 1].append({"distance": calc_dist(self.expand_position[atom_index], self.expand_position[i])})
                tmp += 1

        for i in range(len(nbrs)):
            Graph.add_nodes_from([nbrs[i]])
        for i in range(len(edges)):
            Graph.add_edges_from([edges[i]])

        if layer == 1:
            return

        layer -= 1
        for e in list(Graph.adj[atom_index]):
            self.get_covalent_neighbor(e,layer,Graph)

        return

    def get_radius_covalent(self):
        covalent = [[], []]
        with open('covalent.config', 'r') as f:
            f_tmp = f.readlines()
            for i in range(len(f_tmp)):
                for ii in range(2):
                    covalent[ii].append(float(f_tmp[i].split()[ii]))
        cov=np.array(covalent)

        for i in range(len(self.eles)):
            flag=False
            for ii in range(len(covalent[0])):
                if(self.eles[i]==covalent[0][ii]):
                    flag=True
                    break
            if flag==False:
                print("Wrong!,{} element has not covalent radii!".format(self.eles[i]))

        radii=np.array([cov[1][np.where(cov[0]==i)][0] for i in self.eles])#加一个对于没有的元素的错误判断
        self.cov_radii=radii[:,np.newaxis]+radii[np.newaxis,:]

        return

    def calc_dist_normal(self):
        self.type_list_normal = [[] for i in range(len(self.voronoi_type_tmp))]
        #tmp_eles=np.array(self.eles)
        for i in range(len(self.voronoi_type_tmp)):
            for ii in range(len(self.type_list[i])):
                self.type_list_normal[i].append(self.type_list[i][ii]/self.cov_radii[index(self.eles,self.voronoi_type_tmp[i][0])][index(self.eles,self.voronoi_type_tmp[i][1])])
        return
    
    def get_Gaussian_Mixture(self,points):

        n_cluster=list(range(1,3))
        points_T = points.transpose()

        flag=np.where(points_T[:,0]==np.min(points_T[:,0]))[0].item()
        models=[GaussianMixture(n_components=i,random_state=2021,max_iter=200,covariance_type='full',n_init=3).fit(points_T) for i in n_cluster]
        aic=[m.aic(points_T) for m in models]
        bic=[m.bic(points_T) for m in models]
        aic_less_index=aic.index(min(aic))
        bic_less_index=bic.index(min(bic))
        if aic_less_index!=bic_less_index:
            print('WARNING! the Gaussian Mixture has different aic and bic less select!')
        model=models[aic_less_index]
        label=model.predict(points_T)
        cluster0=points_T[np.where(label==label[flag])].tolist()

        return cluster0

    def voronoi(self):
        self.total_voronoi_area=0
        type_list=[]
        self.voronoi_type=[]
        self.voronoi_type_tmp=[]
        self.voronoi_type_num = []
        self.cluster_center0 = []
        self.cluster_center1 = []
        self.cluster0 = []
        self.cluster1 = []
        self.distance_cluster= []
        self.ridge_area=[]
        vertices=[]
        flag = False
        nbr = [[], [], [], []]
        nbr_list = [[], [], [], []]
        try:
            vor = Voronoi(self.expand_position)
        except:
            print("Voronoi has some wrongs!!!!")

        for i in range(len(vor.ridge_points)):
            flag = False
            for ii in range(len(self.expand_index)):
                if self.expand_index[ii] == vor.ridge_points[i][0] or self.expand_index[ii] == vor.ridge_points[i][1]:
                    flag = True
            if flag == True:
                nbr[0].append(self.ex_eles_list[vor.ridge_points[i][0]])
                nbr[1].append(self.ex_eles_list[vor.ridge_points[i][1]])
                nbr[2].append(calc_dist(self.expand_position[vor.ridge_points[i][0]],
                                        self.expand_position[vor.ridge_points[i][1]]))
                nbr[3].append(vor.ridge_vertices[i])

        for i in range(len(nbr[0])):
            self.total_voronoi_area+=calc_area(vor.vertices[nbr[3][i]])


        for i in range(len(nbr[0])):
            if nbr[0][i]!=3 and nbr[1][i]!=3:
                if nbr[0][i] != nbr[1][i]:
                    for ii in range(len(nbr)):
                        nbr_list[ii].append(nbr[ii][i])

        self.voronoi_type_tmp, self.type_list,self.vertices_list= list_sort(nbr_list)

        for i in range(len(self.vertices_list)):
            self.ridge_area.append([])
            for ii in range(len(self.vertices_list[i])):
                self.ridge_area[i].append(calc_area(vor.vertices[nbr_list[3][self.vertices_list[i][ii]]])/self.total_voronoi_area)

        self.calc_dist_normal()

        dis_area=np.vstack((np.array(sum(self.type_list_normal,[])),np.array(sum(self.ridge_area,[]))))

        self.cluster0=self.get_Gaussian_Mixture(dis_area)

        num=0
        for i in range(len(self.ridge_area)):
            flag=False
            for ii in range(len(self.ridge_area[i])):
                if [x[1] for x in self.cluster0].count(self.ridge_area[i][ii])==1:
                    if flag==False:
                        self.voronoi_type.append(self.voronoi_type_tmp[i])
                        self.voronoi_type_num.append([])
                        self.voronoi_type_num[num].append(self.type_list[i][ii])
                        flag=True
                    else:
                        self.voronoi_type_num[num].append(self.type_list[i][ii])
            if flag==True:
                num=num+1

        return

    def get_crystalNN_cutoff(self):
        crystalNN_type = []
        crystalNN_cutoff = []
        for i, site in enumerate(self.pm_struct):
            nn = CrystalNN().get_nn_info(self.pm_struct, i)
            for j in range(len(nn)):
                if crystalNN_type.count([site.specie.number, nn[j]['site'].specie.number]) == 0:
                    # change the specie to element index
                    crystalNN_type.append([site.specie.number, nn[j]['site'].specie.number])
                    crystalNN_cutoff.append(site.distance(nn[j]['site']))
                else:
                    index = crystalNN_type.index([site.specie.number, nn[j]['site'].specie.number])
                    if crystalNN_cutoff[index] < site.distance(nn[j]['site']):
                        crystalNN_cutoff[index] = site.distance(nn[j]['site'])

        self.crystalNN_cutoff = crystalNN_cutoff
        self.crystalNN_type = crystalNN_type
        return

    def get_crystalNN_neighbor(self,atom_index,layer,Graph):

        print("now, beging the atom index {} layer {} neighbor search".format(atom_index, layer))
        print("the graph nodes are {}".format(list(Graph.nodes)))

        flag = False
        nbrs = []
        edges = []
        tmp = 0
        self.nbr_num = []
        nbrs.append([])
        nbrs[0].append(atom_index)
        nbrs[0].append({"element": self.ex_eles_list[atom_index]})
        tmp += 1

        for i in range(len(self.expand_position)):
            if atom_index==i:
                continue
            try:
                self.crystalNN_type.index([self.ex_eles_list[atom_index], self.ex_eles_list[i]])
            except:
                continue
            if (calc_dist(self.expand_position[atom_index], self.expand_position[i]) < self.crystalNN_cutoff[self.crystalNN_type.index([self.ex_eles_list[atom_index], self.ex_eles_list[i]])]+0.0001):
                nbrs.append([])
                edges.append([])
                nbrs[tmp].append(i)
                nbrs[tmp].append({"element": self.ex_eles_list[i]})
                edges[tmp - 1].append(atom_index)
                edges[tmp - 1].append(i)
                edges[tmp - 1].append({"distance": calc_dist(self.expand_position[atom_index], self.expand_position[i])})
                tmp += 1

        for i in range(len(nbrs)):
            Graph.add_nodes_from([nbrs[i]])
        for i in range(len(edges)):
            Graph.add_edges_from([edges[i]])

        if layer == 1:
            return

        layer -= 1
        for e in list(Graph.adj[atom_index]):
            self.get_crystalNN_neighbor(e,layer,Graph)

        return

    def crystalNN_flow(self,path):

        self.calc_cartesian()
        self.expand_cell(self.expand_cell_ratio)
        self.get_crystalNN_cutoff()
        self.graph_type = []
        self.ele_graph_type = [[] for i in range(len(self.eles))]

        for i in range(self.atom_num):
            flag = True
            flag_ele = True
            eles_index = self.eles.index(self.ex_eles_list[self.expand_index[i]])
            graph_tmp = nx.Graph()
            self.get_crystalNN_neighbor(atom_index=self.expand_index[i], layer=self.layer,
                         Graph=graph_tmp)  # atom index is about cell index in expand cell
            if i == 0:
                self.graph_type.append(graph_tmp)
                self.ele_graph_type[eles_index].append(graph_tmp)
            for ii in range(len(self.graph_type)):
                if (nx.is_isomorphic(self.graph_type[ii], graph_tmp) == True):
                    flag = False
                    break
            if flag == True:
                self.graph_type.append(graph_tmp)
            for ii in range(len(self.ele_graph_type[eles_index])):
                if (nx.is_isomorphic(self.ele_graph_type[eles_index][ii], graph_tmp) == True):
                    flag_ele = False
                    break
            if flag_ele == True:
                self.ele_graph_type[eles_index].append(graph_tmp)

        self.write_output(path)
        return

    def covalent_flow(self,path, factor=1.2):

        self.calc_cartesian()
        self.expand_cell(self.expand_cell_ratio)
        self.get_radius_covalent()
        self.graph_type = []
        self.ele_graph_type = [[] for i in range(len(self.eles))]

        for i in range(self.atom_num):
            flag = True
            flag_ele = True
            eles_index = self.eles.index(self.ex_eles_list[self.expand_index[i]])
            graph_tmp = nx.Graph()
            self.get_covalent_neighbor(atom_index=self.expand_index[i], layer=self.layer,
                         Graph=graph_tmp, factor=factor)  # atom index is about cell index in expand cell
            if i == 0:
                self.graph_type.append(graph_tmp)
                self.ele_graph_type[eles_index].append(graph_tmp)
            for ii in range(len(self.graph_type)):
                if (nx.is_isomorphic(self.graph_type[ii], graph_tmp) == True):
                    flag = False
                    break
            if flag == True:
                self.graph_type.append(graph_tmp)
            for ii in range(len(self.ele_graph_type[eles_index])):
                if (nx.is_isomorphic(self.ele_graph_type[eles_index][ii], graph_tmp) == True):
                    flag_ele = False
                    break
            if flag_ele == True:
                self.ele_graph_type[eles_index].append(graph_tmp)

        self.write_output(path)
        return

    def get_voronoi_area_neighbor(self,atom_index,layer,Graph):

        print("now, beging the atom index {} layer {} neighbor search".format(atom_index, layer))
        print("the graph nodes are {}".format(list(Graph.nodes)))

        flag = False
        nbrs = []
        edges = []
        tmp = 0
        self.nbr_num = []
        nbrs.append([])
        nbrs[0].append(atom_index)
        nbrs[0].append({"element": self.ex_eles_list[atom_index]})
        tmp += 1

        for i in range(len(self.expand_position)):
            if atom_index==i:
                continue
            if self.voronoi_type.count([self.ex_eles_list[atom_index], self.ex_eles_list[i]]) == 1 or \
                self.voronoi_type.count([self.ex_eles_list[i], self.ex_eles_list[atom_index]]) == 1:
                    tmp_index = self.voronoi_type.index([self.ex_eles_list[atom_index], self.ex_eles_list[i]]) if \
                    self.voronoi_type.count([self.ex_eles_list[atom_index], self.ex_eles_list[i]]) != 0 else \
                    self.voronoi_type.index([self.ex_eles_list[i], self.ex_eles_list[atom_index]])
                    cutoff = max(self.voronoi_type_num[tmp_index]) + 0.00001
            else:
                cutoff = 0.0
            if (calc_dist(self.expand_position[atom_index], self.expand_position[i]) < cutoff):
                nbrs.append([])
                edges.append([])
                nbrs[tmp].append(i)
                nbrs[tmp].append({"element": self.ex_eles_list[i]})

                edges[tmp - 1].append(atom_index)
                edges[tmp - 1].append(i)
                edges[tmp - 1].append({"distance": calc_dist(self.expand_position[atom_index], self.expand_position[i])})

                tmp += 1

        for i in range(len(nbrs)):
            Graph.add_nodes_from([nbrs[i]])
        for i in range(len(edges)):
            Graph.add_edges_from([edges[i]])

        if layer == 1:
            return

        layer -= 1
        for e in list(Graph.adj[atom_index]):
            self.get_voronoi_area_neighbor(e,layer,Graph)

        return

    def voronoi_area_flow(self,path):
        self.calc_cartesian()
        self.expand_cell(self.expand_cell_ratio)
        self.get_radius_covalent()
        self.voronoi()
        self.graph_type = []
        self.ele_graph_type = [[] for i in range(len(self.eles))]
        for i in range(self.atom_num):
            flag = True
            flag_ele = True
            eles_index = self.eles.index(self.ex_eles_list[self.expand_index[i]])
            graph_tmp = nx.Graph()
            self.get_voronoi_area_neighbor(atom_index=self.expand_index[i], layer=self.layer,
                                            Graph=graph_tmp)  # atom index is about cell index in expand cell
            if i == 0:
                self.graph_type.append(graph_tmp)
                self.ele_graph_type[eles_index].append(graph_tmp)
            for ii in range(len(self.graph_type)):
                if (nx.is_isomorphic(self.graph_type[ii], graph_tmp) == True):
                    flag = False
                    break
            if flag == True:
                self.graph_type.append(graph_tmp)
            for ii in range(len(self.ele_graph_type[eles_index])):
                if (nx.is_isomorphic(self.ele_graph_type[eles_index][ii], graph_tmp) == True):
                    flag_ele = False
                    break
            if flag_ele == True:
                self.ele_graph_type[eles_index].append(graph_tmp)

        self.write_output(path)
        return

    def write_output(self, path):
        tmp=0
        direc_graph_type = path[:path.index(path.split('/')[-1])] + 'graph_types'
        direc_ele_graph_type = path[:path.index(path.split('/')[-1])] + 'ele_graph_types'
        direc_type = []
        direc_ele = []
        direc_ele_type = [[] for i in range(len(self.eles))]

        for i in range(len(self.graph_type)):
            direc_type.append('/type_{}'.format(len(self.graph_type[i])))
        for i in range(len(self.ele_graph_type)):
            direc_ele.append('/element_{}'.format(self.element_table[self.eles[i]]))
            for ii in range(len(self.ele_graph_type[i])):
                direc_ele_type[i].append('/type_{}'.format(len(self.ele_graph_type[i][ii])))

        #warning! we should consider about the case
        #that the number of neighboring atoms is the same but the connection relation is not

        for i in range(len(self.graph_type)):
            tmp = 0
            multi_num=direc_type.count('/type_{}'.format(len(self.graph_type[i])))
            if multi_num !=1:
                for ii in range(multi_num):
                    tmp+=1
                    index=direc_type.index('/type_{}'.format(len(self.graph_type[i])))
                    direc_type[index]=direc_type[index]+'_{}'.format(tmp)

        for i in range(len(self.ele_graph_type)):
            for ii in range(len(self.ele_graph_type[i])):
                tmp = 0
                multi_num = direc_ele_type[i].count('/type_{}'.format(len(self.ele_graph_type[i][ii])))
                if multi_num != 1:
                    for j in range(multi_num):
                        tmp+=1
                        index = direc_ele_type[i].index('/type_{}'.format(len(self.ele_graph_type[i][ii])))
                        direc_ele_type[i][index]=direc_ele_type[i][index]+'_{}'.format(tmp)

        if not os.path.exists(direc_graph_type):
            os.makedirs(direc_graph_type)
        if not os.path.exists(direc_ele_graph_type):
            os.makedirs(direc_ele_graph_type)
        for i in range(len(direc_type)):
            if not os.path.exists(direc_graph_type + direc_type[i]):
                os.makedirs(direc_graph_type + direc_type[i])
        for i in range(len(direc_ele)):
            if not os.path.exists(direc_ele_graph_type + direc_ele[i]):
                os.makedirs(direc_ele_graph_type + direc_ele[i])
            for ii in range(len(direc_ele_type[i])):
                if not os.path.exists(direc_ele_graph_type + direc_ele[i]+direc_ele_type[i][ii]):
                    os.makedirs(direc_ele_graph_type + direc_ele[i]+direc_ele_type[i][ii])

        for i in range(len(self.graph_type)):
            nx.write_graphml(self.graph_type[i], direc_graph_type + direc_type[i] + f'/layer{self.layer}.gml')
            nx.write_adjlist(self.graph_type[i], direc_graph_type + direc_type[i] + f'/layer{self.layer}.adjlist')
            #nx.draw(self.graph_type[i])
            #plt.savefig(direc_graph_type + direc_type[i] + '/layer3.png')
        for i in range(len(self.ele_graph_type)):
            for ii in range(len(self.ele_graph_type[i])):
                nx.write_graphml(self.ele_graph_type[i][ii],
                                 direc_ele_graph_type + direc_ele[i] + direc_ele_type[i][ii] + f'/layer{self.layer}.gml')
                nx.write_adjlist(self.ele_graph_type[i][ii],
                                 direc_ele_graph_type + direc_ele[i] + direc_ele_type[i][ii] + f'/layer{self.layer}.adjlist')
                #nx.draw(self.ele_graph_type[i][ii])
                #plt.savefig(direc_ele_graph_type + direc_ele[i] + direc_ele_type[i][ii] + '/layer3.png')

        return

def calc_dist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2))

def list_reshape(original_list):

    return [list(row) for row in zip(*original_list)]

def list_sort(list):
    flag = False
    flag_repeat=False
    dtype = 0
    type_list = []
    type = []
    index=[]
    # list_tmp = np.transpose(np.array(list))
    # list_np = list_tmp.tolist()
    list_np = list_reshape(list)
    for i in range(len(list_np)):
        flag = False
        flag_repeat = False
        for ii in range(dtype):
            tmp = list_np[i][:2]
            tmp.sort()
            if operator.eq(tmp, type[ii]) == False:
                flag = False
            else:
                flag = True
                break
        if flag == False:
            type_list.append([])
            type_list[dtype].append(np.around(list_np[i][2],5))
            index.append([])
            index[dtype].append(i)
            tmp = list_np[i][:2]
            tmp.sort()
            type.append(tmp)
            dtype = dtype + 1
        else:
            for jj in range(len(type_list[ii])):#for reduce repeat
                if np.around(type_list[ii][jj],5)==np.around(list_np[i][2],5):
                    flag_repeat = True
                    break
            if flag_repeat == False:
                type_list[ii].append(np.around(list_np[i][2],5))
                index[ii].append(i)

    #np.unique(type)

    return type, type_list,index

def calc_area(list):
    area=0
    for i in range(2,len(list)):
        a=calc_dist(list[i-2],list[i-1])
        b=calc_dist(list[i-2],list[i])
        c=calc_dist(list[i-1],list[i])
        p=(a+b+c)/2
        area += np.sqrt(p*(p-a)*(p-b)*(p-c))
    return area

def index(a,b):
    for i in range(len(a)):
        if a[i]==b:
                return i

if __name__ == '__main__':
    # nbr_type_list = ['Voronoi_area', 'covalent', 'crystalNN']
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str, required=True, help='Path to the structure file')
    argparser.add_argument('--gdist', type=int, required=True, help='The graph distance of subgraph generation')
    argparser.add_argument('--factor', type=float, required=False, default=1.2 ,help='The factor for threshold method')
    argparser.add_argument('--nbr_type', type=str, required=False, default='Voronoi_area', choices=['voronoi_area', 'covalent', 'crystalNN'],help='The type of neighbor judgement')
    args = argparser.parse_args()

    path = args.path
    layer = args.gdist
    factor = args.factor
    nbr_type = args.nbr_type
    
    struct = Structure(path, layer=layer, type='CONFIG')
    if nbr_type == 'voronoi_area':
        struct.voronoi_area_flow(path)
    elif nbr_type == 'covalent':
        struct.covalent_flow(path, factor=factor)
    elif nbr_type == 'crystalNN':
        struct.crystalNN_flow(path)