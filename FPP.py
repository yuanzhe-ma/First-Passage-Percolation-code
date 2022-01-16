import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import igraph
import itertools
import random
import math
from ast import literal_eval
from joblib import Parallel, delayed
import joblib
import scipy
import os
from scipy import spatial


# In[2]:


def generate_neighbor_2d(X):     
    neighbor_list = [(X[0]-1,X[1]),(X[0]+1,X[1]),(X[0],X[1]-1),(X[0],X[1]+1)]
    return neighbor_list

#return all nodes in List that are one of the neighbor of node_a
def neighbor_in_list(node_a,List,if_edge = 1):
    if if_edge == 1:
        return ([(str(min(node_a,item)),str(max(node_a,item))) for item in generate_neighbor_2d(node_a) if item in List])
    else:
        return ([item for item in generate_neighbor_2d(node_a) if item in List])

class SingleNode:
    def __init__(self, time, coordinate):
        self.time = time
        self.coordinate = coordinate
        #self.norm = np.linalg.norm(coordinate, 2)
        self.neighbor =generate_neighbor_2d(coordinate)
        
        
def converge_id_from_old_graph(old_graph,new_graph,new_id):
    coord = [literal_eval(new_graph.vs[item]['name']) for item in new_id]
    old_id = [old_graph.vs.find(str(item)).index for item in coord]
    return ([coord,old_id])

#return the coord list of a boudary of a set B_t
def generate_boundary_coord(B_t, threshold  = 4):
    degree_list = B_t.degree([item for item in B_t.vs], mode='ALL', loops=False)
    #combine info for degree and vertex id
    degree_list= zip([item.index for item in B_t.vs], degree_list)
    degree_list = [item for item in degree_list if item[1] < threshold]
    #list every boundary point
    boundary_coord =[literal_eval(B_t.vs[item[0]]["name"]) for item in degree_list]
    return boundary_coord


def distance_between_set(A,B):
    try:
        return(np.min(scipy.spatial.distance.cdist(A,B)))
    except:
        return 0


# In[16]:


#Count hole function,  also measure the size of the boundary 
def count_hole_2d(ig,check_time,T_0_x_dict,D,B_t_vertex_coord,B_t_vertex_id,if_print):
    start_time = time.time()
    #Select a large box such that the complement of the graph is in the box, C_t = (B_t)^C
    C_t_vertex_id = [key[0] for key, value in T_0_x_dict.items() if (value > check_time) and abs(key[1][0]) <= D and abs(key[1][1]) <= D]
    #C_t_vertex_coord = [key[1] for key, value in T_0_x_dict.items() if value > check_time]
    G = ig.subgraph(C_t_vertex_id)
    #print('The running minutes to generate the graph of hole is ' + str((time.time() - start_time)/60.0))
    start_time_new = time.time()
    component_list = list(G.components(mode='WEAK'))
    N_hole = len(component_list) - 1
    #The one connected to the corner point must be unbouned
    corner_point_index = G.vs.find(str((-D,-D))).index
    hole_bounded = [corner_point_index not in item for item in component_list] 
    
    #list all the hole    
    Hole_list = [component_list[i] for i in range(len(component_list)) if hole_bounded[i] == True]
    Hole_coord_list = [[literal_eval(G.vs[x]['name']) for x in item] for item in Hole_list]
    N_hole_size = [len(item) for item in Hole_list] 
    
    
    #If the list of hole is empty, then max_size = 0
    try:
        max_size = max(N_hole_size)
    except:
        max_size = 0  
    
    '''
        ##largest hole largest l2 norm
    try:
        #find the largest l2 norm in the largest hole
        largest_hole = [Hole_list[i] for i in range(len(Hole_list)) if N_hole_size[i] == max_size][0]
        #convert the vertex index to coordinate 
        largest_hole = [literal_eval(G.vs[item]['name']) for item in largest_hole]
        largest_hole_l2_norm = [np.linalg.norm(item, 2) for item in largest_hole]
        largest_hole_largest_l2_norm = max(largest_hole_l2_norm)
    except:
        largest_hole_largest_l2_norm = 0
    '''

    #There should only be one unbounded_hole_component
    unbounded_hole_component = [component_list[i] for i in range(len(component_list)) if hole_bounded[i] == False][0]
    #convert the vertex index to coordinate
    #unbounded_hole_component = [literal_eval(G.vs[item]['name']) for item in unbounded_hole_component]
    [unbounded_hole_component_coord,unbounded_hole_component_id] = converge_id_from_old_graph(ig,G,unbounded_hole_component)
    
     
    #if the degree of a point < 4, then it is a boundary point    
    hole_list_id = [item for sublist in Hole_list for item in sublist]
    #converge id in B_t to the original graph, then generate subgraph
    [hole_list_coord,hole_list_id] = converge_id_from_old_graph(ig,G,hole_list_id)
    B_t = ig.subgraph(B_t_vertex_id)
    boundary_coord = generate_boundary_coord(B_t)
    
    #Exterior boundary, which we will call E_t
    E_t = ig.subgraph(unbounded_hole_component_id)
    exterior_boundary_coord = generate_boundary_coord(E_t)
    #We have to move the point on the boundary of the box
    exterior_boundary_coord = [item for item in exterior_boundary_coord if abs(item[0])!= D and abs(item[1])!= D]
    
    
    Hole_exterior_distance_list = [distance_between_set(hole,exterior_boundary_coord) for hole in Hole_coord_list]
 
   
    Boundary_size = len(boundary_coord)
    elapsed_time_G = time.time() - start_time
    total_size = sum(N_hole_size)
    avg_hole_boundary_distance = np.mean(Hole_exterior_distance_list)
    max_hole_boundary_distance = np.max(Hole_exterior_distance_list)	

    if if_print == 1:
        print('--At time ' + str(check_time))
        print('The number of holes is ' + str(N_hole))
        print('The total size of holes is ' + str(total_size))
        print('The maximum size of holes is ' + str(max_size))
        print('The size of the boundary is ' + str(Boundary_size))
        print('The average distance between a hole and the exterior boundary  is ' + str(avg_hole_boundary_distance))
        print('The running minutes to complete the counting process is ' + str(elapsed_time_G/60.0))




        f, axarr = plt.subplots(nrows=1,ncols=3)
        f.set_figheight(5)
        f.set_figwidth(15)

        plt.sca(axarr[0]); 
        plt.scatter([item[0] for item in B_t_vertex_coord], [item[1] for item in B_t_vertex_coord])
        plt.scatter([item[0] for item in hole_list_coord], [item[1] for item in hole_list_coord], c = 'r')
        plt.title('B_t at time ' + str(check_time))

        plt.sca(axarr[1]); 
        plt.scatter([item[0] for item in boundary_coord], [item[1] for item in boundary_coord],c='g')
        plt.title('Boundary of B_t at time ' + str(check_time))


        plt.sca(axarr[2]); 
        plt.scatter([item[0] for item in boundary_coord], [item[1] for item in boundary_coord],c='g')
        plt.scatter([item[0] for item in exterior_boundary_coord], [item[1] for item in exterior_boundary_coord],c='orange')
        plt.title('Exterior boundary of B_t at time ' + str(check_time))
        plt.show()
    
    return {'N_hole':N_hole,
	    'Total_hole_size':total_size,
            'max_hole_size': max_size, 'max_hole_boundary_distance':max_hole_boundary_distance, 
            'avg_hole_boundary_distance':avg_hole_boundary_distance,
            'Boundary_size': Boundary_size,'time':check_time,'D':D}


def generate_First_Percolation_2d(D, count_time_list, distribution,paras, if_print = 0):
    start_time = time.time()
    #Generate the lattice graph
    ig = igraph.Graph.Lattice([2*D+1, 2*D+1], 1, False, False, False) 
    ##save the mapping from lattice to coordinate
    coord_list = [(x,(-D + x % (2*D+1), -D + int(np.floor(x/(2*D+1))))) for x in range(len(ig.vs))]
    #assign the name to the graph
    ig.vs["name"] =[str(item[1]) for item in coord_list]
    for edge in ig.es:
        if distribution == 'Exponential':
            edge['weight'] = np.random.exponential(paras[0])
        elif distribution == 'Gamma':
            edge['weight'] = np.random.gamma(shape = paras[1], scale= paras[0])
        elif distribution == 'Uniform':
            edge['weight'] = random.uniform(paras[0], paras[1])
        elif distribution == 'Pareto':
            edge['weight'] = np.random.pareto(paras[0])
        else:
            pass
    zero_index = int(((2*D+1)*(2*D+1)+1)/2-1)    
    #Calculate T(0,x) for each x
    shortest_weight = ig.shortest_paths_dijkstra(source=zero_index, target=ig.vs, weights=ig.es['weight'], mode = 'ALL')[0]
    #Create a dictionary to indicate T（0,x) for each x
    T_0_x_dict = dict(zip(coord_list, shortest_weight))
    hole_information_list = []
    for check_time in count_time_list:
        #Generate Bt 
        B_t_vertex_coord = [key[1] for key, value in T_0_x_dict.items() if value <= check_time]
        B_t_vertex_id = [key[0] for key, value in T_0_x_dict.items() if value <= check_time]
        max_norm_updated_0 = max([abs(item[0]) for item in B_t_vertex_coord])
        max_norm_updated_1 = max([abs(item[1]) for item in B_t_vertex_coord])
        D_now = max(max_norm_updated_0,max_norm_updated_1)
        #consider a larger box
        D_now = D_now + 3
        #Not hit the boundary
        if D_now <= D:
            hole_result = count_hole_2d(ig,check_time,T_0_x_dict,D_now,B_t_vertex_coord,B_t_vertex_id, if_print)
            hole_information_list.append(hole_result)
    ## plot holes information vs time
    #t_list = [item['time'] for item in hole_information_list]
    #Total_Size_list = [item['Total_hole_size'] for item in hole_information_list]
    #Boundary_size_list = [item['Boundary_size'] for item in hole_information_list]
    #N_hole_list = [item['N_hole'] for item in hole_information_list]
    #avg_hole_boundary_distance_list = [item['avg_hole_boundary_distance'] for item in hole_information_list]

    
    if if_print == 1:
        plt.scatter(t_list, Total_Size_list)
        plt.title(str(distribution) + ': Total hole size vs time')
        plt.xlabel('Time')
        plt.ylabel('Total hole size')
        plt.show()

        plt.scatter(t_list, Boundary_size_list)
        plt.title(str(distribution) + ': Boundary size vs time')
        plt.xlabel('Time')
        plt.ylabel('Total boundary size')
        plt.show()

        plt.scatter(t_list, N_hole_list)
        plt.title(str(distribution) +  ': Number of holes vs time')
        plt.xlabel('Time')
        plt.ylabel('Number of holes')
        plt.show()

        plt.scatter(t_list, avg_hole_boundary_distance_list)
        plt.title(str(distribution) +  ': Average distance between a hole and boundary vs time')
        plt.xlabel('Time')
        plt.ylabel('Average distance between a hole and boundary')
        plt.show()   
    
    
    elapsed_time_B_t = time.time() - start_time
    print('The total running minutes of B_t is ' + str(elapsed_time_B_t/60.0))
    return {'weight_distribution':distribution,'parameter':paras,'hole_information_list': hole_information_list}


# In[ ]:



time_list = list(range(3000, 50500, 500))    
def simulation_FPP(i):
    print (str(i) + ' start.')
    np.random.seed(i+1000)
    result_ffp= generate_First_Percolation_2d(D = 1500, count_time_list = time_list, distribution = 'Pareto', paras = [0.1])
    joblib.dump(result_ffp, 'result_ffp_20201111_data_' + str(i)+'.pkl')
    print (str(i) + ' complete.')
    
 
 

if __name__ == "__main__":
    assert len(sys.argv) > 1 # Make sure an argument is given
    job_index = int(sys.argv[1]) # convert the CLI to an integer

     dir = '/nv/hp16/yma412/data/1111'
    os.chdir(dir)

    # Run the simulation with the index
    simulation_FFP(job_index)
