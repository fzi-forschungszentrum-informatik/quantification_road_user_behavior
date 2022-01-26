import os
import math
import numpy as np
import itertools
import math
import uuid
import time
import sys

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from IPython.display import HTML
import itertools
import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import scenario_pb2

class Helper(object):

    ##############################################################################################
    ##############################################################################################
    ###
    ### Vehicle methods
    ###
    ##############################################################################################
    ##############################################################################################

    ''' returns an array of scenarios'''
    @staticmethod
    def load_Data(filename):
        dataset = tf.data.TFRecordDataset(filename)
        scenario_data = []
        for data in dataset:
            proto_string = data.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)
            scenario_data.append(proto)

        return scenario_data
    
    
    # returns a state of certain car of one scenario
    # state is [p_x, p_y, p_z, length, width, height, heading, vel_x, vel_y]
    @staticmethod
    def states_to_numpy(state):

        state_list = []
        if str(state.valid) == "True":
            state_list.append(state.center_x)
            state_list.append(state.center_y)
            state_list.append(state.center_z)
            state_list.append(state.length)
            state_list.append(state.width)
            state_list.append(state.height)
            state_list.append(state.heading + math.pi) # waymo radian range = [-pi, pi]. Therfore "+ pi" -> range = [0, 2pi]
            state_list.append(state.velocity_x)
            state_list.append(state.velocity_y)
        else:
            state_list = [np.nan]
        
        return np.array(state_list)

    # returns list of states of one scenario
    # step_sequence = every ... state
    @staticmethod
    def create_state_list(vehicles, step_sequence=1):
        vehicle_list = []
        for x in range(0,len(vehicles[0].states), step_sequence):
            state_list = []
            for y in range(len(vehicles)):
                state = Helper.states_to_numpy(vehicles[y].states[x])
                state_list.append(state)
            vehicle_list.append(state_list)

        return np.array(vehicle_list)



    ''' given a list of state lists from certain vehicles the method plots their trajectories + their roadgraph
        ! invalid states are beiing ignored !
        state_list = list of state lists
        cars = array of displayed cars
        range_low, range_high = area of intrest among the statelist 
        step_size = incrementing points in the area
        scope_x = [low, high] plot x area of interest. None=total graph
        scope_y = [low, high] plot y area of interest. None=total graph
        road_graph = list of polylines. None=no roadgraph
    '''
    @staticmethod
    def plot_vehicle_points(state_list, cars, range_low, range_high, step_size, scope_x, scope_y, road_graph):
        positions_list = len(state_list[0])*[[]]

        for state in state_list:
            for x in range(len(state)):
                car = state[x]
                if len(car) > 1:
                    positions_list[x].append(np.array([car[0], car[1]]))
        
        positions_list = np.array(positions_list)
        print(positions_list[0])
        
        for car in cars:
            position = positions_list[car].T
            plt.plot(position[0,:20], position[1,:20], linestyle="--", marker="o", color="b", ms=5.5)
        
        if not road_graph == None:
            point_list = np.array(road_graph[0][1])
            for x in range(1,len(road_graph)):
                if not len(road_graph[x][1]) == 0:
                    point_list = np.concatenate((point_list, np.array(road_graph[x][1])))
        
            point_list = np.array(point_list[:,:2].T).astype("float32")
            plt.plot(point_list[0,:], point_list[1,:], '.k',alpha=1, ms=0.5)
        
        if not scope_x == None and not scope_y == None:
            plt.xlim(scope_x[0],scope_x[1])
            plt.ylim(scope_y[0],scope_y[1])
        plt.show()

    ##############################################################################################
    ##############################################################################################
    ###
    ### Road_graph methods
    ###
    ##############################################################################################
    ##############################################################################################
        
    # 1 mile = 1.609344 
    @staticmethod
    def miles_in_kilometers(miles):
        return miles * 1.609344


        '''
        ## Lane ###
        Structure: [id,[waypoints], speedlimit, lanetype, interpolating]
        TYPE_UNDEFINED = 0;
        TYPE_FREEWAY = 1;
        TYPE_SURFACE_STREET = 2;
        TYPE_BIKE_LANE = 3;
        interpolating = true,false

        ## Road_line ###
        Structure: [id,[waypoints], lanetype]
        TYPE_UNKNOWN = 0;
        TYPE_BROKEN_SINGLE_WHITE = 1;
        TYPE_SOLID_SINGLE_WHITE = 2;
        TYPE_SOLID_DOUBLE_WHITE = 3;
        TYPE_BROKEN_SINGLE_YELLOW = 4;
        TYPE_BROKEN_DOUBLE_YELLOW = 5;
        TYPE_SOLID_SINGLE_YELLOW = 6;
        TYPE_SOLID_DOUBLE_YELLOW = 7;
        TYPE_PASSING_DOUBLE_YELLOW = 8;

        ## Road_edge ###
        Structure: [id,[waypoints], lanetype]
        TYPE_UNKNOWN = 0;
        TYPE_ROAD_EDGE_BOUNDARY = 1;
        TYPE_ROAD_EDGE_MEDIAN = 2;
        '''
    @staticmethod
    def create_road_graph(total_map):
        lane_graph = []
        line_graph = []
        edge_graph = []
        
        for x in range(len(total_map)):
            chunk = total_map[x]
            
            if not str(chunk.lane) == "":   
                lane = []
                lane.append(chunk.id)
            
                poly_list = chunk.lane.polyline
                lane_point_list = Helper.create_waypoints(poly_list)
                lane.append(lane_point_list)
            
                lane.append(chunk.lane.speed_limit_mph)
                lane.append(chunk.lane.type)
                lane.append(chunk.lane.interpolating)
            
                lane_graph.append(lane)
            
            elif not str(chunk.road_line) == "":
                line = []
                line.append(chunk.id)
            
                poly_list = chunk.road_line.polyline
                line_point_list = Helper.create_waypoints(poly_list)
                line.append(line_point_list)

                line.append(chunk.road_line.type)
            
                line_graph.append(line)
        
            elif not str(chunk.road_edge) == "":
                edge = []
                edge.append(chunk.id)
            
                poly_list = chunk.road_edge.polyline
                edge_point_list = Helper.create_waypoints(poly_list)
                edge.append(edge_point_list)
            
                edge.append(chunk.road_edge.type)
            
                edge_graph.append(edge)

        return np.array(lane_graph), np.array(line_graph), np.array(edge_graph)

    #     '''
    #     ## Lane ###
    #     Structure: [id,[waypoints], speedlimit, left_neighbors, right_neighbors]

    #     ## Road_line ###
    #     Structure: [id,[waypoints]]

    #     ## Road_edge ###
    #     Structure: [id,[waypoints]]
    #     '''
    # @staticmethod
    # def synthesised_graph(total_map):
    #     lane_graph = []
        
    #     for x in range(len(total_map)):
    #         chunk = total_map[x]
            
    #         if not str(chunk.lane) == "":
    #             lane = []
    #             lane.append(chunk.id)
            
    #             poly_list = chunk.lane.polyline
    #             lane_point_list = Helper.create_waypoints(poly_list)
    #             lane.append(lane_point_list)
            
    #             lane.append(chunk.lane.speed_limit_mph)
    #             lane.append(chunk.lane.left_neighbors)
    #             lane.append(chunk.lane.right_neighbors)
            
    #             lane_graph.append(lane)
            
    #         elif not str(chunk.road_line) == "":
    #             line = []
    #             line.append(chunk.id)
            
    #             poly_list = chunk.road_line.polyline
    #             line_point_list = Helper.create_waypoints(poly_list)
    #             line.append(line_point_list)
            
    #             lane_graph.append(line)
        
    #         elif not str(chunk.road_edge) == "":
    #             edge = []
    #             edge.append(chunk.id)
            
    #             poly_list = chunk.road_edge.polyline
    #             edge_point_list = Helper.create_waypoints(poly_list)
    #             edge.append(edge_point_list)
            
            
    #             lane_graph.append(edge)

    #     return np.array(lane_graph)

    
    # @staticmethod
    # def find_index_by_id(data, id):
    #     for x in range(len(data)):
    #         if data[x][0] == id:
    #             return x


    # @staticmethod
    # def build_pairs(lane_list):
    #     pairs = []
    #     for lane in lane_list:
    #         if len(lane) == 5: #no edge or border
    #             for neighbor in lane[3]: #left
    #                 line_lane = lane[1][neighbor.self_start_index:neighbor.self_end_index]
    #                 index = Helper.find_index_by_id(lane_list, neighbor.feature_id)
    #                 neighbor_lane = lane_list[index][1][neighbor.neighbor_start_index:neighbor.neighbor_end_index]
    #                 tmp_pair = [line_lane, neighbor_lane, lane[2]]
    #                 # print(str(lane[0]) + " " + str(sorted_lane[index][0]))
    #                 if len(line_lane) > 1 and len(neighbor_lane) > 1:
    #                     pairs.append(tmp_pair)
    #                 else:
    #                     print("Happened")

    #             for neighbor in lane[4]:
    #                 line_lane = lane[1][neighbor.self_start_index:neighbor.self_end_index]
    #                 index = Helper.find_index_by_id(lane_list, neighbor.feature_id)
    #                 neighbor_lane = lane_list[index][1][neighbor.neighbor_start_index:neighbor.neighbor_end_index]
    #                 tmp_pair = [line_lane, neighbor_lane, lane[2]]
    #                 # print(str(lane[0]) + " " + str(sorted_lane[index][0]))
    #                 if len(line_lane) > 1 and len(neighbor_lane) > 1:
    #                     pairs.append(tmp_pair)
    #                 else:
    #                     print("Happened")
        
    #     return pairs
    
    @staticmethod
    def euclid_dist(v, w):
        return math.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)

    # @staticmethod
    # def get_adjacent_pair(pairA, pairB):
    #     union = None
    #     threshi = 0.5
    #     if not pairA[2] == pairB[2]:
    #         return None
        
    #     # print(len(pairA[0]))
    #     leftA1 = pairA[0][0][:2]
    #     leftA2 = pairA[1][0][:2]
    #     rightA1 = pairA[0][-1][:2]
    #     rightA2 = pairA[1][-1][:2]

    #     leftB1 = pairB[0][0][:2]
    #     leftB2 = pairB[1][0][:2]
    #     rightB1 = pairB[0][-1][:2]
    #     rightB2 = pairB[1][-1][:2]

    #     # print(Helper.euclid_dist(rightA1, leftB1))
    #     # if Helper.euclid_dist(leftA1, leftB1) < threshi and Helper.euclid_dist(leftA2, leftB2) < threshi:
    #     #     union = [np.concatenate((pairB[0],(pairA[0]))), np.concatenate((pairB[1],(pairA[1]))), pairA[2]]
    #     # elif Helper.euclid_dist(leftA1, leftB2) < threshi and Helper.euclid_dist(leftA2, leftB1) < threshi:
    #     #     union = [leftA1 + leftB2, leftA2 + leftB1, pairA[2]]
    #     if Helper.euclid_dist(leftA1, rightB1) < threshi and Helper.euclid_dist(leftA2, rightB2) < threshi:
    #         union = [np.concatenate((pairB[0],pairA[0])), np.concatenate((pairB[1],pairA[1])), pairA[2]]
    #     # elif Helper.euclid_dist(leftA1, rightB2) < threshi and Helper.euclid_dist(leftA2, rightB1) < threshi:
    #     #     union = [np.concatenate((pairB[1],(pairA[0]))), np.concatenate((pairB[0],(pairA[1]))), pairA[2]]

    #     elif Helper.euclid_dist(rightA1, leftB1) < threshi and Helper.euclid_dist(rightA2, leftB2) < threshi:
    #         union = [np.concatenate((pairA[0],pairB[0])), np.concatenate((pairA[1],pairB[1])), pairA[2]]
    #     # elif Helper.euclid_dist(rightA1, leftB2) < threshi and Helper.euclid_dist(rightA2, leftB1) < threshi:
    #     #     union = [np.concatenate((pairA[0],pairB[1])), np.concatenate((pairA[1],pairB[0])), pairA[2]]
    #     # elif Helper.euclid_dist(rightA1, rightB1) < threshi and Helper.euclid_dist(rightA2, rightB2) < threshi:
    #     #     union = [rightA1 + rightB1, rightA2 + rightB2, pairA[2]]
    #     # elif Helper.euclid_dist(rightA1, rightB2) < threshi and Helper.euclid_dist(rightA2, rightB1) < threshi:
    #     #     union = [rightA1 + rightB2, rightA2 + rightB1, pairA[2]]
        
    #     return union

                
    # @staticmethod
    # def glue_pairs(pair_list):
    #     found_pair = True
    #     current_pairs = pair_list.copy()
    #     chunk_list = []
    #     while(found_pair):
    #         tmp = []
    #         found_pair = False
    #         for x in range(len(current_pairs)):
    #             has_pair = False
    #             for y in range(len(pair_list)):
    #                 new_pair = Helper.get_adjacent_pair(current_pairs[x], pair_list[y])
    #                 if not new_pair == None:
    #                     # print("jap")
    #                     tmp.append(new_pair)
    #                     found_pair = True
    #                     has_pair = True
    #             if has_pair == False:
    #                 chunk_list.append(current_pairs[x])
                
    #         current_pairs = tmp.copy()
        
    #     return chunk_list


    @staticmethod
    def create_lane_graph(total_map):
        lane_graph = []
        
        for x in range(len(total_map)):
            chunk = total_map[x]
            
            if not str(chunk.lane) == "" and not chunk.lane.type == 3:   # 3 := bykelane with no speed_limit
                lane = []
            
                poly_list = chunk.lane.polyline
                lane_point_list = Helper.create_waypoints(poly_list)
                lane.append(lane_point_list)

                #convert miles to kilometers
                speed_limit = Helper.miles_in_kilometers(chunk.lane.speed_limit_mph)
                if speed_limit == 0.0:
                    speed_limit = np.nan
                lane.append(speed_limit)

                if len(lane[0]) > 0:
                    lane_graph.append(lane)
        return np.array(lane_graph)

    @staticmethod
    def create_waypoints(points):
        result = []
        
        for p in points:
            holder = []
            tmp = str(p).split("\n")
            p_x = tmp[0].split(" ")
            holder.append(p_x[1])
            p_y = tmp[1].split(" ")
            holder.append(p_y[1])
            p_z = tmp[2].split(" ")
            holder.append(p_z[1])
            result.append(np.array(holder).astype("float32"))
        
        return np.array(result)

    # @staticmethod
    # def plot_lines(lines):
    #     point_list = np.array(lines[0])
    #     for x in range(1,len(lines)):
    #         if not len(lines[x]) == 0:
    #             point_list = np.concatenate((point_list, np.array(lines[x])))
        
    #     point_list = np.array(point_list[:,:2].T).astype("float32")
    #     plt.plot(point_list[0,:], point_list[1,:], '.k',alpha=1, ms=0.5)
        
    # @staticmethod 
    # def plot_graph(road_graph):
    #     point_list = np.array(road_graph[0][1])
    #     for x in range(1,len(road_graph)):
    #         if not len(road_graph[x][1]) == 0:
    #             point_list = np.concatenate((point_list, np.array(road_graph[x][1])))
        
    #     point_list = np.array(point_list[:,:2].T).astype("float32")
    #     plt.plot(point_list[0,:], point_list[1,:], '.k',alpha=1, ms=0.5)

    # @staticmethod  
    # def plot_total_roadgraph(lane_graph, line_graph, edge_graph, x_dim, y_dim):
    #     lane_list = np.array(lane_graph[0][1])
    #     line_list = np.array(line_graph[0][1])
    #     edge_list = np.array(edge_graph[0][1])
        
    #     for x in range(1,len(lane_graph)):
    #         if not len(lane_graph[x][1]) == 0:
    #             lane_list = np.concatenate((lane_list, np.array(lane_graph[x][1])))
    #     for x in range(1,len(line_graph)):
    #         if not len(line_graph[x][1]) == 0:
    #             line_list = np.concatenate((line_list, np.array(line_graph[x][1])))
    #     for x in range(1,len(edge_graph)):
    #         if not len(edge_graph[x][1]) == 0:
    #             edge_list = np.concatenate((edge_list, np.array(edge_graph[x][1])))
        
    #     lane_list = np.array(lane_list[:,:2].T).astype("float32")
    #     line_list = np.array(line_list[:,:2].T).astype("float32")
    #     edge_list = np.array(edge_list[:,:2].T).astype("float32")
        
    #     plt.plot(lane_list[0,:], lane_list[1,:], '.k',alpha=1, ms=0.5)
    #     plt.plot(line_list[0,:], line_list[1,:], '.r',alpha=1, ms=0.5)
    #     plt.plot(edge_list[0,:], edge_list[1,:], '.b',alpha=1, ms=0.5)
    #     if not x_dim == None and not y_dim == None:
    #         plt.xlim(x_dim)
    #         plt.ylim(y_dim)
    #     plt.show()