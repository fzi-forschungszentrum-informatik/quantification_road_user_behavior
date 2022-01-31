from lib2to3.pgen2 import driver
import os
import math
import numpy as np
import itertools
import math
import uuid
import time
import argparse
import sys

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from IPython import display
import multiprocessing as mp

from IPython.display import HTML
import itertools

from waymo_open_dataset.protos import scenario_pb2
sys.path.insert(1, '../python_scripts/') 
from Helper import Helper as Helper
from CreateDistribution import CreateDistribution

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class DistanceRule:

    

    def __init__(self, path, latency=3, step_sequence=10, min_speed=5.0, angle_range=20.0, max_files=None, save_freq=5, save_detail=True):
        self.path = path
        self.latency = latency
        self.step_sequence = step_sequence
        self.max_files = max_files
        self.min_speed = min_speed
        self.save_freq = save_freq
        self.angle_range = angle_range
        self.save_detail = save_detail

        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
        # os.path.expanduser('~') 
        self.root_path = "../data/results/Distance_rule_" + timestr + "/"
        self.storage_path = self.root_path + "files/"
        self.file_path = ""
        self.scenario_path = ""

        # worst scenario filter
        self.ws_path = ""
        self.ws_id = None
        self.ws_score = 9999999
        # worst driver filter
        self.wd_path = ""
        self.wd_id = None
        self.wd_score = 9999999


    def get_root_path(self):
        return self.root_path

    def toVelocity(self, x_vel=np.nan, y_vel=np.nan):
        velocity = (x_vel**2 + y_vel**2)**0.5
        # originaly in meters/second -> km/h
        velocity = velocity * 60. * 60 / 1000.
        return velocity

    # returns a list of states 
    # each states contains [vel_x, vel_y, vel_km/h] per car
    def create_velocity_list(self, state_list):
        state_vel_list = []
        for state in state_list:
            car_vel = []
            for car in state:
                if len(car) > 1:
                    vel = self.toVelocity(car[7], car[8])
                    velocity = [car[7], car[8], vel]
                else:
                    velocity = [np.nan, np.nan, np.nan]
                car_vel.append(velocity)
            state_vel_list.append(car_vel)
        state_vel_list = np.array(state_vel_list)

        return state_vel_list

    # input = list of every car in certain state [[p_x{0}, p_y{1}, p_z{2}, length{3}, width{4}, height{5}, heading{6}, vel_x{7}, vel_y{8}]]
    # output = every car_box for each state
    #
    # p[1]-----p[2]
    #   |       |
    #   |       |
    #   |       |
    # p[0]-----p[3]
    # p[4]  |
    #
    #
    def calc_boxes(self, state_list):
        
        state_box_list = []
        pi = math.pi

        for state in state_list:
            box_list = []
            for car in state:
                if len(car) > 1:
                    p_center = np.array([car[0], car[1]])
                    length = car[3] / 2
                    width = car[4] / 2
                    heading = car[6]

                    naive_points = []
                    dir_point_x = length * math.cos(heading)
                    dir_point_y = length * math.sin(heading)
                    dir_vec = np.array([dir_point_x, dir_point_y])
                    if (heading + 0.5*pi) > 2*pi:
                        heading = heading - 2*pi + 0.5*pi
                    else:
                        heading = heading + 0.5*pi
                        
                    p_left_x = width * math.cos(heading)
                    p_left_y = width * math.sin(heading)
                    left_point = p_center + np.array([p_left_x, p_left_y])
                    
                    if (heading + pi) > 2*pi:
                        heading = heading - 2*pi + pi
                    else:
                        heading = heading + pi
                        
                    p_right_x = width * math.cos(heading)
                    p_right_y = width * math.sin(heading)
                    right_point = p_center + np.array([p_right_x, p_right_y])
                    
                    point = left_point - dir_vec
                    naive_points.append(point)
                    point = left_point + dir_vec
                    naive_points.append(point)
                    point = right_point + dir_vec
                    naive_points.append(point)
                    point = right_point - dir_vec
                    naive_points.append(point)
                    point = left_point - dir_vec
                    naive_points.append(point)
                else:
                    naive_points = [np.nan]
            
                box_list.append(naive_points)
            state_box_list.append(box_list)
        return np.array(state_box_list)

    def euclid_dist(self, v, w):
        return math.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)

    # average value of an array with nan
    # None if emtpy
    def average_ex_nan(self, arr):
        values = []
        for x in range(len(arr)):
            if not np.isnan(arr[x]):
                values.append(arr[x])
        if len(values) > 0:
            avg = np.sum(values) / len(values)
        else:
            avg = None

        return avg

    def line_intersection(self, line1, line2):
        def ccw(A,B,C):
            return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

        # Return true if line segments AB and CD intersect
        def intersect(A,B,C,D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
            
        a1 = line1[0]
        a2 = line1[1]
        b1 = line2[0]
        b2 = line2[1]

        p_intersect = [None,None]
        distance = None
        if intersect(a1,a2,b1,b2):
            """ 
            Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
            a1: [x, y] a point on the first line
            a2: [x, y] another point on the first line
            b1: [x, y] a point on the second line
            b2: [x, y] another point on the second line
            """
            s = np.vstack([a1,a2,b1,b2])        # s for stacked
            h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
            l1 = np.cross(h[0], h[1])           # get first line
            l2 = np.cross(h[2], h[3])           # get second line
            x, y, z = np.cross(l1, l2)          # point of intersection
            if z == 0:                          # lines are parallel
                return None, None
            
            p_intersect = np.array([x/z, y/z])
            distance = self.euclid_dist(line1[0], p_intersect)

        return p_intersect, distance

    def is_pointing(self, lineA, lineB):
        angle = self.get_angle(lineA, lineB)
        if angle >= 90.0 - self.angle_range and angle <= 90.0 + self.angle_range:
            return True
        return False

    def get_angle(self, lineA, lineB):
        def dot(vA, vB):
            return vA[0]*vB[0]+vA[1]*vB[1]
        
        # Get nicer vector form
        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        # Get dot prod
        dot_prod = dot(vA, vB)
        # Get magnitudes
        magA = dot(vA, vA)**0.5
        magB = dot(vB, vB)**0.5
        # Get cosine value

        cos_ = dot_prod/magB/magA
        if cos_ > 1 : cos_ = 1
        if cos_ < -1 : cos_ = -1
        # print(lineA)
        # print(lineB)
        # print(dot_prod)
        # print(magA)
        # print(magB)
        # print("++++++++++++")
        # Get angle in radians and then convert to degrees
        angle = math.acos(cos_)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle)%360

        if ang_deg-180>=0:
            # As in if statement
            return 360 - ang_deg
        else: 

            return ang_deg


    def get_violations(self, box_list, vel_list):
        violation_lines = []
        degree_list = []
        dist_list = [] #distance between ego car and the closest car
        max_dist_list = [] #distance from ego car to its future point given the time-delay

        for x in range(len(box_list)):
            # checks whether the driver does not exist in current state or is too slow
            if len(box_list[x])  > 1 and vel_list[x][2] > self.min_speed:
                x_box = box_list[x]
                # x_vel = vel_list[x][0] * self.latency + ((vel_list[x][0] / 10)**2) / 2
                # y_vel = vel_list[x][1] * self.latency + ((vel_list[x][1] / 10)**2) / 2
                x_vel = vel_list[x][0] * self.latency
                y_vel = vel_list[x][1] * self.latency
                head_points = (x_box[0], x_box[3], np.array([abs(x_box[0][0] - x_box[3][0]) , abs(x_box[0][1] - x_box[3][1])]))
                trajectories = []
                for point in head_points:
                    p_future = point + np.array([x_vel, y_vel])
                    trajectory = [point, p_future]
                    trajectories.append(trajectory)
                
                max_dist = self.euclid_dist(trajectories[0][0], trajectories[0][1])

                
                car_dist_list = []
                car_degree_list = []
                for y in range(len(box_list)):
                    if not x == y and len(box_list[y]) > 1:
                        y_box = box_list[y]

                        dist = 9999999999
                        p_final_inter = [None,None]
                        p_head = [None,None]

                        for trajectory in trajectories:
                            # Check if the heading of both cars is similar 
                            if not self.is_pointing(trajectory, [y_box[0], y_box[3]]):
                                break

                            for k in range(len(y_box) - 1):
                                p_intersect, distance = self.line_intersection(trajectory, [y_box[k], y_box[k+1]])
                                if not distance == None:
                                    if distance < dist:
                                        p_final_inter = p_intersect
                                        p_head = trajectory[0]
                                        dist = distance

                        if not p_final_inter[0] == None:
                            degree = dist / max_dist

                            car_dist_list.append(dist)
                            car_degree_list.append(degree)
                            violation_lines.append(np.array([p_head, p_final_inter]))

                if not np.array(car_degree_list).size == 0:
                    smallest_degree = min(np.array(car_degree_list))
                    degree_list.append(smallest_degree)
                    dist_list.append(min(car_dist_list))
                else:
                    degree_list.append(1.0)
                    dist_list.append(np.nan)

                max_dist_list.append(max_dist) # Is always the same for a car
            else:
                degree_list.append(np.nan)
                dist_list.append(np.nan)
                max_dist_list.append(np.nan)


        return np.array(violation_lines), np.array(degree_list), np.array(dist_list), np.array(max_dist_list)

    def get_waymo_paths(self):
        path_list = []
        # FILENAME = "/disk/ml/datasets/waymo/motion/scenario/"
        FILENAME = self.path
        for root, dirs, files in os.walk(os.path.abspath(FILENAME)):
            for file in files:
                path_list.append(os.path.join(root, file))
        return path_list
    
    def get_all_violations(self, state_box_list, state_vel_list):
        rule_stack = []
        violation_stack = []
        dist_stack = []
        max_dist_stack = []
        for x in range(len(state_box_list)):
            violation_lines, rule_values, dist_list, max_dist_list = self.get_violations(state_box_list[x], state_vel_list[x])
            rule_stack.append(rule_values)
            violation_stack.append(violation_lines)
            dist_stack.append(dist_list)
            max_dist_stack.append(max_dist_list)

        
        # rule_stack = np.array(rule_stack).T
        return np.array(violation_stack), np.array(rule_stack), np.array(dist_stack), np.array(max_dist_stack)
    
    def save_states(self, rule_values, dist_list, state_vel_list, vel_list, score_list):
        for x in range(len(rule_values)):
            velocity_list = vel_list[x][:,2]
            tableu = np.array([rule_values[x], dist_list[x], state_vel_list[x], velocity_list]).T

            np.savetxt(self.scenario_path + 'state_' + self.convert_Index(x+1) +'.csv', tableu, delimiter=',', header="Rule value,Distance(Closest Car)(m),Min distance given latency(m),Velocity(km/h)", comments="")

        np.savetxt(self.scenario_path + 'Average_rule_score.csv', score_list, delimiter=',')

    def convert_Index(self, index):
        index = str(index)
        if len(index) == 2:
            index = "0" + index
        elif len(index) == 1:
            index = "00" + index

        return str(index)
    
    def rule_Of_scenario(self, vehicle_list, test_mode):
        # transforms waymo_structure into list of states.
        # a state is a list of cars: [p_x{0}, p_y{1}, p_z{2}, length{3}, width{4}, height{5}, heading{6}, vel_x{7}, vel_y{8}]
        state_list = Helper.create_state_list(vehicle_list, self.step_sequence)

        # create body(boxes) of each car
        state_box_list = self.calc_boxes(state_list)
        
        # extract velocity -> generate vel_list
        state_vel_list = self.create_velocity_list(state_list)
        
        # calucalte violations
        violation_lines, rule_stack, dist_stack, max_dist_stack = self.get_all_violations(state_box_list, state_vel_list)

        
        # calculate rule score of every driver in this scenario
        rule_stack = rule_stack.T
        score_list = []
        for driver in rule_stack:
            score = self.average_ex_nan(driver)
            if score == None:
                score_list.append(np.nan)
            else:
                score_list.append(score)
        
        # save every state of the scenario
        if self.save_detail and not test_mode:
            self.save_states(rule_stack.T, dist_stack, max_dist_stack, state_vel_list, np.array(score_list))

        # return differs inside testmode
        if (test_mode):
            return np.array(rule_stack), violation_lines, state_list, state_box_list, np.array(score_list), state_vel_list
        else:
            return np.array(score_list)
    
    def rule_Of_file(self, path):
        score_list = []
        scenario_data = Helper.load_Data(path)

        file_name = path.split("/")
        self.file_path = self.storage_path + file_name[-2] + "." + file_name[-1] + "/"
        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)

        pool = mp.Pool(mp.cpu_count())

        results = []
        results = pool.starmap(self.for_parallel, [(x, scenario_data[x]) for x in range(len(scenario_data))])
        # results = pool.starmap(self.for_parallel, [(x, scenario_data[x]) for x in range(240)])

        pool.close()

        for result in results:
            if not result == None:
                score_list.append(result)

        score_list = np.array(score_list)
        #print(score_list)
        number_of_scenarios = len(score_list)
        score = np.sum(score_list) / number_of_scenarios # average rule score among the file
        return score, 
        
    def mock_rule_Of_file(self, path):
        score_list = []
        scenario_data = Helper.load_Data(path)

        file_name = path.split("/")
        self.file_path = self.storage_path + file_name[-2] + "." + file_name[-1] + "/"
        if not os.path.isdir(self.file_path) and self.save_detail:
            os.mkdir(self.file_path)

        pool = mp.Pool(mp.cpu_count())

        results = []
        results = pool.starmap(self.mock_for_parallel, [(x, scenario_data[x]) for x in range(len(scenario_data))])
        # results = pool.starmap(self.for_parallel, [(x, scenario_data[x]) for x in range(240)])

        pool.close()

        for result in results:
            if not result[0] == None:
                score_list.append(result[0])
                if not result[1] == None and result[1] < self.ws_score:
                    self.ws_score = result[1]
                    self.ws_id = result[2]
                    self.ws_path = result[3]
                if result[4] < self.wd_score:
                    self.wd_score = result[4]
                    self.wd_id = result[5]
                    self.wd_path = result[6]

        score_list = np.array(score_list)
        #print(score_list)
        number_of_scenarios = len(score_list)
        score = np.sum(score_list) / number_of_scenarios # average rule score among the file
        return score, number_of_scenarios
    
    # acceleration purpose only
    def mock_for_parallel(self, index, scenario):
        self.scenario_path = self.file_path + "scenario_" + self.convert_Index(index+1) + "/"
        if not os.path.isdir(self.scenario_path) and self.save_detail:
            os.mkdir(self.scenario_path)

        total_tracks = scenario.tracks
        type_id = "1" # Id 1 == car
        vehicle_list = []

        for track in total_tracks:
            if str(track.object_type) == type_id:
                vehicle_list.append(track)
        
        score_list = self.rule_Of_scenario(vehicle_list, False) # rule score of every driver among one scenario
        score_list = np.array(score_list)
        scenario_score = self.average_ex_nan(score_list) # average rule score among scenario

        # worst scenario filter
        ws_path = ""
        ws_id = None
        ws_score = 9999999
        # worst driver filter
        wd_path = ""
        wd_id = None
        wd_score = 9999999

        if not scenario_score == None and scenario_score < self.ws_score:
            ws_score = scenario_score
            ws_id = index
            ws_path = self.file_path

        tmp = []
        for x in range(len(score_list)):
            if not np.isnan(score_list[x]):
                tmp.append(score_list[x])
                        
        if len(tmp) > 0:
            tmp = np.array(tmp)
            worst_driver_score = min(tmp)
            
            if worst_driver_score < self.wd_score:
                wd_score = worst_driver_score
                wd_id = index
                wd_path = self.file_path

        return scenario_score, ws_score, ws_id, ws_path, wd_score, wd_id, wd_path


# acceleration purpose only
    def for_parallel(self, index, scenario):
        self.scenario_path = self.file_path + "scenario_" + self.convert_Index(index+1) + "/"
        if not os.path.isdir(self.scenario_path):
            os.mkdir(self.scenario_path)

        total_tracks = scenario.tracks
        type_id = "1" # Id 1 == car
        vehicle_list = []

        for track in total_tracks:
            if str(track.object_type) == type_id:
                vehicle_list.append(track)
        
        score_list = self.rule_Of_scenario(vehicle_list, False) # rule score of every driver among one scenario
        score_list = np.array(score_list)
        scenario_score = self.average_ex_nan(score_list) # average rule score among scenario

        return scenario_score
    
    def rule_of_waymo(self):
        number_of_scenarios = 0
        start_time = time.time()
        path_list = self.get_waymo_paths()
        degree_list = []

        if self.max_files == None:
            self.max_files = len(path_list)
        
        if not os.path.isdir(self.root_path):
            os.mkdir(self.root_path)
        
        if not os.path.isdir(self.storage_path) and self.save_detail:
            os.mkdir(self.storage_path)

        print("######## Start ########")
        for x in range(self.max_files):
        #for x in range(len(path_list)):
            degree, num_scenario = self.mock_rule_Of_file(path_list[x])
            number_of_scenarios = number_of_scenarios + num_scenario
            degree_list.append(degree)
            total_degree = np.sum(degree_list) / len(degree_list)

            print("File " + str(x+1) + "/" + str(self.max_files) + " | Score of current file: " + str(degree) + " | Total degree: " + str(total_degree))
            if x % self.save_freq == 0 and not x == 0:
                print("Saving score...")
                self.save_score(total_degree, start_time, x+1, number_of_scenarios, False)
                print("Saving done")
        
        self.save_score(total_degree, start_time, self.max_files, number_of_scenarios, True)

    def save_score(self, total_degree, start_time, num_files, number_of_scenarios, done):
        end_time = time.time()
        time_elapsed = ((end_time - start_time) / 60.0) / 60.0

        total_degree = int(total_degree * 100000)/ 100000.0
        time_elapsed = int(time_elapsed * 100000)/ 100000.0

        if done:
            print("######## Done ########")
            print("Average score: " + str(total_degree) + " (" + str(num_files) + " files with " + str(number_of_scenarios) + " individual scenarios)")
            print("Time elapsed: " + str(time_elapsed) + " hours")

        file = open(self.root_path + "distance_rule_degree.txt", "w")
        file.write("Average score: " + str(total_degree) + " (" + str(num_files) + " files with " + str(number_of_scenarios) + " individual scenarios)")
        file.write("\nTime elapsed: " + str(time_elapsed) + " hours")
        file.write("\nSettings: Latency = " + str(self.latency) + " Step_sequence = " + str(self.step_sequence) + " Min speed = " + str(self.min_speed) + " Angle range = " + str(self.angle_range))
        file.write("\nWorst scenario: path:" + self.ws_path + " index: " +str(self.ws_id) + " score: " + str(self.ws_score))
        file.write("\nWorst driver: path:" + self.wd_path + " index: " + str(self.wd_id) + " score: " + str(self.wd_score))
        file.close() 
        
# -------------------------------------------## Testing ##----------------------------------------------

    # returns the index of the worst driver and his inital position
    def get_point_of_interest(self, state_box_list, driver_score_list):
        print(driver_score_list)
        no_nan_list = []
        for score in driver_score_list:
            if np.isnan(score):
                no_nan_list.append(1.1)
            else:
                no_nan_list.append(score)

        driver_index = np.where(no_nan_list == np.amin(no_nan_list))[0][0]
        print(driver_index)
        point = np.array([0,0])
        for state in state_box_list:
            car = state[driver_index]
            if len(car) > 1:
                point = np.array([car[0][0], car[0][1]])
                break
        return point, driver_index
    
    def get_future_pos(self, state_box_list, state_vel_list):
        state_line_future = []
        for x in range(len(state_box_list)):
            line_future = []
            for y in range(len(state_box_list[x])):
                car = state_box_list[x][y]
                if len(car) > 1:
                    p_center = (car[0] + car[3]) / 2
                    v_vec = np.array([state_vel_list[x][y][0], state_vel_list[x][y][1]]) * self.latency
                    p_future = p_center + v_vec
                    line_future.append(np.array([p_center, p_future]))
            state_line_future.append(np.array(line_future))
        
        return np.array(state_line_future)

    def generate_video(self, state_scenario_score, state_box_list, state_violation_lines, driver_score, render_speed, state_vel_list, lane_graph, line_graph, edge_graph, future, plot_graph):
        color_list = ["b","g", "y", "m", "c", "k", "r"]

        focus_p, focus_index = self.get_point_of_interest(state_box_list, driver_score)
        state_line_future = self.get_future_pos(state_box_list, state_vel_list)

        fig = plt.figure(1, figsize=(12,8))
        gs = GridSpec(1, 1, figure=fig)
        # #simulation = fig.add(221, xlim = (0,70), ylim = (0,30))
        simulation = fig.add_subplot()
        #current_rule = fig.add_subplot(222, xlim = (0,70), ylim = (0,1.5))
        #bar_rule = fig.add_subplot(223, xlim = (0,5), ylim = (0,1.5))

        def animate_simulation(index): 
            simulation.clear()
            
            state = state_box_list[index]
            viol_list = state_violation_lines[index]
            line_future = state_line_future[index]
            
            # print cars
            for x in range(len(state)):
                car = state[x]
                if len(car) > 1:
                    box = np.array(car).T
                    # # print scores attached to cars
                    # if not np.isnan(state_scenario_score[x][index]):
                    #     simulation.text(box[0][0], box[1][0], s=str(int(state_scenario_score[x][index] * 10000)/ 10000.0))
                    # else:
                    #     simulation.text(box[0][0], box[1][0], s="nan")
                    if x == focus_index:
                        simulation.plot(box[0,:], box[1,:], linestyle="solid", color="#DC267F", linewidth=1.3)
                    else:
                        simulation.plot(box[0,:], box[1,:], linestyle="solid", color="#648FFF", linewidth=0.9)

            
            for line in viol_list:
                line = line.T
                simulation.plot(line[0,:], line[1,:], linestyle="solid", color="#DC267F", lw=1.75)

            if future:
                for line in line_future:
                    # line = line.T
                    # simulation.plot(line[0,:], line[1,:], linestyle="solid", color="#DC267F", lw=0.75)
                    simulation.arrow(line[0][0], line[0][1], line[1][0] - line[0][0], line[1][1] - line[0][1], color="#785EF0", width=0.09, head_width= 2.*0.4, head_length=2.3*0.4)

            # if plot_graph:
            #     for lane in lane_graph:
            #         polyline = lane[1][:,:2].T
            #         simulation.plot(polyline[0,:], polyline[1,:], linestyle=(0, (40, 110)), color="k", lw=0.15)
                for line in line_graph:
                    polyline = line[1][:,:2].T
                    simulation.plot(polyline[0,:], polyline[1,:], linestyle="solid", color="k", lw=0.15)
                # for edge in edge_graph:
                #     polyline = edge[1][:,:2].T
                #     simulation.plot(polyline[0,:], polyline[1,:], linestyle="solid", color="k", lw=0.55)

            
            x_pos = focus_p[0]
            y_pos = focus_p[1]
            simulation.set_xlim(x_pos - 50, x_pos + 50)
            simulation.set_ylim(y_pos - 50, y_pos + 50)
            # simulation.set_title("Rule: Distance Keeping")
            simulation.axes.get_xaxis().set_ticklabels([])
            simulation.axes.get_yaxis().set_ticklabels([])

            legend_elements = [Line2D([0], [0], color='#785EF0', label='Position in ' + str(self.latency) + " s", lw=1.5),
                            Line2D([0], [0], color='#DC267F', label='Violation', lw=1.5),
                            Line2D([0], [0], color='k', linestyle="solid", label='Lane', lw=1.)
                            # Patch(facecolor='white', edgecolor='#DC267F', label='Worst Driver', lw=1.5)
                            # Rectangle((0,0),width=2,height=50,color='#648FFF', label=r"$\varphi_1$" + " Distance Keeping"),
                            # Rectangle((0,0),width=2,height=50,color='#FFB000', label=r"$\varphi_2$" + " Speed Limiting")
                            ]
            simulation.legend(handles=legend_elements, loc="upper left", prop={'size': 13})
            

            fig.savefig("test" + str(index) + ".svg", bbox_inches='tight', pad_inches = 0)
            return simulation
        
        
        def animate_all(index):
            sim = animate_simulation(index)
            return sim
        

        animation_1 = animation.FuncAnimation(fig,animate_all,interval=render_speed,frames=len(state_box_list), cache_frame_data=False)
        writervideo = animation.FFMpegWriter(fps=10)

        if not os.path.isdir("../data/sample_plots/"):
            os.mkdir("../data/sample_plots")
        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")

        animation_1.save('../data/sample_plots/Distance_rule_' + timestr + '.mp4', writer=writervideo)
        video_1 = animation_1.to_html5_video()
        html_code_1 = display.HTML(video_1)
        display.display(html_code_1)

    # creates a video that is, at start, centered on the worst driver (could be nan at the beginning)
    # random = True -> random file
    def run_test(self, random=True, render_speed=100, future=True, plot_graph=False):
        self.step_sequence = 1 # force rendering every 100ms to ensure proper video quality
        path = "/disk/ml/datasets/waymo/motion/scenario/validation/validation.tfrecord-00018-of-00150" #best 
        # path = "/disk/ml/datasets/waymo/motion/scenario/training/training.tfrecord-00328-of-01000"
        # path = "/disk/ml/datasets/waymo/motion/scenario/training/training.tfrecord-00641-of-01000"


        if random:
            path_list = self.get_waymo_paths()
            i_random = np.random.randint(0, high=len(path_list))
            path = path_list[i_random]
        print(path)
        scenario_data = Helper.load_Data(path)
        scenario = scenario_data[0]
        
        type_id = "1" # Id 1 == car
        vehicle_list = []
        total_tracks = scenario.tracks
        for track in total_tracks:
            if str(track.object_type) == type_id:
                vehicle_list.append(track)
        

        lane_graph = list(scenario.map_features)
        lane_graph, line_graph, edge_graph = Helper.create_road_graph(lane_graph)


        
        state_scenario_score, state_violation_lines, state_list, state_box_list, driver_score, state_vel_list = self.rule_Of_scenario(vehicle_list, True)
        self.generate_video(state_scenario_score, state_box_list, state_violation_lines, driver_score, render_speed, state_vel_list, lane_graph, line_graph, edge_graph, future, plot_graph)
        driver_score = self.average_ex_nan(driver_score)
        print("Total score in this scenario: " + str(driver_score))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/disk/ml/datasets/waymo/motion/scenario/training_20s/")
    parser.add_argument("--latency", type=float, default=3.0)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--min_speed", type=float, default=5.0)
    parser.add_argument("--angle_range", type=float, default=20.0)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--save_detail", type=str, default="True")
    parser.add_argument("--sample", type=str, default="False")

    args = parser.parse_args()

    s_detail = args.save_detail
    if s_detail == "True" or s_detail == "true":
        s_detail = True
    elif s_detail == "False" or s_detail == "false":
        s_detail = False

    sample = args.sample
    if sample == "True" or sample == "true":
        sample = True
    elif sample == "False" or sample == "false":
        sample = False

    dr = DistanceRule(path=args.path, latency=args.latency, step_sequence=args.step_size, min_speed=args.min_speed, angle_range=args.angle_range, max_files=args.size, save_freq=args.save_freq, save_detail=s_detail)
    if not sample:        
        dr.rule_of_waymo()
        cr = CreateDistribution("Distance")
        cr.generate_hist(dr.get_root_path())
    else:
        dr.run_test()  







