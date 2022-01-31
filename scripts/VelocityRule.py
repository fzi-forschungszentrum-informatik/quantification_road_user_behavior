import os

from matplotlib import lines
import math
import numpy as np
import itertools
import math
import uuid
import time
import argparse
import sys
import multiprocessing as mp

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from IPython import display

from IPython.display import HTML
import itertools

from waymo_open_dataset.protos import scenario_pb2
sys.path.insert(1, '../python_scripts/') 
from Helper import Helper as Helper
from CreateDistribution import CreateDistribution

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class VelocityRule:

    def __init__(self, path, max_dist, step_sequence=10, max_files=None, min_speed=0.85, save_freq=5, save_detail=True):
        self.path = path
        self.max_dist = max_dist
        self.step_sequence = step_sequence
        self.max_files = max_files
        self.save_freq = save_freq
        self.min_speed = min_speed

        self.save_detail = save_detail
        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
        # os.path.expanduser('~') + 
        self.root_path = "../data/results/Velocity_rule_" + timestr + "/"
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

        self.line_graph = None
   
    def get_root_path(self):
        return self.root_path

    def toVelocity(self, x_vel=np.nan, y_vel=np.nan):
        velocity = (x_vel**2 + y_vel**2)**0.5
        # originaly in meters/second -> km/h
        velocity = velocity * 60. * 60 / 1000.
        return velocity
    
    def formatVelocity(self, state_list):
        result_state_list = []
        for state in state_list:
            car_list = []
            for car in state:
                if len(car) > 1:
                    p_center = np.array([car[0], car[1]])
                    velocity = self.toVelocity(car[7], car[8])
                    car_list.append(np.array([p_center, velocity]))
                else:
                    car_list.append(np.array([np.array([np.nan, np.nan]), np.nan]))
            result_state_list.append(np.array(car_list))
        return result_state_list

    def euclid_dist(self, v, w):
        return ((v[0] - w[0])**2 + (v[1] - w[1])**2) ** 0.5

    # no root -> speed up
    def dist2(self, v, w):
        return (v[0] - w[0])**2 + (v[1] - w[1])**2

    # returns closest point to p that is on the line between v and w (only between, including v,w)
    def closest_point(self, p, v, w):
        l2 = self.dist2(v,w)
        if l2 == 0: return v
        t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
        t = max(0, min(1, t))
        x_p =  v[0] + t * (w[0] - v[0])
        y_p =  v[1] + t * (w[1] - v[1])

        return np.array([x_p, y_p])

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

    def get_waymo_paths(self):
        path_list = []
        # FILENAME = "/disk/ml/datasets/waymo/motion/scenario/"

        FILENAME = self.path
        for root, dirs, files in os.walk(os.path.abspath(FILENAME)):
            for file in files:
                path_list.append(os.path.join(root, file))
        
        return path_list

    # def get_waymo_paths(self):
    #     path_list = []
    #     FILENAME = "/disk/ml/datasets/waymo/motion/scenario/"
    #     # FILENAME = self.path
    #     for root, dirs, files in os.walk(os.path.abspath(FILENAME)):
    #         for file in files:
    #             path_list.append(os.path.join(root, file))
    #     return path_list


    def get_violations(self, car_list, lane_graph):
        violation_lines = []
        flag_list = []
        degree_list = []
        speed_limit_list = []

        for x in range(len(car_list)):
            # checks whether the driver does not exist in current state
            if not np.isnan(car_list[x][0][0]):
                shortest_dist = 9999999999
                lane_index = 0
                p_center = car_list[x][0]
                p_lane = None

                for y in range(len(lane_graph)):
                    polyline = lane_graph[y][0]

                    for i in range(len(polyline)):
                        temp = np.array([polyline[i][0], polyline[i][1]])
                        distance = self.euclid_dist(p_center, temp)
                        if distance < shortest_dist:
                            shortest_dist = distance
                            lane_index = y
                            p_lane = temp
                

                shortest_line = np.array([p_center, p_lane])

                car_speed = car_list[x][1]
                # original in miles per hour but already transformed in Helper class
                speed_limit = lane_graph[lane_index][1]

                rule_degree = np.nan
                flag = None # no violation = 0, violation = 1, no speed_limit = 2,  too high distance_to_lane = 3, below_min_speed_percentage = 4
                if car_speed / speed_limit > self.min_speed:
                    if shortest_dist <= self.max_dist:
                        if not np.isnan(speed_limit):
                            rule_degree = speed_limit / car_speed
                            if rule_degree > 1.0:
                                rule_degree = 1.0
                                flag = 0
                            else:
                                flag = 1
                        else:
                            flag = 2
                    else:
                        flag = 3
                else:
                    flag = 4

                flag_list.append(flag)
                violation_lines.append(shortest_line)
                degree_list.append(rule_degree)
                speed_limit_list.append(speed_limit)

            else:
                flag_list.append(np.nan)
                degree_list.append(np.nan)
                speed_limit_list.append(np.nan)
                
        return np.array(violation_lines), np.array(degree_list), np.array(flag_list), np.array(speed_limit_list)



    def get_all_violations(self, state_list, lane_graph):
        rule_stack = []
        violation_list = []
        flag_list = []
        speed_limit_list = []

        for x in range(len(state_list)):
            violations, rules, flags, speed_limit = self.get_violations(state_list[x], lane_graph)
            violation_list.append(violations)
            rule_stack.append(rules)
            flag_list.append(flags)
            speed_limit_list.append(speed_limit)
        
       
        return violation_list, rule_stack, flag_list, speed_limit_list


    def save_states(self, rule_values, state_vel_list, speed_limit_list, score_list):
        for x in range(len(rule_values)):
            velocity_list = state_vel_list[x][:,1]
            tableu = np.array([rule_values[x], speed_limit_list[x], velocity_list]).T

            np.savetxt(self.scenario_path + 'state_' + self.convert_Index(x+1) +'.csv', tableu, delimiter=',', header="Rule value,Speed limit(km/h),Velocity(km/h)", comments="")

        np.savetxt(self.scenario_path + 'Average_rule_score.csv', score_list, delimiter=',')

    def convert_Index(self, index):
        index = str(index)
        if len(index) == 2:
            index = "0" + index
        elif len(index) == 1:
            index = "00" + index

        return str(index)

    def rule_Of_scenario(self, vehicle_list, lane_graph, test_mode):
        # transforms waymo_structure into list of states.
        # a state is a list of cars: [p_x{0}, p_y{1}, p_z{2}, length{3}, width{4}, height{5}, heading{6}, vel_x{7}, vel_y{8}]
        all_state_list = Helper.create_state_list(vehicle_list, self.step_sequence)
        # state list of [car_center, velocity]
        state_list = self.formatVelocity(all_state_list)
        # load only roadlines since they solely contain speed limits
        # lane_graph = Helper.create_lane_graph(lane_graph)
        # if len(lane_graph) == 0: # there exists an empty lanegraph :p
        #     return None
        
        # calucalte violations
        violation_lines, rule_values, flag_list, speed_limit_list = self.get_all_violations(state_list, lane_graph)
        
        # calculate rule score of every driver in this scenario
        rule_values_T = np.array(rule_values).T
        score_list = []
        for driver in rule_values_T:
            score = self.average_ex_nan(driver)
            if score == None:
                score_list.append(np.nan)
            else:
                score_list.append(score)


        # save every state of the scenario
        if self.save_detail and not test_mode:
            self.save_states(rule_values, state_list, speed_limit_list, np.array(score_list))
       
        # return differs inside testmode
        if (test_mode):
            state_box_list = self.calc_boxes(all_state_list)
            return np.array(rule_values), violation_lines, state_list, state_box_list, np.array(score_list), lane_graph, flag_list
        else:
            return np.array(score_list)

    def rule_Of_file(self, path):
        score_list = []
        scenario_data = Helper.load_Data(path)

        file_name = path.split("/")
        self.file_path = self.storage_path + file_name[-2] + "." + file_name[-1] + "/"
        if not os.path.isdir(self.file_path) and self.save_detail:
            os.mkdir(self.file_path)

        # for x in range(4):
        for x in range(len(scenario_data)):

            self.scenario_path = self.file_path + "scenario_" + self.convert_Index(x+1) + "/"
            if not os.path.isdir(self.scenario_path) and self.save_detail:
                os.mkdir(self.scenario_path)

            scenario = scenario_data[x]
            total_tracks = scenario.tracks
            lane_graph = list(scenario.map_features)

            type_id = "1" # Id 1 == car
            vehicle_list = []

            for track in total_tracks:
                if str(track.object_type) == type_id:
                    vehicle_list.append(track)
            
            scenario_score = self.rule_Of_scenario(vehicle_list, lane_graph, False) # rule score of every driver among one scenario
            scenario_score = np.array(scenario_score)
            scenario_score = self.average_ex_nan(scenario_score) # average rule score among scenario
            if not scenario_score == None:
                score_list.append(scenario_score)
            
        score_list = np.array(score_list)
        number_of_scenarios = len(score_list)
        score = np.sum(score_list) / number_of_scenarios # average rule score among the file
        return score, number_of_scenarios

    # acceleration purpose only
    def for_parallel(self, path, index):
        degree, num_scenario = self.rule_Of_file(path)
        print("File " + str(index+1) + " done")
        return  degree, num_scenario


    def rule_of_waymo(self):
        number_of_scenarios = 0
        start_time = time.time()
        # path_list = self.get_waymo_paths()
        with open('../scripts/paths.txt') as f:
            path_list = f.readlines()
        degree_list = []

        if self.max_files == None:
            self.max_files = len(path_list)
        
        if not os.path.isdir(self.root_path):
            os.mkdir(self.root_path)
        
        if not os.path.isdir(self.storage_path) and self.save_detail:
            os.mkdir(self.storage_path)

        
        print("######## Start ########")
        pool = mp.Pool(mp.cpu_count())

        results = []
        results = pool.starmap(self.for_parallel, [(path_list[x], x) for x in range(self.max_files)])
        # results = pool.starmap(self.for_parallel, [(x, scenario_data[x]) for x in range(240)])

        pool.close()

        for result in results:
            if not result[0] == None:
                degree_list.append(result[0])
                number_of_scenarios = number_of_scenarios + result[1]
        
        total_degree = np.sum(degree_list) / len(degree_list)
        # # for x in range(19,self.max_files):
        # for x in range(len(path_list)):
        #     degree, num_scenario = self.rule_Of_file(path_list[x])
        #     number_of_scenarios = number_of_scenarios + num_scenario
        #     degree_list.append(degree)
        #     total_degree = np.sum(degree_list) / len(degree_list)
        #     print("File " + str(x+1) + "/" + str(self.max_files) + " | Score of current file: " + str(degree) + " | Total degree: " + str(total_degree))
        #     if x % self.save_freq == 0 and not x == 0:
        #         print("Saving score...")
        #         self.save_score(total_degree, start_time, x+1, number_of_scenarios, False)
        #         print("Saving done")
        
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

        file = open(self.root_path + "Speed_limit_rule_degree.txt", "w")
        file.write("Average score: " + str(total_degree) + " (" + str(num_files) + " files with " + str(number_of_scenarios) + " individual scenarios)")
        file.write("\nTime elapsed: " + str(time_elapsed) + " hours")
        file.write("\nSettings: Max_Distance = " + str(self.max_dist) + " Step_sequence = " + str(self.step_sequence) + " Min speed = " + str(self.min_speed))
        file.close() 


# -------------------------------------------## Testing ##----------------------------------------------
    
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

    # returns the index of the worst driver and his inital position
    def get_point_of_interest(self, state_list, driver_score_list):
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
        for state in state_list:
            car = state[driver_index]
            if len(car) > 1:
                point = np.array([car[0][0], car[0][1]])
                break
        return point, driver_index


    def generate_video(self, state_scenario_score, state_box_list, state_violation_lines, driver_score, render_speed, lane_graph, state_flag_list, plot_connection):
        color_list = ["b","g", "y", "m", "c", "k", "r"]

        focus_p, focus_index = self.get_point_of_interest(state_box_list, driver_score)

        fig = plt.figure(1, figsize=(12,8))
        gs = GridSpec(1, 1, figure=fig)
        # #simulation = fig.add(221, xlim = (0,70), ylim = (0,30))
        simulation = fig.add_subplot()

        def animate_simulation(index): 
            simulation.clear()
            
            state = state_box_list[index]
            viol_list = state_violation_lines[index]
            flag_list = state_flag_list[index]



            for x in range(len(state)):
                car = state[x]
                if len(car) > 1:
                    box = np.array(car).T
                    condition = flag_list[x] # no violation = 0, violation = 1, no speed_limit = 2,  too high distance_to_lane = 3
                    if x == focus_index:
                        simulation.plot(box[0,:], box[1,:], linestyle="--", color="#DC267F", linewidth=1.5)
                    else:
                        if condition == 0:
                            simulation.plot(box[0,:], box[1,:], linestyle="solid", color="#785EF0", lw=1.3)
                        if condition == 1:
                            simulation.plot(box[0,:], box[1,:], linestyle="solid", color="#DC267F", lw=1.3)
                        if condition == 2:
                            simulation.plot(box[0,:], box[1,:], linestyle="solid", color="y", lw=0.9)
                        if condition == 3:
                            simulation.plot(box[0,:], box[1,:], linestyle="solid", color="black", lw=0.9)
                        if condition == 4:
                            simulation.plot(box[0,:], box[1,:], linestyle="solid", color="#FFB000", lw=0.8)

            
            if plot_connection:
                for line in viol_list:
                    p_line = line.T
                    simulation.plot(p_line[0,:], p_line[1,:], linestyle="--", color="m", lw=0.75)

            for lane in lane_graph:
                polyline = lane[0][:,:2].T
                simulation.plot(polyline[0,:], polyline[1,:], linestyle=(0, (40, 110)), color="k", lw=0.15)

            for line in self.line_graph:
                polyline = line[1][:,:2].T
                simulation.plot(polyline[0,:], polyline[1,:], linestyle="solid", color="k", lw=0.48)


            legend_elements = [Patch(facecolor='white', ls="solid", edgecolor='#785EF0', label='No violation'),
                   Patch(facecolor='white', ls="solid", edgecolor='#DC267F', label='Violation'),
                #    Patch(facecolor='white', ls="solid", edgecolor='y', label='No speed_limit'),
                #    Patch(facecolor='white', ls="solid", edgecolor='black', label='Extending distance'),
                   Patch(facecolor='white', ls="--", edgecolor='#DC267F', label='Worst Driver'),
                   Patch(facecolor='white', ls="solid", edgecolor='#FFB000', label='Below min_speed'),
                   Line2D([0], [0], color='k', linestyle="--", label='Center line', lw=1.5),
                   Line2D([0], [0], color='k', linestyle="solid", label='Lane', lw=1.5)
                   ]
            simulation.legend(handles=legend_elements, loc="upper left", prop={'size': 13})
            
            simulation.axes.get_xaxis().set_ticklabels([])
            simulation.axes.get_yaxis().set_ticklabels([])

            x_pos = focus_p[0]
            y_pos = focus_p[1]
            simulation.set_xlim(x_pos - 50, x_pos + 50)
            simulation.set_ylim(y_pos - 50, y_pos + 50)
            # simulation.set_title("Rule: Speed limit", fontsize=17)


            fig.savefig("test" + str(index) + ".svg", bbox_inches='tight', pad_inches = 0)
            # return simulation
        
        
        def animate_all(index):
            sim = animate_simulation(index)
            return sim
        


        

        #current_rule = fig.add_subplot(222, xlim = (0,70), ylim = (0,1.5))
        #bar_rule = fig.add_subplot(223, xlim = (0,5), ylim = (0,1.5))

        animation_1 = animation.FuncAnimation(fig,animate_all,interval=render_speed,frames=len(state_box_list), cache_frame_data=False)
        writervideo = animation.FFMpegWriter(fps=10)

        if not os.path.isdir("../data/sample_plots/"):
            os.mkdir("../data/sample_plots")
        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")

        animation_1.save('../data/sample_plots/Velocity_rule_' + timestr + '.mp4', writer=writervideo)
        video_1 = animation_1.to_html5_video()
        html_code_1 = display.HTML(video_1)
        display.display(html_code_1)




    # creates a video that is, at start,  centered on the worst driver (could be nan at the beginning)
    # random = True -> random file
    def run_test(self, random=True, render_speed=100, plot_connection=False):
        self.step_sequence = 1 # force rendering every 100ms to ensure proper video quality
        # path = "/disk/ml/datasets/waymo/motion/scenario/training/training.tfrecord-00720-of-01000"
        path = "/disk/ml/datasets/waymo/motion/scenario/testing_interactive/testing_interactive.tfrecord-00120-of-00150" #best
        # path = "/disk/ml/datasets/waymo/motion/scenario/testing_interactive/testing_interactive.tfrecord-00068-of-00150/"
        if random:
            path_list = self.get_waymo_paths()
            i_random = np.random.randint(0, high=len(path_list))
            path = path_list[i_random]
        print(path)
        scenario_data = Helper.load_Data(path)
        scenario = scenario_data[0]
        total_tracks = scenario.tracks
        lane_graph = list(scenario.map_features)
        tt, self.line_graph, edge_graph = Helper.create_road_graph(lane_graph)
        lane_graph = Helper.create_lane_graph(lane_graph)



        type_id = "1" # Id 1 == car
        vehicle_list = []

        for track in total_tracks:
            if str(track.object_type) == type_id:
                vehicle_list.append(track)

        state_scenario_score, state_violation_lines, state_list, state_box_list, driver_score, lane_graph, flag_list = self.rule_Of_scenario(vehicle_list, lane_graph, True)
        self.generate_video(state_scenario_score, state_box_list, state_violation_lines, driver_score, render_speed, lane_graph, flag_list, plot_connection)
        driver_score = self.average_ex_nan(driver_score)
        print("Total score in this scenario: " + str(driver_score))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/disk/ml/datasets/waymo/motion/scenario/training_20s/")
    parser.add_argument("--dist", type=float, default=10.0)
    parser.add_argument("--min_speed", type=float, default=0.8)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)
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

    vr = VelocityRule(path=args.path, max_dist=args.dist, min_speed=args.min_speed, step_sequence=args.step_size, max_files=args.size, save_freq=args.save_freq, save_detail=s_detail)
    if not sample:       
        vr.rule_of_waymo()
        cr = CreateDistribution("Velocity")
        cr.generate_hist(vr.get_root_path())
    else:
        vr.run_test()  
