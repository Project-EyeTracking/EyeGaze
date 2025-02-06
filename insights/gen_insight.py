import pandas as pd
import os
import math

def distance(x, y):
    return math.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)

def main():

    # Get the current working directory
    WD = os.getcwd()
    
    screen_width = 1080
    screen_height = 720

    x_divisions = 2
    y_divisions = 2

    instruction_df = pd.read_csv("D:\MAI\THWS\project\EyeGaze\insights\data\inst.csv")    # need to be configured later
    observation_df = pd.read_csv("D:\MAI\THWS\project\EyeGaze\insights\data\obs.csv")


    #print(instruction_df)
    #print('\n')
    #print(observation_df)

    time_levels=3    # vgoof, good, ok, 
    time_window = 3  # time_period of  a time level,  if the    
    time_out = time_levels*time_window

    latency = list()
    hit_accuracy = list()
    hit_x = list()
    hit_y = list()
    hit_time = list()

    accuracy_threshold = 1600  # maximum accuracy is 2000 that is if the isntruction coordinates are followed exactly

    for i in range(len(instruction_df['time'])):
        for j in range(i,len(observation_df['time'])):

            if(observation_df['time'][j]>instruction_df['time'][i]+time_out): 
                 break
            
            t1 = instruction_df['time'][i]
            t2 = observation_df['time'][j]

            x1 = instruction_df['x_cordinate'][i]
            x2 = observation_df['x_cordinate'][j]
            x = [x1,x2]

            y1 = instruction_df['y_cordinate'][i]
            y2 = observation_df['y_cordinate'][j]
            y = [y1,y2]
            
            accuracy = 2000 - distance(x,y)  # maximum accuracy is defined to be 2000
            if(accuracy>accuracy_threshold):
                    hit_accuracy.append(accuracy)
                    latency.append(t2-t1)
                    hit_x.append(x2)
                    hit_y.append(y2)
                    hit_time.append(t2)
                    break
    avg_accuracy = sum(hit_accuracy)/len(instruction_df['time'])
    #print(f"hit_accuracy: {hit_accuracy}")
    print(f"avg_accuracy: {avg_accuracy}")    

    avg_latency = sum(latency)/len(instruction_df['time'])
    #print(f"latency: {latency}")
    print(f"avg_latency: {avg_latency}")    
    #print(f"{hit_x},{hit_y}")   
    #print(f"{hit_time}") 

    return hit_accuracy, latency, hit_time, hit_x, hit_y      

          
          
          
          
if __name__ == "__main__":

    hit_accuracy, latency, hit_time, hit_x, hit_y =  main()
