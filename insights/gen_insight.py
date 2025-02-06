import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

WD = os.getcwd()

with open(f'{WD}\calibration\screen_spec.json', 'r') as file:
    data = json.load(file)


screen_width = data['width_pixels']
screen_height = data['height_pixels']
screen_cm_width = data['height_cm']
screen_cm_height = data['width_cm']

def distance(x, y):
    return math.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)

def main():
    
    pix_per_cm_width =  screen_width/screen_cm_width
    pix_per_cm_height = screen_height/screen_cm_height


    instruction_df = pd.read_csv(f"{WD}\insights\data\inst.csv")    # need to be configured later
    observation_df = pd.read_csv(f"{WD}\insights\data\obs.csv")

    time_levels = 3    # vgoof, good, ok, 
    time_window = .10  # time_period of a time level, if the    
    time_out = time_levels * time_window

    latency = list()
    hit_distance = list()
    hit_x = list()
    hit_y = list()
    hit_time = list()
                                    # all units in the below block is in cm
    foveal_area_diameter = .015   #
    eyeball_radius = 2.422
    avg_distance_to_screen = 70

    relaxation = 3

    distance_threshold_cm =   foveal_area_diameter/eyeball_radius*avg_distance_to_screen  + relaxation
    # distance_threshold in pixels
    distance_threshold = distance_threshold_cm * pix_per_cm_width     # approximating pixel size using the biggest dimension

    entry = 0
    for i in range(len(instruction_df['time'])):
        for j in range(entry, len(observation_df['time'])):
            
            t1 = instruction_df['time'][i]
            t2 = observation_df['time'][j]

            if t2<t1:
                continue
            if t2 > t1+time_out: 
                entry = i+1
                break


            x1 = instruction_df['x_cordinate'][i]
            x2 = observation_df['x_cordinate'][j]
            x = [x1, x2]

            y1 = instruction_df['y_cordinate'][i]
            y2 = observation_df['y_cordinate'][j]
            y = [y1, y2]
            
            d = distance(x,y)

            if d < distance_threshold and t2>t1:   
                hit_distance.append(d)
                latency.append(t2 - t1)
                hit_x.append(x2)
                hit_y.append(y2)
                hit_time.append(t2)
                entry = j+1            # once an obs datapoint is assosiated with an inst datapoint only the next datapoints of obs shall be considered for future inst points
                break 
            entry = i + 2              # if no hit occurs in the sliding window the entry becomes the next frame parallel to inst datapoint
               
    accuracy = len(hit_distance)/len(instruction_df['time'])*100

    print(f"accuracy: {accuracy}")    

    avg_latency = sum(latency) / len(latency)
    print(f"avg_latency: {avg_latency}")    

    return hit_distance, latency, hit_time, hit_x, hit_y, instruction_df, observation_df      

def animate_points(hit_time, hit_x, hit_y, inst_time, inst_x, inst_y, obs_time, obs_x, obs_y):
    fig, ax = plt.subplots()
    ax.set_xlim(0, screen_width)
    ax.set_ylim(0, screen_height)

    # Plot instruction points
    inst_line, = ax.plot([], [], 'ro', label='Instruction Points')
    # Plot observation points
    obs_line, = ax.plot([], [], 'bo', label='Observation Points', alpha=0.5)
    # Plot hit points
    hit_line, = ax.plot([], [], 'gx', label='Hit Points', markersize=10)  # Green 'x' for hit points

    def init():
        inst_line.set_data([], [])
        obs_line.set_data([], [])
        hit_line.set_data([], [])
        return inst_line, obs_line, hit_line

    def update(frame_time):
        # Find the instruction points that correspond to the current frame time
        inst_frame = [i for i, t in enumerate(inst_time) if t <= frame_time]
        obs_frame = [i for i, t in enumerate(obs_time) if t <= frame_time]
        hit_frame = [i for i, t in enumerate(hit_time) if t <= frame_time]

    
        # Update instruction points (red)
        inst_line.set_data(inst_x[:len(inst_frame)], inst_y[:len(inst_frame)])
        # Update observation points (blue)
        obs_line.set_data(obs_x[:len(obs_frame)], obs_y[:len(obs_frame)])
        # Update hit points (green)
        hit_line.set_data(hit_x[:len(hit_frame)], hit_y[:len(hit_frame)])
        
        return inst_line, obs_line, hit_line

    # Create animation where frames are time-based using the max of inst_time and obs_time
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=sorted(set(inst_time)|set(obs_time)),  # Use unique times from both datasets
        init_func=init,
        blit=True,
        interval=20  # Adjust the speed as necessary
    )

    # Add legend
    ax.legend()

    # Save the animation as a video file
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, 
                    metadata=dict(artist='Me'), 
                    bitrate=2000,    # Increase the bitrate for better quality
                    extra_args=['-vcodec', 'libx264'])  # Use the high-quality codec)
    ani.save(f"{WD}/insights/animation.mp4", writer=writer)

    plt.show()

if __name__ == "__main__":
    hit_accuracy, latency, hit_time, hit_x, hit_y, instruction_df, observation_df = main()

    # Extract instruction and observation points
    inst_time = instruction_df['time']
    inst_x = instruction_df['x_cordinate']
    inst_y = instruction_df['y_cordinate']
    obs_time = observation_df['time']
    obs_x = observation_df['x_cordinate']
    obs_y = observation_df['y_cordinate']

    animate_points(hit_time, hit_x, hit_y, inst_time, inst_x, inst_y, obs_time, obs_x, obs_y)
