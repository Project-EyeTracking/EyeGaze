import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def distance(x, y):
    return math.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)

def main():
    # Get the current working directory
    WD = os.getcwd()

    x_divisions = 2
    y_divisions = 2

    instruction_df = pd.read_csv("D:\MAI\THWS\project\EyeGaze\insights\data\inst.csv")    # need to be configured later
    observation_df = pd.read_csv("D:\MAI\THWS\project\EyeGaze\insights\data\obs.csv")

    time_levels = 3    # vgoof, good, ok, 
    time_window = .10  # time_period of a time level, if the    
    time_out = time_levels * time_window

    latency = list()
    hit_accuracy = list()
    hit_x = list()
    hit_y = list()
    hit_time = list()

    accuracy_threshold = 1930  # maximum accuracy is 2000 that is if the instruction coordinates are followed exactly

    for i in range(len(instruction_df['time'])):
        for j in range(i, len(observation_df['time'])):
            
            t1 = instruction_df['time'][i]
            t2 = observation_df['time'][j]

            if t2 > t1+time_out: 
                break


            x1 = instruction_df['x_cordinate'][i]
            x2 = observation_df['x_cordinate'][j]
            x = [x1, x2]

            y1 = instruction_df['y_cordinate'][i]
            y2 = observation_df['y_cordinate'][j]
            y = [y1, y2]
            
            accuracy = 2000 - distance(x, y)  # maximum accuracy is defined to be 2000
            if accuracy > accuracy_threshold and t2>t1:
                hit_accuracy.append(accuracy)
                latency.append(t2 - t1)
                hit_x.append(x2)
                hit_y.append(y2)
                hit_time.append(t2)
                break

   
    avg_accuracy = sum(hit_accuracy) / len(hit_accuracy)
    avg_accuracy = avg_accuracy/2000*100
    print(f"avg_accuracy: {avg_accuracy}")    

    avg_latency = sum(latency) / len(latency)
    print(f"avg_latency: {avg_latency}")    

    return hit_accuracy, latency, hit_time, hit_x, hit_y, instruction_df, observation_df      

def animate_points(hit_time, hit_x, hit_y, inst_time, inst_x, inst_y, obs_time, obs_x, obs_y):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2560)
    ax.set_ylim(0, 1440)

    # Plot instruction points
    inst_line, = ax.plot([], [], 'ro', label='Instruction Points')
    # Plot observation points
    obs_line, = ax.plot([], [], 'bo', label='Observation Points')
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
    ani.save('EyeGaze/insights/animation.mp4', writer=writer)

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
