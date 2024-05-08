import tkinter as tk
import random
from modell import *

class TrafficLightGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Lights")
        self.current_lane_index = 0
        self.traffic_lights = []
        self.timers = []
        self.traffic_time = 0
        self.vehicle_num = []
        self.num_of_vehicles = 0

        # Create labels for lanes
        self.lanes = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]
        self.lane_labels = []
        for i, lane in enumerate(self.lanes):
            label = tk.Label(root, text=lane, font=("Helvetica", 12))
            label.grid(row=0, column=i, padx=10)
            self.lane_labels.append(label)
        
        for i in range(len(self.lane_labels)):
            timer_label = tk.Label(root, text=0, font=("Helvetica", 12))
            timer_label.grid(row=3, column=i, padx=10)
            self.timers.append(timer_label)

        # Create traffic lights
        for i in range(len(self.lanes)):
            light = self.create_traffic_light(row=1, column=i)
            self.traffic_lights.append(light)

        self.count_label = tk.Label(root, text=0, font=("Helvetica", 12))
        self.count_label.grid(row = 4, column=3)

    def create_traffic_light(self, row, column):
        traffic_light = tk.Canvas(self.root, width=40, height=100, bg="black", highlightthickness=0)
        traffic_light.create_oval(10, 10, 30, 30, fill="red", outline="")
        traffic_light.create_oval(10, 40, 30, 60, fill="gray", outline="")
        traffic_light.create_oval(10, 70, 30, 90, fill="gray", outline="")
        traffic_light.grid(row=row, column=column, pady=10)
        return traffic_light
    
    def start_light(self):
        self.change_light()

        self.num_of_vehicles = random.randint(5,15)
        count_string = "Vehicle count: " + str(self.num_of_vehicles)
        self.count_label.config(text=count_string)

        self.vehicle_num.append(self.num_of_vehicles)
        print(self.vehicle_num)
        self.traffic_time = int(vehicle_count(self.vehicle_num)) * 1000
        self.vehicle_num = []
        print(self.traffic_time//1000)
        self.timers[self.current_lane_index].config(text=self.traffic_time//1000)
        self.timers[(self.current_lane_index+1)%4].config(text=0)
        self.timers[(self.current_lane_index+2)%4].config(text=0)
        self.timers[(self.current_lane_index+3)%4].config(text=0)

        self.root.after(self.traffic_time + 1000, self.start_light)  # Change light every 3 seconds
    
    def change_light(self):
        # Set all lights to red
        for light in self.traffic_lights:
            light.itemconfigure(1, fill="red")
            light.itemconfigure(2, fill="gray")  # Hide yellow
            light.itemconfigure(3, fill="gray")  # Hide green

        self.root.after(2000)
        self.traffic_lights[self.current_lane_index].itemconfigure(1, fill="gray")  # Hide red
        self.traffic_lights[self.current_lane_index].itemconfigure(2, fill="yellow")

        # Schedule the next change after a delay of 2000 milliseconds
        self.root.after(3000, self.change_to_green)
           
        
        #self.traffic_lights[next_lane_index].itemconfigure(1, fill="black")  # Hide red
        #self.traffic_lights[next_lane_index].itemconfigure(2, fill="yellow")  # Show yellow
    def change_to_green(self):
        self.traffic_lights[self.current_lane_index].itemconfigure(2, fill="gray")
        self.traffic_lights[self.current_lane_index].itemconfigure(3, fill="green")
        next_lane_index = (self.current_lane_index + 1) % len(self.traffic_lights)
        self.current_lane_index = next_lane_index


        
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")
    app = TrafficLightGUI(root)
    app.start_light()
    root.mainloop()
