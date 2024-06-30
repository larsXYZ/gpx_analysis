from scipy.interpolate import make_interp_spline
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xmltodict as xml
import numpy as np
import os

def kph2mps(kph):
    return kph/3.6
def mps2kph(kph):
    return kph*3.6

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius_of_earth = 6371  # Radius of earth in kilometers. Use 3958.8 for miles
    distance = radius_of_earth * c

    return distance * 1000  # Convert to meters

def speed_to_pace(speed_kph):
    if speed_kph <= 0:
        return "Invalid speed"
    
    # Convert speed from kilometers per hour to minutes per kilometer
    pace_minutes = 60 / speed_kph
    
    # Format pace_minutes into a string with minutes and seconds
    pace_minutes_int = int(pace_minutes)
    pace_seconds = (pace_minutes - pace_minutes_int) * 60
    pace_seconds_int = int(pace_seconds)
    
    # Return formatted pace string
    return f"{pace_minutes_int}:{pace_seconds_int:02d}"

class TrainingData:
    def __init__(self):
        self.latitude = []
        self.longitude = []
        self.elevation = []
        self.heartrate = []
        self.start_time = None
        self.delta_elev = None
        self.avg_heartrate = None
        self.avg_speed = None
        self.total_distance = None
    
    def close(self):
        
        #Converting to numpy array
        for attr_name in dir(self):
            if isinstance(getattr(self, attr_name), list):
                setattr(self, attr_name, np.array(getattr(self, attr_name)))

        #Calculating total delta elevation
        if self.elevation.any():
            self.delta_elev = 0
            for i in range(1, len(self.elevation)):
                self.delta_elev += np.abs(self.elevation[i, 1])

        #Calculate speeds
        self.speed = []
        self.total_distance = 0
        for i in range(1, len(self.latitude)):
            lat1, lon1 = self.latitude[i-1][1], self.longitude[i-1][1]
            lat2, lon2 = self.latitude[i][1], self.longitude[i][1]
            distance = haversine(lat1, lon1, lat2, lon2)
            self.total_distance += distance
            time_diff = self.latitude[i][0] - self.latitude[i-1][0]
            if time_diff < 0.01:
                continue
            speed = distance / time_diff
            self.speed.append([self.latitude[i][0], speed])
        self.speed = np.array(self.speed)
        self.avg_speed = np.mean(self.speed[:,1])

        #Calculate average heartrate
        if self.heartrate.any():
            self.avg_heartrate = np.mean(self.heartrate[:,1])

        #Creating state vector
        self.state_attrs = ["latitude", "longitude", "speed", "heartrate", "elevation"]
        prelim_state_vectors, self.state_vectors = {}, []
        for i, attr_name in enumerate(self.state_attrs):
            data = getattr(self, attr_name)
            for time, val in data:
                if time not in prelim_state_vectors.keys():
                    prelim_state_vectors[time] = [None] * 5
                prelim_state_vectors[time][i] = val
        prelim_state_vectors = dict(sorted(prelim_state_vectors.items()))
        for key, val in prelim_state_vectors.items():
            self.state_vectors.append([key] + val)
        self.state_vectors = np.array(self.state_vectors)
        for row in range(self.state_vectors.shape[1]): #Backward extrapolation at start
            i = 0
            while i < self.state_vectors.shape[0]-1 and self.state_vectors[i, row] == None:
                i += 1
            else:
                for q in range(0, i):
                    self.state_vectors[q, row] = self.state_vectors[i, row]
        for row in range(self.state_vectors.shape[1]): #Then forward extrapolation
            for i in range(1, self.state_vectors.shape[0]):
                if self.state_vectors[i, row] == None:
                    self.state_vectors[i, row] = self.state_vectors[i-1, row]
        self.state_attrs = ["time"] + self.state_attrs #Adding time to state attributions
        
        #Logging duration
        self.total_duration = self.state_vectors[-1,0]
    
    class Segment:
        def __init__(self, state_vectors, state_attrs, start_time):
            self.start_time = start_time
            self.state_vectors = state_vectors
            self.state_attrs = state_attrs
            self.time_into_run = state_vectors[0,0]
            self.valid = True
            try:
                self.avg_hr = np.average(state_vectors[:, self.state_attrs.index("heartrate")])
                self.avg_speed = np.average(state_vectors[:, self.state_attrs.index("speed")])
            except TypeError:
                self.valid = False

    def splitIntoSegments(self, segm_duration):
        output = []
        n_segments = int(self.total_duration / segm_duration)
        segment_sample_count = int(self.state_vectors.shape[0] * (segm_duration/self.total_duration))
        for i in range(n_segments):
            n_start, n_stop = i * segment_sample_count, (i+1) * segment_sample_count
            segm_start_time = self.start_time + timedelta(seconds=self.state_vectors[n_start, 0])
            output.append(self.Segment(self.state_vectors[n_start:n_stop, :], self.state_attrs, segm_start_time))
        return output


target_folder = r"C:\Users\thehu\Documents\gpx_analysis\gpx"
file_paths = [target_folder+"/"+f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
file_paths = [path for path in file_paths if "RUNNING" in path]

trainings = []
for i, path in enumerate(file_paths):
    with open(path, "r") as file:

        training = TrainingData()

        data = xml.parse(file.read())
        points = data['gpx']['trk']['trkseg']['trkpt']
        
        for point in points:

            try:
                time = datetime.strptime(point['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                time = datetime.strptime(point['time'], '%Y-%m-%dT%H:%M:%SZ')
            
            if training.start_time is None:
                training.start_time = time
                

            delta_time = (time - training.start_time).total_seconds()

            if "@lat" in point.keys():
                training.latitude.append([delta_time, float(point['@lat'])])

            if "@lon" in point.keys():
                training.longitude.append([delta_time, float(point['@lon'])])

            if "ele" in point.keys():
                training.elevation.append([delta_time, float(point['ele'])])

            try:
                training.heartrate.append([delta_time, float(point['extensions']['gpxtpx:TrackPointExtension']['gpxtpx:hr'])])
            except KeyError:
                pass
        
        training.close()
        trainings.append(training)

trainings.sort(key=lambda d: d.start_time)



#Collecting segments
all_segments = []
for training in trainings:
    segments = training.splitIntoSegments(600)
    valid_segments = [segment for segment in segments if segment.valid]
    all_segments = all_segments + valid_segments


#For use later in plotting
months_for_plotting = [(2023, 12), (2024,1), (2024, 2), (2024, 3), (2024, 4), (2024, 5), (2024, 6)]

#Heartrate and pace relationship
max_speed = kph2mps(15)
min_speed = kph2mps(7)
start_time_cutoff = 1800 #Discard the first N seconds of the workout
for month in months_for_plotting:
    subset_segments = [s for s in all_segments  if s.start_time.year == month[0] and s.start_time.month == month[1]]
    subset_segments.sort(key= lambda s : s.avg_speed)
    subset_segments = [s for s in subset_segments if s.avg_speed < max_speed and s.avg_speed > min_speed and s.time_into_run > start_time_cutoff]
    xs, ys = np.array([mps2kph(s.avg_speed) for s in subset_segments]), np.array([s.avg_hr for s in subset_segments])
    plt.scatter(xs, ys, label=f"{month[0]}-{month[1]} ({len(xs)})")
    # Perform polynomial regression
    try:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        plt.plot(xs, p(xs), label=f"{month[0]}-{month[1]} ({len(xs)})")
    except TypeError:
        pass
plt.legend()
plt.grid()
plt.xlabel("Average pace [min:sec per km]")
plt.ylabel("Heartrate [bpm]")
speeds = [i for i in range(1,15)]
plt.xticks(speeds, [speed_to_pace(speed) for speed in speeds])
plt.show()

for month in months_for_plotting:
    subset_trainings = [t for t in trainings 
                        if t.start_time.year == month[0] and t.start_time.month == month[1]]
    speeds = [speed for subset in subset_trainings for speed in list(subset.speed[:,1]) if speed < kph2mps(20)]
    plt.hist(mps2kph(np.array(speeds)), bins=400, label=f"{month[0]}-{month[1]} (avg: {mps2kph(np.average(speeds)):.2f})")
plt.xlabel("Speed [kph]")
plt.legend()
plt.grid()
plt.show()


for month in months_for_plotting:
    subset_trainings = [t for t in trainings 
                        if t.start_time.year == month[0] and t.start_time.month == month[1]]
    hr = [hr for subset in subset_trainings if subset.heartrate.any() for hr in list(subset.heartrate[:,1])]
    plt.hist(np.array(hr), label=f"{month[0]}-{month[1]} (avg: {np.average(hr):.2f})")
plt.xlabel("Average heartrates")
plt.legend()
plt.grid()
plt.show()


start_date = datetime(2023,12,1)
trainings = [training for training in trainings if training.start_time > start_date]
average_heartrates = np.array([[training.start_time.timestamp(), training.avg_heartrate] for training in trainings 
                               if training.avg_heartrate])
# Extracting timestamps and average heartrates
timestamps = average_heartrates[:, 0]
heartrates = average_heartrates[:, 1]
# Perform linear regression
coefficients = np.polyfit(timestamps, heartrates, 1)
poly_function = np.poly1d(coefficients)
# Plotting the data and linear regression line
plt.scatter([datetime.fromtimestamp(ts) for ts in timestamps], heartrates, label='Average Heart Rates')
plt.plot([datetime.fromtimestamp(ts) for ts in timestamps], poly_function(timestamps), color='red', label='Linear Regression')
plt.grid()
plt.xlabel('Date')
plt.ylabel('Average Heart Rate')
plt.legend()
plt.show()



start_date = datetime(2023,12,1)
trainings = [training for training in trainings if training.start_time > start_date]
total_distances = np.array([[training.start_time.timestamp(), training.total_distance] for training in trainings 
                               if training.avg_heartrate])
# Extracting timestamps and distances
timestamps = total_distances[:, 0]
distances = total_distances[:, 1]
# Perform linear regression
coefficients = np.polyfit(timestamps, distances, 1)
poly_function = np.poly1d(coefficients)
# Plotting the data and linear regression line
plt.scatter([datetime.fromtimestamp(ts) for ts in timestamps], distances, label='Total distances')
plt.plot([datetime.fromtimestamp(ts) for ts in timestamps], poly_function(timestamps), color='red', label='Linear Regression')
plt.grid()
plt.xlabel('Date')
plt.ylabel('Total Distance Ran [m]')
plt.legend()
plt.show()






for month in months_for_plotting:
    subset_trainings = [training for training in trainings 
                        if training.start_time.year == month[0] and training.start_time.month == month[1]]
    subset_trainings.sort(key= lambda t : t.total_distance)
    distances = [training.total_distance/1000 for training in subset_trainings]
    avg_speed = [training.avg_speed*3.6 for training in subset_trainings]
    plt.scatter(np.array(distances), np.array(avg_speed), label=f"{month[0]}-{month[1]}")

    # Perform polynomial regression
    z = np.polyfit(distances, avg_speed, 2)
    p = np.poly1d(z)
    
    # Plot polynomial regression line
    plt.plot(distances, p(distances))
plt.xlabel("Distance [km]")
plt.ylabel("Average pace [min:sec per km]")
speeds = [i for i in range(1,15)]
plt.yticks(speeds, [speed_to_pace(speed) for speed in speeds])
plt.legend()
plt.grid()
plt.show()