import matplotlib.pyplot as plt
import contextily as cx
import numpy as np
import gpxpy
import os

target_folder = r"C:\Users\thehu\Documents\gpx_analysis\gpx_all"
file_paths = [target_folder+"/"+f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]

trajs = {}
for i, path in enumerate(file_paths):
    with open(path, "r") as file:
        gpx = gpxpy.parse(file)

        print(i+1,"/",len(file_paths))
        #if gpx.tracks[0].segments[0].points[0].time.year != 2023:
        #    continue

        if gpx.tracks[0].name not in trajs.keys():
            trajs[gpx.tracks[0].name] = []

        traj = [[point.longitude, point.latitude] for point in gpx.tracks[0].segments[0].points]
        trajs[gpx.tracks[0].name].append(traj)

for kind, routes in trajs.items():
    for i in range(len(routes)):
        routes[i].append([np.nan, np.nan])
    arr = np.array(sum(routes, []))
    plt.plot(arr[:,0], arr[:,1], label=kind)

plt.xlabel("lon")
plt.ylabel("lat")
plt.grid()
plt.legend()
plt.gca().axis("Equal")
cx.add_basemap(plt.gca(),crs="WGS84",zoom=10)
plt.show()
