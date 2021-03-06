# Track_Python

This routine can plot both the observed and modeled drifter tracks. It has various options including how to specify start positions, how long to track,whether to generate animation output, etc. See the options description below:

Option 1 : We can call it "Drifter Track", simply. In this option, we got a one day's drifter track and forecast it a few days (depends on the parameter 'track_days') start at the last point of the drifter. What you need do is give Drifter ID, Filestyle and track_days. See the sample(image_style=animation), [click here.](./Samples of Animation/Option-1-drifter_track.gif) 

Option 2 : Named "Line Track". Allow us enter one or two points and specify numbers between two points as start points, then forecast or hindcast(depends on parameter 'track_way',forward or backward) a few days. You can also choose 'track_way=both' to track both forward and backward at the same time. There are optional features available, like boundary_switch, streamline,etc. [Here](./Samples of Animation/streamline.gif) is a sample of "streamline=ON" case.

Option 3 : Named "Multiple Track".  This option generates a simulation map of Cape Cod Bay first, you click on the map to add points where you want track. and the features are same to option 2 . Sample [here](./Samples of Animation/Option-3.gif) is an animation of three points forecast one day.

Option 4 : Named "Box Track". Specify a point, radius, a number(kind of density), you will got a box contains a certain amount of points that is the start points we will track. The features are available same to other options. See the sample(image_style=animation), [click here.](./Samples of Animation/Option4.gif)
