# Self Driving Car using OpenCV (No AI implemented yet !)
 Important : Make sure you play in 640x480 resolution   
Current Task : Stay in same lane as longer as possible
## Latest preview of this branch in-game : [YouTube](https://youtu.be/e3yZM15-OuE)
**Here's a preview of how it works :**    
\
Line detection preview :   
![Line detection](/visuals/lineDetectionGIF.gif)
\
\
In-game preview :   
![Original gameplay](/visuals/ingamePreviewGIF.gif)  
\
\
The lower red pixel is RLcontrolPoint (controls steering of the vehicle) and the upper red point is FWcontrolPoint (controls speed of the vehicle (not fully implemented yet))
- The first value of check() function returns the distance between the control Point and first white pixel it finds in a specific direction from the control point.
- When the distance between RLControlPoint is less than some threshold value it turns to the opposite direction. It also draws a red line between those pixel when it does that (Look At Above Preview !)
- Lets say the distance between RLcontrolPoint and the white pixel in East Direction is less than threshold value. So it detected that the car is going too much in the right and needs to turn left in order to stay in the lane. That's why is pressed the **A** key to turn left.
- For FWControlPoint : if the distance in the north direction is less than some value than car apply brakes in order to avoid accident. and if it detects any line below FWcontrolPoint (because a car crashed into other car or object) it will take reverse.
- FWcontrolPoint will be removed in future.
#### Kind of a dumb algorithm but hey it works! Click on YouTube link for full preview.
  
## Dependencies :
- OpenCV (pip install opencv-python)
- pywin32 (pip install pywin32)
- numpy (pip install numpy)
- pynput (pip install pynput)
- time   
  _**Made in python 3.7**_
