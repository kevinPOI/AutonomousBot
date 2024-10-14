This is the repository for the autonomous robot project of team Carnegie Mellon Combat Robots. We aim to build an autonomous driving stack the controls the robot from a external perspective: essentially a robot that drive robots. 

For the perception demo, clone the repo and go into folder "autonomousBot/externalPerception", and in command line, run "python subtraction_test.py"\
When a image (first frame of the video) Click on the four corners of the arena to calibrate the birds-eye-view transformation parameter. Then close the window. Video with tracking should start\
It may prompt you to install some libraries before it can run\
(Do not run from the autonomousBot folder, that will give you a path error or "None type does not have... " error)\
To save transformed (and unannotated) video, edit subtraction_test.py and change SAVEVIDEO to true. It saves the video to "output_video.mp4" by default\
