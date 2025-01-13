import numpy as np
def cap_input(input, limit):
    """
    Caps the input value within the range (-limit, limit).

    Parameters:
        input (float): The value to be capped.
        limit (float): The limit for the range.

    Returns:
        float: The capped value.
    """
    return max(-limit, min(input, limit))

def diff_angles(angle1, angle2):
    difference = angle1 - angle2
    return (difference + np.pi) % (2 * np.pi) - np.pi

class Controller:
    def __init__(self, us, opp):
        self.us = us
        self.opp = opp
        #set w and h after first warped frame is obtained
        self.frame_w = None
        self.frame_h = None
    
    def aim_on_opponnent(self):
        us_to_opp = self.opp.pose[:2] - self.us.pose[:2]
        desired_angle = np.arctan2(us_to_opp[1], us_to_opp[0])
        delta_angle = diff_angles(desired_angle, self.us.pose[2])
        thro = 0
        steer = cap_input(-delta_angle / 2, 0.6)
        return np.array([thro, steer])
        pass
    def near_walls(self):
        stride = 100
        x,y,th = self.us.pose
        near_left_or_right = x <= stride or x >= self.frame_w - stride
        near_top_or_bottom = y <= stride or y >= self.frame_h - stride
        return near_left_or_right or near_top_or_bottom
    
    def aimed_at_opponent(self):
        us_to_opp = self.opp.pose[:2] - self.us.pose[:2]
        desired_angle = np.arctan2(us_to_opp[1], us_to_opp[0])
        delta_angle = diff_angles(desired_angle, self.us.pose[2])
        print("desired angle: ", desired_angle, "actual ", self.us.pose[2])
        if abs(delta_angle) < 0.5:
            return True
        else:
            return False
    def move_away_from_walls(self):
        center = np.array([self.frame_w/2, self.frame_h/2])
        us_to_center = center - self.us.pose[:2]
        desired_angle = np.arctan2(us_to_center[1], us_to_center[0])
        delta_angle = diff_angles(desired_angle, self.us.pose[2])

        if abs(delta_angle) < 0.6:#facing center
            thro = 0.5
            steer = cap_input(-delta_angle / 2, 0.3)
        else:
            thro = 0
            steer = cap_input(-delta_angle / 2, 0.6)
        return np.array([thro, steer])
    
    def move_towards_opponent(self):
        us_to_opp = self.opp.pose[:2] - self.us.pose[:2]
        desired_angle = np.arctan2(us_to_opp[0], us_to_opp[1])
        delta_angle = diff_angles(desired_angle, self.us.pose[2])
        thro = 0.6
        steer = cap_input(-delta_angle / 3, 0.2)
        return np.array([thro, steer])
    

    def get_controls(self):
        """
        Main function. Returns desired controls
        rightnow only position based, not considering velocity / acce / prediction

        Returns:
            [throttle, steering] in range (-1, 1)
            positive steering go counter clockwise
        """
        controls = np.array([0.0,0.0])
        if self.near_walls():
            print("controller: moving away from walls")
            controls = self.move_away_from_walls()
        else:
            if self.aimed_at_opponent():
                print("controller: moving towards opponent")
                controls = self.move_towards_opponent()
            else:
                print("controller: aiming on  opponent")
                controls = self.aim_on_opponnent()
        return controls