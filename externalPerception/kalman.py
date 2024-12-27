import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, process_noise, measurement_noise):
        """
        initial_state: Initial state vector [x, y, theta, dx, dy, dtheta].
        process_noise: Process noise covariance matrix.
        measurement_noise: Measurement noise covariance matrix.
        """
        self.state = np.array(initial_state)  # [x, y, theta, dx, dy, dtheta]
        self.P = np.eye(6)  # Initial state covariance
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.dt = 0.1 #placeholder for variable dt
        
        # State transition matrix (updated in the predict step)
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        
        # Measurement matrix
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

    def predict(self, dt):
        """
        prediction without control model (for now)
        x_k =  F x_k-1
        P_k = F P_k-1 F + Q
        """
        self.F = np.eye(6)
        self.F[0, 3] = dt  # x += dx * dt
        self.F[1, 4] = dt  # y += dy * dt
        self.F[2, 5] = dt* 0.1  # theta += dtheta * dt, but dtheta is unreliable

        self.state = self.F @ self.state
        
        #covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        :param measurement: [x, y, theta].
        """
        # Compute the measurement residual
        z = np.array(measurement)
        y = z - (self.H @ self.state)
        
        # Compute the Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.P = (np.eye(len(self.state)) - K @ self.H) @ self.P

    def get_state(self):
        return self.state

if __name__ == "__main__":
    #testing code
    initial_state = [0, 0, 0, 0, 0, 0]
    process_noise = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    measurement_noise = np.diag([0.05, 0.05, 0.01])

    kf = KalmanFilter(initial_state, process_noise, measurement_noise)
    measurements = [
        [1.0, 1.0, 0.1],
        [2.0, 1.8, 0.15],
        [5.0, 2.5, 0.2],
        [4.0, 3.3, 0.25],
    ]

    for measurement in measurements:
        kf.predict()
        kf.update(measurement)
        print("Estimated state:", kf.get_state())
