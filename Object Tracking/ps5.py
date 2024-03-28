"""Problem Set 7: Particle Filter Tracking."""

import cv2
import numpy as np

from ps5_utils import run_kalman_filter, run_particle_filter

np.random.seed(100)  #DO NOT CHANGE THIS SEED VALUE

# I/O directories
input_dir = "input"
output_dir = "output"

# TODO: Remove unnecessary classes



def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out

# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.]).reshape(4, 1)  # state


        self.Dt = np.matrix(np.diag([1.,1.,1.,1.]))   # update the state from previous state
        self.Dt[:2, 2:] = np.diag([1.,1.])

        self.Mt = np.matrix(np.diag([1.,1.,1.,1.])[:2])  #update obervation from state

        self.Q = Q
        self.R = R

        self.covariance = np.matrix(np.diag([1.0, 1.0, 1.0, 1.0])) #initialize the covariance matrix

    def predict(self):
        self.state = self.Dt * self.state
        self.covariance = self.Dt * self.covariance * self.Dt.T + self.Q


    def correct(self, meas_x, meas_y):
        measurement = np.array([meas_x, meas_y]).reshape(2, 1)
        Kg = self.covariance * self.Mt.T * np.linalg.inv(self.Mt * self.covariance * self.Mt.T + self.R)
        self.state +=  Kg * (measurement - (self.Mt * self.state))

        self.covariance = (np.diag([1.,1.,1.,1.]) - (Kg * self.Mt)) * self.covariance

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = cv2.cvtColor(template.copy(), cv2.COLOR_BGR2GRAY)
        self.frame = frame.copy()
        self.particles = None  # Initialize your particles array. Read the docstring.
        self.weights = np.ones(self.num_particles,dtype=np.float32) / self.num_particles  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

        self.state_x = self.template_rect['x'] + np.floor(self.template_rect['w'] / 2.)
        self.state_y = self.template_rect['y'] + np.floor(self.template_rect['h'] / 2.)


        self.template_h, self.template_w = self.template.shape[:2]
        self.mse_list = []


    def get_cutout_around_particle(self,frame,particle):
        particle = particle.astype(np.int16)

        x0, x1 = particle[0] - np.floor(self.template_w / 2), particle[0] + np.floor(self.template_w / 2)
        y0, y1 = particle[1] - np.floor(self.template_h / 2), particle[1] + np.floor(self.template_h / 2)


        if self.template_w % 2:
            x1 += 1
        if self.template_h % 2:
            y1 += 1

        # correct if falls out of boundary
        if x0 < 1:
            x0, x1 = 0, (x1 - x0)

        if y0 < 1:
            y0, y1 = 0, (y1 - y0)

        if x1 > frame.shape[1] - 1:
            x0, x1 = x0 - (x1 - frame.shape[1]) - 1, frame.shape[1]-1

        if y1 > frame.shape[0] - 1:
            y0, y1 = y0 - (y1 - frame.shape[0]) - 1, frame.shape[0]-1

        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)
        output = frame[y0:y1, x0:x1]


        if not output.shape == self.template.shape:
            print(frame.shape)
            print(output.shape)

            raise AssertionError("cutout size not equal to template size")


        return output



    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        mse = np.sum((template.astype(np.float32) - frame_cutout.astype(np.float32)) ** 2) / (template.shape[0] * template.shape[1])
        return mse

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        return np.random.choice(a=self.num_particles, size=self.num_particles, p=self.weights, replace=True)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        rand_x = np.random.normal(self.state_x, self.sigma_dyn, self.num_particles).reshape(self.num_particles,1)
        rand_y = np.random.normal(self.state_y, self.sigma_dyn, self.num_particles).reshape(self.num_particles,1)
        self.particles = np.hstack([rand_x, rand_y])


        error_list = []
        for i in range(self.particles.shape[0]):
            particle = self.particles[i]
            mse = self.get_error_metric(self.template,self.get_cutout_around_particle(frame,particle))
            self.weights[i] = np.exp(-mse/ (2. * self.sigma_exp ** 2.))
            error_list.append(mse)

        self.mse_list.append(np.mean(error_list))


        self.weights = self.weights / sum(self.weights) #normalize
        self.particles = self.particles[self.resample_particles()] #resample particles
        self.state_x,self.state_y = np.average(self.particles, axis=0, weights=self.weights).astype(np.int16)




    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """


        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        x_weighted_mean = x_weighted_mean.astype(np.int16)
        y_weighted_mean = y_weighted_mean.astype(np.int16)

        for particle in self.particles:
            cv2.circle(frame_in, tuple(particle.astype(np.int16)), radius=1, color=(0, 0, 255), thickness=1)

        x0 = x_weighted_mean - np.floor(self.template_w / 2)
        y0 = y_weighted_mean - np.floor(self.template_h / 2)
        x1 = x_weighted_mean + np.floor(self.template_w/ 2)
        y1 = y_weighted_mean + np.floor(self.template_h/ 2)

        cv2.rectangle(frame_in,(int(x0),int(y0) ), (int(x1),int(y1) ), color = (0, 255, 255),thickness=2)

        dist = np.linalg.norm(self.particles - [x_weighted_mean, y_weighted_mean]) ** 0.5
        radius = (np.sum(dist * self.weights.reshape((-1, 1)))).astype(int)
        cv2.circle(frame_in, (x_weighted_mean, y_weighted_mean), radius=radius, color=(0, 255, 0), thickness=2)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        particle = np.array([self.state_x, self.state_y])
        best_frame = self.get_cutout_around_particle(frame, particle)


        template_t = self.alpha * best_frame + (1 - self.alpha) * self.template
        self.template = template_t.astype(np.uint8)
        self.template_h, self.template_w = self.template.shape[:2]




class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.scale = kwargs.get('scale', 1)
        self.max_mse = kwargs.get('max_mse',5000)
        self.scale_rate = kwargs.get('scale_rate',0.9999)
        self.template_min_h = kwargs.get('template_min_h',120)

        self.frame_id = 0
        self.avg_mse = 0


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """




        self.frame_id += 1
        print("Now processing frame:{}".format(self.frame_id))
        pre_x = self.state_x #save previous state
        pre_y = self.state_y

        ParticleFilter.process(self,frame.copy())
        current_avg = self.mse_list[self.frame_id-1]
        print(current_avg)
        if current_avg > self.max_mse:
            self.state_x = pre_x
            self.state_y = pre_y
            # scale = np.exp(-0.12 * self.frame_id)
            self.scale *= self.scale_rate
        elif self.template_h > self.template_min_h:
            particle = np.array([self.state_x, self.state_y])
            frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            best_frame = self.get_cutout_around_particle(frame, particle)
            template_t = self.alpha * best_frame + (1 - self.alpha) * self.template
            resized_temp = cv2.resize(template_t, (0, 0), fx=self.scale, fy=self.scale)

            if resized_temp.shape[0] > 0.8*frame.shape[0]:
                resized_temp = template_t

            # scale = np.exp(-0.12*self.frame_id)
            self.scale *= self.scale_rate
            print(self.scale)
            self.template = resized_temp.astype(np.uint8)
            self.template_h, self.template_w = self.template.shape[0], self.template.shape[1]









def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10 # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 400  # Define the number of particles
    sigma_mse = 5 # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 12  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.8  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 300  # Define the number of particles
    sigma_md = 15  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.1
    max_mse = 4300
    scale = 1
    scale_rate = 0.9999
    template_min_h = 120

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        max_mse = max_mse,
        scale = scale,
        scale_rate = scale_rate,
        template_min_h = template_min_h,
        template_coords=template_rect)  # Add more if you need to
    return out
