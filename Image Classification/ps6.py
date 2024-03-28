"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder)]
    x, labels =[], []
    for image in images_files:
        img = cv2.imread(os.path.join(folder, image), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        img_array = img.flatten()
        x.append(img_array)
        labels.append(int(image.split('subject')[1][:2]))
    return np.array(x), np.array(labels)
    # raise NotImplementedError


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """

    # raise NotImplementedError
    num_train_samples = int(p * X.shape[0])
    shuffled_indices = np.random.permutation(X.shape[0])
    train_indices = shuffled_indices[:num_train_samples]
    test_indices = shuffled_indices[num_train_samples:]
    Xtrain, Xtest = X[train_indices], X[test_indices]
    ytrain, ytest = y[train_indices], y[test_indices]

    return Xtrain, ytrain, Xtest, ytest


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x, axis=0)
    # raise NotImplementedError


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    # raise NotImplementedError
    centered_data = X - get_mean_face(X)
    covariance_matrix  = np.dot(centered_data.T,centered_data)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix )
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    top_k_eigenvectors = eigenvectors[:, :k]
    top_k_eigenvalues = eigenvalues[:k]
    return top_k_eigenvectors, top_k_eigenvalues


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        # raise NotImplementedError
        for i in range(self.num_iterations):
            self.weights /= np.sum(self.weights)
            weak_classifer = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            weak_classifer.train()
            prediction = weak_classifer.predict(np.transpose(self.Xtrain))
            error = np.sum(self.weights[np.where(self.ytrain != prediction)])
            alpha = 0.5 * np.log((1 - error) / error)
            self.weights *= np.exp(-alpha * self.ytrain * prediction)
            self.alphas.append(alpha)
            self.weakClassifiers.append(weak_classifer)
            if error < self.eps:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        # raise NotImplementedError
        predictions = self.predict(self.Xtrain)
        correct = np.sum(predictions == self.ytrain)
        incorrect = len(self.ytrain) - correct
        return correct, incorrect

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        # raise NotImplementedError
        final_pred = []
        for i, j in zip(self.alphas, self.weakClassifiers):
            predictions = j.predict(np.transpose(X))
            final_pred.append(i * predictions)
        return np.sign(np.sum(final_pred, axis=0))


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        # raise NotImplementedError
        y, x = self.position
        h, w = self.size
        feature = np.zeros(shape, dtype=np.uint8)
        half_h = h // 2
        feature[y:y + half_h, x:x + w] = 255
        feature[y + half_h:y + h, x:x + w] = 126
        return feature

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        # raise NotImplementedError
        y, x = self.position
        h, w = self.size
        feature = np.zeros(shape, dtype=np.uint8)
        half_w = w // 2
        feature[y:y + h, x:x + half_w] = 255
        feature[y:y + h, x + half_w:x + w] = 126
        return feature

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        # raise NotImplementedError
        y, x = self.position
        h, w = self.size
        feature = np.zeros(shape, dtype=np.uint8)
        third_h = h // 3
        feature[y:y + third_h, x:x + w] = 255
        feature[y + third_h:y + 2 * third_h, x:x + w] = 126
        feature[y + 2 * third_h:y + h, x:x + w] = 255
        return feature

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        # raise NotImplementedError
        y, x = self.position
        h, w = self.size
        feature = np.zeros(shape, dtype=np.uint8)
        third_w = w // 3
        feature[y:y + h, x:x + third_w] = 255
        feature[y:y + h, x + third_w: x + 2 * third_w] = 126
        feature[y:y + h, x + 2 * third_w:x + w] = 255
        return feature

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        # raise NotImplementedError
        y, x = self.position
        h, w = self.size
        feature = np.zeros(shape, dtype=np.uint8)
        half_h = h // 2
        half_w = w // 2
        feature[y:y + half_h, x:x + half_w] = 126
        feature[y:y + half_h, x + half_w:x + w] = 255
        feature[y + half_h:y + h, x:x + half_w] = 255
        feature[y + half_h:y + h, x + half_w:x + w] = 126
        return feature

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        # raise NotImplementedError
        y, x = self.position
        h, w = self.size

        if self.feat_type == (2, 1):
            A = ii[y + int(h / 2) - 1, x + w - 1] - ii[y + int(h / 2) - 1, x - 1] - ii[y - 1, x + w - 1] + ii[y - 1, x - 1]
            B = ii[y + h - 1, x + w - 1] - ii[y + int(h / 2) - 1, x + w - 1] - ii[y + h - 1, x - 1] + ii[y + int(h / 2) - 1, x - 1]
            return A - B

        elif self.feat_type == (1, 2):
            A = ii[y + h - 1, x + int(w / 2) - 1] - ii[y - 1, x + int(w / 2) - 1] - ii[y + h - 1, x - 1] + ii[y - 1, x - 1]
            B = ii[y + h - 1, x + w - 1] - ii[y - 1, x + w - 1] - ii[y + h - 1, x + int(w / 2) - 1] + ii[y - 1, x + int(w / 2) - 1]
            return A - B

        elif self.feat_type == (3, 1):
            A = ii[y + int(h / 3) - 1, x + w - 1] - ii[y - 1, x + w - 1] - ii[y + int(h / 3) - 1, x - 1] + ii[y - 1, x - 1]
            B = ii[y + int(2 * h / 3) - 1, x + w - 1] - ii[y + int(h / 3) - 1, x + w - 1] - ii[y + int(2 * h / 3) - 1, x - 1] + ii[y + int(h / 3) - 1, x - 1]
            C = ii[y + h - 1, x + w - 1] - ii[y + int(2 * h / 3) - 1, x + w - 1] - ii[y + h - 1, x - 1] + ii[y + int(2 * h / 3) - 1, x - 1]
            return A - B + C

        elif self.feat_type == (1, 3):
            A = ii[y + h - 1, x + int(w / 3) - 1] - ii[y - 1, x + int(w / 3) - 1] - ii[y + h - 1, x - 1] + ii[y - 1, x - 1]
            B = ii[y + h - 1, x + int(2 * w / 3) - 1] - ii[y - 1, x + int(2 * w / 3) - 1] - ii[y + h - 1, x + int(w / 3) - 1] + ii[y - 1, x + int(w / 3) - 1]
            C = ii[y + h - 1, x + w - 1] - ii[y - 1, x + w - 1] - ii[y + h - 1, x + int(2 * w / 3) - 1] + ii[y - 1, x + int(2 * w / 3) - 1]
            return A - B + C

        elif self.feat_type == (2, 2):
            A = ii[y + int(h / 2) - 1, x + int(w / 2) - 1] - ii[y - 1, x + int(w / 2) - 1] - ii[y + int(h / 2) - 1, x - 1] + ii[y - 1, x - 1]
            B = ii[y + int(h / 2) - 1, x + w - 1] - ii[y - 1, x + w - 1] - ii[y + int(h / 2) - 1, x + int(w / 2) - 1] + ii[y - 1, x + int(w / 2) - 1]
            C = ii[y + h - 1, x + int(w / 2) - 1] - ii[y + int(h / 2) - 1, x + int(w / 2) - 1] - ii[y + h - 1, x - 1] + ii[y + int(h / 2) - 1, x - 1]
            D = ii[y + h - 1, x + w - 1] - ii[y + int(h / 2) - 1, x + w - 1] - ii[y + h - 1, x + int(w / 2) - 1] + ii[y + int(h / 2) - 1, x + int(w / 2) - 1]
            return -A + B + C - D


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    # raise NotImplementedError
    integral_images = []

    for img in images:
        img_array = np.array(img, dtype=np.float32)
        integral_img = np.zeros_like(img_array, dtype=np.float32)
        integral_img = np.cumsum(np.cumsum(img_array, axis=0), axis=1)
        integral_images.append(integral_img)

    return integral_images


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))
        self.threshold = 2.0

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_train(self):
        """ This function initializes self.scores, self.weights

        Args:
            None

        Returns:
            None
        """
    
        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        if not self.integralImages or not self.haarFeatures:
            print("No images provided. run convertImagesToIntegralImages() first")
            print("       Or no features provided. run creatHaarFeatures() first")
            return

        self.scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            self.scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        self.weights = np.hstack((weights_pos, weights_neg))

    def train(self, num_classifiers):
        """ Initialize and train Viola Jones face detector

        The function should modify self.weights, self.classifiers, self.alphas, and self.threshold

        Args:
            None

        Returns:
            None
        """
        self.init_train()
        print(" -- select classifiers --")
        for _ in range(num_classifiers):
            # TODO: Complete the Viola Jones algorithm
            # raise NotImplementedError
            self.weights = self.weights / np.sum(self.weights)

            vj = VJ_Classifier(self.scores, self.labels, self.weights)
            vj.train()

            self.classifiers.append(vj)

            error = vj.error
            beta = error / (1.0 - error)
            alpha = np.log(1.0 / beta)
            self.alphas.append(alpha)

            preds = vj.predict(np.transpose(self.scores))
            et = [-1 if preds[i] == self.labels[i] else 1 for i in range(len(preds))]
            self.weights = self.weights * np.power(beta, 1 - np.array(et))


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.
        for i, im in enumerate(ii):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        # Calculate predictions using the strong classifier H(x)

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        for x in scores:
            lhs = 0
            for i in range(len(self.alphas)):
                vj = self.classifiers[i]
                p = vj.predict(x)
                lhs += self.alphas[i] * p

            rhs = sum(self.alphas) / 2
            if lhs >= rhs:
                result.append(1)
            else:
                result.append(-1)

        return result
        

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        # raise NotImplementedError
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        window_height, window_width = 24, 24
        windows = []
        centers = []

        for y in range(int(np.floor(window_height / 2)), image.shape[0] - int(np.floor(window_height / 2)) - 1, 4):
            for x in range(int(np.floor(window_width / 2)), image.shape[1] - int(np.floor(window_width / 2)) - 1, 6):
                window = gray[(y - int(np.floor(window_height / 2))):(y + int(np.floor(window_height / 2))),
                            (x - int(np.floor(window_width / 2))):(x + int(np.floor(window_width / 2)))]
                windows.append(window)
                centers.append((y, x))
        predictions = self.predict(windows)
        centers = np.array(centers)
        predictions = np.array(predictions)
        positive_indices = np.where(predictions == 1)
        positive_centers = centers[positive_indices]
        mean_center = np.mean(positive_centers, axis=0)
        mean_center = np.floor(mean_center).astype(np.int16)


        cv2.rectangle(img, (mean_center[1] - int(np.floor(window_width / 2)), mean_center[0] - int(np.floor(window_height / 2))),
                    (mean_center[1] + int(np.floor(window_width / 2)), mean_center[0] + int(np.floor(window_height / 2))),
                    (255, 0, 0), 1)

        cv2.imwrite('output/' + filename + '.png', img)

class CascadeClassifier:
    """Viola Jones Cascade Classifier Face Detection Method

    Lesson: 8C-L2, Boosting and face detection

    Args:
        f_max (float): maximum acceptable false positive rate per layer
        d_min (float): minimum acceptable detection rate per layer
        f_target (float): overall target false positive rate
        pos (list): List of positive images.
        neg (list): List of negative images.

    Attributes:
        f_target: overall false positive rate
        classifiers (list): Adaboost classifiers
        train_pos (list of numpy arrays):  
        train_neg (list of numpy arrays): 

    """
    def __init__(self, pos, neg, f_max_rate=0.30, d_min_rate=0.70, f_target = 0.07):
        
        train_percentage = 0.85

        pos_indices = np.random.permutation(len(pos)).tolist()
        neg_indices = np.random.permutation(len(neg)).tolist()

        train_pos_num = int(train_percentage * len(pos))
        train_neg_num = int(train_percentage * len(neg))

        pos_train_indices = pos_indices[:train_pos_num]
        pos_validate_indices = pos_indices[train_pos_num:]

        neg_train_indices = neg_indices[:train_neg_num]
        neg_validate_indices = neg_indices[train_neg_num:]

        self.train_pos = [pos[i] for i in pos_train_indices]
        self.train_neg = [neg[i] for i in neg_train_indices]

        self.validate_pos = [pos[i] for i in pos_validate_indices]
        self.validate_neg = [neg[i] for i in neg_validate_indices]

        self.f_max_rate = f_max_rate
        self.d_min_rate = d_min_rate
        self.f_target = f_target
        self.classifiers = []

    def predict(self, classifiers, img):
        """Predict face in a single image given a list of cascaded classifiers

        Args:
            classifiers (list of element type ViolaJones): list of ViolaJones classifiers to predict 
                where index i is the i'th consecutive ViolaJones classifier
            img (numpy.array): Input image

        Returns:
            Return 1 (face detected) or -1 (no face detected) 
        """

        # TODO
        # raise NotImplementedError
        for classifier in classifiers:
            prediction = classifier.predict([img])
            if prediction[0] == -1:
                return -1
        return 1

    def evaluate_classifiers(self, pos, neg, classifiers):
        """ 
        Given a set of classifiers and positive and negative set
        return false positive rate and detection rate 

        Args:
            pos (list): Input image.
            neg (list): Output image file name.
            classifiers (list):  

        Returns:
            f (float): false positive rate
            d (float): detection rate
            false_positives (list): list of false positive images
        """

        # TODO
        # raise NotImplementedError
        false_positives = []
        num_false_positives = 0
        num_positives = len(pos)
        num_negatives = len(neg)
        
        for img in pos:
            result = self.predict(classifiers, img)
            if result == -1:
                false_positives.append(img)
                num_false_positives += 1

        true_negatives = num_negatives - num_false_positives

        f = num_false_positives / num_positives
        d = true_negatives / num_negatives

        return f, d, false_positives

    def train(self):
        """ 
        Trains a cascaded face detector

        Sets self.classifiers (list): List of ViolaJones classifiers where index i is the i'th consecutive ViolaJones classifier

        Args:
            None

        Returns:
            None
             
        """
        # TODO
        # raise NotImplementedError
        f_target = self.f_target
        f_max_rate = self.f_max_rate
        d_min_rate = self.d_min_rate

        self.classifiers = []

        i = 0
        f = f_last = 1.0
        integral_images = None
        while f > self.f_target:
            vj = ViolaJones(self.train_pos, self.train_neg, integral_images)
            i += 1
            ni = 0
            while f > f_max_rate * f_last:
                ni += 1
                integral_images = convert_images_to_integral_images(self.train_pos + self.train_neg)
                vj.train(ni)
                classifiers = self.classifiers + [vj]
                f, d, _ = self.evaluate_classifiers(self.validate_pos, self.validate_neg, classifiers)

                while d < self.d_min_rate * d_last:
                    vj.set_threshold(vj.threshold - 0.1)
                    classifiers = self.classifiers + [vj]
                    f, d, _ = self.evaluate_classifiers(self.validate_pos, self.validate_neg, classifiers)

            f_last = f
            d_last = d
            self.classifiers.append(vj)


    def faceDetection(self, image, filename="ps6-5-b-1.jpg"):
        """Scans for faces in a given image using the Cascaded Classifier.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        # raise NotImplementedError
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        window_height, window_width = 24, 24
        for y in range(0, image.shape[0] - window_height, 4):
            for x in range(0, image.shape[1] - window_width, 6):
                window = gray[y:y + window_height, x:x + window_width]
                prediction = self.predict(self.classifiers, window)
                if prediction == 1:
                    cv2.rectangle(img, (x, y), (x + window_width, y + window_height), (255, 0, 0), 1)

        cv2.imwrite(filename, img)
