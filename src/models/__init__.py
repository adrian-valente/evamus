

class Model:
    def train(self, dataset):
        """
        :param dataset: training set
        :return: predictions over the training dataset
        """
        pass

    def predict(self, dataset):
        """
        :param dataset: test set
        :return: predictions over the test dataset
        """
        pass

    def generate(self, n_songs, N):
        """
        :param n_songs:
        :param N: length of song
        :return: generated dataset
        """
        pass
