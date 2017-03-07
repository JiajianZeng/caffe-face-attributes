import caffe
import numpy as np
import yaml

DEBUG = False

class RoISelectLayer(caffe.Layer):
    """
    Selects num_class rois from the input rois
    Whose class label are 1, 2, ..., num_class - 1, 0 respectively
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self._num_class = layer_params['num_class']

        if DEBUG:
            print 'num_class: {}'.format(self._num_class)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(num_class, 5)
        top[0].reshape(self._num_class, 5)

    def forward(self, bottom, top):
        # the input rois and corresponding scores
        # input_rois: holds N regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)

        # input_scores: holds the corresponding scores of the N regions of interest
        # w.r.t the num_class classes, each is a num_class tuple
        # (score_of_class_0, score_of_class_1, ... ,score_of_class_num_class-1)
        input_rois = bottom[0].data
        input_scores = bottom[1].data

        # labels of the input rois
        labels = np.argmax(input_scores, axis=1)

        # max scores of the input rois
        max_scores = np.amax(input_scores, axis=1)

        # Select foreground RoIs per class
        fg_inds = []
        for i in np.arange(1, self._num_class):
            inds_this_class = np.where(labels == i)[0]
            if len(inds_this_class) == 0:
                scores_this_class = input_scores[:, i]
                fg_inds.append(np.argmax(scores_this_class))
                print 'No class {:d} detected in the input image.'.format(i)
            else:
                max_scores_this_class = max_scores[inds_this_class]
                max_score_ind = np.argmax(max_scores_this_class)
                fg_inds.append(inds_this_class[max_score_ind])

        # Select background RoIs
        bg_inds = []
        inds_background = np.where(labels == 0)[0]
        if len(inds_background) == 0:
            scores_background = input_scores[:, 0]
            bg_inds.append(np.argmax(scores_background))
            print 'No background detected in the input image.'
        else:
            max_scores_background = max_scores[inds_background]
            max_score_ind = np.argmax(max_scores_background)
            bg_inds.append(inds_background[max_score_ind])

        keep_inds = np.append(fg_inds, bg_inds)
        selected_rois = input_rois[keep_inds]

        top[0].reshape(*(selected_rois.shape))
        top[0].data[...] = selected_rois


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass