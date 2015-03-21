from skimage.data import coffee, camera
from sklearn_theano.feature_extraction.caffe.vggcnns import (
    VGGCNNSTransformer, VGGCNNSClassifier)
import numpy as np
from nose import SkipTest
import os

co = coffee().astype(np.float32)
ca = camera().astype(np.float32)[:, :, np.newaxis] * np.ones((1, 1, 3),
                                                             dtype='float32')


def test_vggcnn_s_transformer():
    """smoke test for VGG CNN S transformer"""
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")
    t = VGGCNNSTransformer()

    t.transform(co)
    t.transform(ca)


def test_googlenet_classifier():
    """smoke test for VGG CNN S classifier"""
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")
    c = VGGCNNSClassifier()

    c.predict(co)
    c.predict(ca)
