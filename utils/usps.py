# coding=utf-8

""" USPS dataset. 
    Segmented numerals digitized from handwritten zipcodes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import os
import itertools
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

URL = "https://papers.nips.cc/paper/293-handwritten-digit-recognition-with-a-back-propagation-network"

CITATION = r"""@incollection{lecun90handwritten,
    title = {Handwritten Digit Recognition with a Back-Propagation Network},
    author = {LeCun, Yann and Bernhard E. Boser and John S. Denker and Donnie Henderson and R. E. Howard and Wayne E. Hubbard and Lawrence D. Jackel},
    booktitle = {Advances in Neural Information Processing Systems 2},
    editor = {D. S. Touretzky},
    pages = {396--404},
    year = {1990},
    publisher = {Morgan-Kaufmann},
    url = {http://papers.nips.cc/paper/293-handwritten-digit-recognition-with-a-back-propagation-network.pdf}
}"""

DESCRIPTION = (
    "The USPS Dataset is an image digit recognition dataset consisting of "
    "segmented numerals digitized from handwritten zipcodes that appeared "
    "on real U.S. Mail passing through the Buffalo, N.Y. post office."
)

SHAPE=(16, 16, 1)


class USPS(tfds.core.GeneratorBasedBuilder):
    """ USPS dataset. 
        Segmented numerals digitized from handwritten zipcodes
    """

    VERSION = tfds.core.Version("0.1.0")

    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Please run ./scripts/get_digits.sh to download the USPS dataset. "
    )

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=SHAPE),
                "label": tfds.features.ClassLabel(num_classes=10),
            }),
            supervised_keys=("image", "label"),
            homepage=URL,
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):

        # file_path = dl_manager.download('https://cs.nyu.edu/~roweis/data/usps_all.mat')
        file_path = dl_manager.manual_dir

        # There is no predefined train/val/test split for this dataset.
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "data_path": os.path.join(file_path, "usps_all.mat"),
                },
            ),
        ]

    def _generate_examples(self, data_path:str):
        """Generate examples as dicts.
        Args:
        filepath: `str` path of the file to process.
        Yields:
        Generator yielding the next samples
        """

        # the labels file consists of lines of image-names and label pairs, e.g. "00000001.png 2"
        with tf.io.gfile.GFile(data_path, "rb") as f:
            data = tfds.core.lazy_imports.scipy.io.loadmat(f)['data']

        # Maybe we should shuffle ?

        # data dimensions are [256, 1100, 10], i.e. [16x16, n_examples, n_classes]

        for i, (example_num, label) in enumerate(
            itertools.product(range(data.shape[1]), range(data.shape[2]))
        ):
            image = np.swapaxes(
                data[:, example_num, label].reshape(SHAPE),
                0,1,
            )
            record = {
                "image": image,
                "label": (label+1) % 10, #
            }
            yield i, record


if __name__ == "__main__":
    ds, info = tfds.load("usps", split="train", with_info=True)
    print(info)

