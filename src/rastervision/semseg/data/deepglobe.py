from os.path import join

import numpy as np

from rastervision.common.utils import (
    save_img, load_img, get_img_size, _makedirs,
    save_numpy_array)
from rastervision.common.settings import (
    TRAIN, VALIDATION, TEST
)

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from rastervision.common.utils import (
    expand_dims, compute_ndvi, plot_img_row, download_dataset)
from rastervision.common.data.generators import Batch, FileGenerator

DEEPGLOBE = 'deepglobe/land-train'
PROCESSED_DEEPGLOBE = 'processed_deepglobe/'


class DeepGlobeBatch(Batch):
    def __init__(self):
        self.y_mask = None

        super().__init__()


class DeepGlobeDataset():
    def __init__(self):
        self.red_ind = 0
        self.green_ind = 1
        self.blue_ind = 2
        self.rgb_inds = [self.red_ind, self.green_ind, self.blue_ind]

        self.nb_channels = 3

        self.display_means = np.array([0.5] * self.nb_channels)
        self.display_stds = np.array([0.2] * self.nb_channels)

        # RGB vectors corresponding to different labels
        # Urban (RGB: 0, 255, 255) - Man-made, built up areas with human artifacts (can ignore roads for now which is hard to label)
        # Agriculture (RGB: 255, 255, 0) - Farms, any planned (i.e. regular) plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations.
        # Rangeland (RGB: 255, 0, 255) - Any non-forest, non-farm, green land, grass
        # Forest (RGB: 0, 255, 0) - Any land with x% tree crown density plus clearcuts.
        # Water (RGB: 0, 0, 255) - Rivers, oceans, lakes, wetland, ponds.
        # Barren (RGB: 255, 255, 255) - Mountain, land, rock, dessert, beach, no vegetation
        # Unknown (RGB: 0, 0, 0) - Clouds and others
        self.label_keys = [
            [0, 255, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [0, 0, 0],
        ]

        self.nb_labels = len(self.label_keys)

        self.label_names = [
            'Urban',
            'Agriculture',
            'Rangeland',
            'Forest',
            'Water',
            'Barren',
            'Unknown'
        ]

    @expand_dims
    def rgb_to_mask_batch(self, batch):
        """Convert a label image with black boundary pixels into a mask.

        Since there is uncertainty associated with the boundary of
        objects/regions in the ground truth segmentation, it makes sense
        to ignore these boundaries during evaluation. To help, the contest
        organizers have provided special ground truth images where the boundary
        pixels (in a 3 pixel radius) are black.

        # Returns
            A boolean array where an element is True if it should be used in
            the evaluation, and ignored otherwise.
        """
        mask = (batch[:, :, :, 0] == 0) & \
               (batch[:, :, :, 1] == 0) & \
               (batch[:, :, :, 2] == 0)
        mask = np.bitwise_not(mask)
        mask = np.expand_dims(mask, axis=3)

        return mask

    @expand_dims
    def rgb_to_label_batch(self, batch):
        label_batch = np.zeros(batch.shape[:-1])
        for label, key in enumerate(self.label_keys):
            mask = (batch[:, :, :, 0] == key[0]) & \
                   (batch[:, :, :, 1] == key[1]) & \
                   (batch[:, :, :, 2] == key[2])
            label_batch[mask] = label

        return np.expand_dims(label_batch, axis=3)

    @expand_dims
    def label_to_one_hot_batch(self, label_batch):
        if label_batch.ndim == 4:
            label_batch = np.squeeze(label_batch, axis=3)

        nb_labels = len(self.label_keys)
        shape = np.concatenate([label_batch.shape, [nb_labels]])
        one_hot_batch = np.zeros(shape)

        for label in range(nb_labels):
            one_hot_batch[:, :, :, label][label_batch == label] = 1.
        return one_hot_batch

    @expand_dims
    def rgb_to_one_hot_batch(self, rgb_batch):
        label_batch = self.rgb_to_label_batch(rgb_batch)
        return self.label_to_one_hot_batch(label_batch)

    @expand_dims
    def label_to_rgb_batch(self, label_batch):
        if label_batch.ndim == 4:
            label_batch = np.squeeze(label_batch, axis=3)

        rgb_batch = np.zeros(np.concatenate([label_batch.shape, [3]]),
                             dtype=np.uint8)
        for label, key in enumerate(self.label_keys):
            mask = label_batch == label
            rgb_batch[mask, :] = key

        return rgb_batch

    @expand_dims
    def one_hot_to_label_batch(self, one_hot_batch):
        one_hot_batch = np.argmax(one_hot_batch, axis=3)
        return np.expand_dims(one_hot_batch, axis=3)

    @expand_dims
    def one_hot_to_rgb_batch(self, one_hot_batch):
        label_batch = self.one_hot_to_label_batch(one_hot_batch)
        return self.label_to_rgb_batch(label_batch)

    def get_output_file_name(self, file_ind):
        return '{}_mask.png'.format(file_ind)


class DeepGlobeFileGenerator(FileGenerator):
    """
    A data generator for the DeepGlobe dataset that creates batches from
    files on disk.
    """
    def __init__(self, options):
        self.dataset = DeepGlobeDataset()

        # 423 out of 803
        self.dev_file_inds = [119, 855, 6399, 7791, 10233, 10901, 13415, 15573, 16453, 19627, 20187, 21023, 21717, 24813, 26261, 28559, 28689, 28935, 29419, 33573, 34359, 34567, 36183, 37755, 45357, 51911, 53987, 61245, 71125, 71619, 72807, 76759, 77669, 79049, 81011, 81039, 86805, 88571, 95613, 95683, 95863, 96841, 97337, 103665, 104113, 111335, 114433, 114473, 114577, 119079, 120245, 120625, 124529, 125795, 129297, 133209, 134465, 137499, 139581, 140299, 141685, 143353, 147545, 148381, 152569, 155165, 156951, 157839, 158163, 159177, 160037, 161109, 164029, 166293, 166805, 170535, 172307, 176225, 181447, 182027, 186739, 195769, 200561, 200589, 202277, 207663, 207743, 208495, 208695, 209787, 210473, 210669, 211739, 215525, 217085, 218329, 219555, 225393, 225945, 229383, 232373, 233615, 234269, 235869, 239955, 242583, 244423, 247179, 252743, 253691, 254565, 255711, 255889, 256189, 257695, 262885, 267065, 267163, 268881, 269601, 271245, 271609, 271941, 276761, 277049, 280703, 280861, 286339, 291781, 294697, 296279, 298817, 299287, 300745, 300967, 303327, 308959, 310419, 321711, 323581, 326173, 329017, 331421, 331533, 333661, 334677, 334811, 335737, 338111, 338661, 343215, 343425, 347725, 350033, 351271, 351727, 354033, 358591, 361129, 362191, 365555, 373103, 375563, 376441, 383637, 384477, 388811, 392711, 393043, 396979, 397137, 397351, 400179, 402209, 406425, 407467, 411741, 413779, 416381, 416463, 417313, 418261, 423117, 427037, 428327, 428597, 428841, 430587, 432089, 434243, 435277, 437963, 438721, 442329, 443271, 449319, 454655, 457265, 458687, 461001, 461755, 463855, 467855, 468103, 471187, 476991, 482365, 485061, 491491, 492365, 498049, 499161, 499325, 499511, 501053, 505217, 507241, 508571, 512669, 513585, 514385, 515521, 516317, 518833, 525105, 528163, 537221, 538243, 541353, 544537, 547201, 547785, 548423, 549959, 552001, 557175, 557309, 557439, 559477, 560353, 561117, 572237, 574789, 576417, 584663, 584865, 584941, 585043, 591815, 596837, 599743, 599975, 602453, 603617, 604647, 604833, 605037, 605707, 608673, 611015, 613687, 614561, 621459, 621633, 622733, 623857, 626323, 627583, 628479, 632489, 634421, 634717, 635157, 635841, 636849, 638937, 639149, 641771, 642909, 644103, 645001, 650253, 650751, 651537, 652183, 652733, 652883, 655313, 659953, 660069, 660933, 668465, 669779, 672041, 672823, 673927, 675849, 679507, 682949, 684377, 686781, 695475, 696257, 698065, 703413, 707319, 708527, 711893, 713813, 715633, 717225, 723067, 723719, 726265, 728521, 730821, 730889, 732669, 736869, 736933, 740937, 741105, 748225, 749375, 749523, 751939, 755453, 757745, 759855, 761189, 762359, 762937, 763075, 768475, 771393, 772567, 774779, 777185, 782103, 798411, 799523, 801361, 802645, 806805, 810749, 811075, 820347, 820543, 834433, 835147, 838669, 838873, 839641, 841621, 845069, 848649, 849797, 857201, 858771, 861353, 867017, 867349, 867983, 868003, 870705, 875327, 875409, 882451, 888263, 888343, 889145, 890145, 891153, 893261, 893651, 895509, 897901, 898741, 899693, 900985, 901715, 903649, 906113, 908837, 910525, 911457, 912087, 916141, 917081, 918105, 919051, 923223, 925425, 930491, 934795, 935193, 940229, 941237, 942307, 943463, 943943, 946475, 949235, 949559, 958243, 958443, 961407, 961919, 965977, 970925, 978039, 981253, 983603, 987079, 987381, 987427, 988517, 989499, 990573, 990617, 990619, 992507, 997521]

        # 380 out of 803
        self.test_file_inds = [266, 606, 2334, 2774, 3484, 7892, 7906, 10452, 27460, 33262, 34330, 37586, 40168, 40350, 41944, 43814, 44070, 45676, 55374, 56924, 58864, 58910, 62078, 65170, 66344, 68078, 69628, 77388, 78298, 78430, 80318, 80808, 96870, 98150, 100694, 102122, 103730, 115444, 119012, 122104, 122178, 123172, 125510, 126796, 127660, 127976, 129298, 131720, 133254, 136252, 137806, 139482, 142766, 143364, 143794, 147716, 148260, 149624, 154124, 154626, 156574, 159280, 159322, 161838, 162310, 168514, 172854, 174980, 176112, 176506, 180902, 182422, 185522, 185562, 192576, 192602, 192918, 194156, 204494, 204562, 210436, 211316, 219670, 221278, 226788, 238322, 245846, 246378, 255876, 263576, 264436, 273002, 273274, 276912, 277644, 277900, 277994, 282120, 283326, 291214, 293776, 294978, 296368, 298396, 300626, 306486, 307626, 309818, 311386, 312676, 315352, 315848, 316446, 318338, 321724, 322400, 324170, 325354, 326238, 330838, 331994, 332354, 337272, 338798, 340798, 340898, 343016, 345134, 345494, 347676, 349442, 350328, 351228, 352808, 358314, 358464, 362274, 373186, 382428, 383392, 387018, 387554, 394500, 397864, 402002, 403978, 405378, 405744, 412210, 416794, 419820, 420066, 420078, 424590, 427774, 434210, 439854, 442338, 444902, 455374, 457070, 457982, 458776, 462612, 467076, 470446, 470798, 471930, 472774, 476582, 479682, 483506, 491356, 491696, 495406, 495876, 496948, 499266, 499418, 499600, 501284, 501804, 503968, 504704, 508676, 509290, 511850, 513968, 514414, 516056, 520614, 524056, 524518, 530040, 533948, 533952, 534154, 536496, 538922, 541060, 543806, 544078, 544464, 547080, 548686, 549870, 550312, 552206, 552396, 556452, 556572, 563092, 565914, 568270, 570332, 570992, 571520, 575902, 577164, 584712, 586222, 586670, 586806, 587968, 588542, 589940, 599842, 600230, 601966, 605764, 606014, 606370, 607622, 609234, 612214, 615420, 616234, 616860, 617844, 618372, 619800, 620018, 621206, 624916, 625296, 626208, 627806, 629198, 638158, 638168, 639004, 639314, 644150, 646596, 649042, 649260, 651312, 651774, 654770, 661864, 664140, 664396, 665914, 669010, 669156, 671164, 675424, 676758, 678520, 679036, 682046, 682688, 688544, 691384, 692004, 692982, 698628, 699650, 702918, 705728, 706996, 708588, 714414, 715846, 725646, 727832, 733758, 739122, 739760, 747824, 753408, 759668, 762470, 763892, 765792, 767012, 772130, 772144, 772452, 775304, 778804, 784140, 784518, 786226, 794214, 803958, 805150, 807146, 808980, 810368, 818254, 819442, 825592, 825816, 827126, 828684, 829962, 830444, 831146, 834900, 839012, 841286, 841404, 842556, 847604, 848728, 848780, 850510, 853702, 860326, 864488, 866782, 873132, 875328, 876248, 877160, 878990, 880610, 889920, 893904, 902350, 904606, 912620, 914008, 916336, 916518, 918446, 919602, 923618, 924236, 925382, 926392, 927126, 927644, 930028, 935318, 937922, 939614, 942594, 942986, 946386, 946408, 947994, 950926, 951120, 952430, 954552, 956410, 956928, 965276, 967818, 969934, 971880, 981852, 982744, 986342, 991758, 994520, 995492, 998002]

        super().__init__(options)

    def plot_sample(self, file_path, x, y):
        fig = plt.figure()
        nb_cols = max(self.dataset.nb_channels + 1, self.dataset.nb_labels + 1)
        grid_spec = mpl.gridspec.GridSpec(2, nb_cols)

        # Plot x channels
        x = self.calibrate_image(x)
        rgb_x = x[:, :, self.dataset.rgb_inds]
        imgs = [rgb_x]
        nb_channels = x.shape[2]
        for channel_ind in range(nb_channels):
            img = x[:, :, channel_ind]
            imgs.append(img)
        row_ind = 0
        plot_img_row(fig, grid_spec, row_ind, imgs)

        # Plot y channels
        rgb_y = self.dataset.one_hot_to_rgb_batch(y)
        imgs = [rgb_y]
        for channel_ind in range(y.shape[2]):
            img = y[:, :, channel_ind]
            imgs.append(img)
        row_ind = 1
        plot_img_row(fig, grid_spec, row_ind, imgs)

        plt.savefig(file_path, bbox_inches='tight', format='pdf', dpi=600)
        plt.close(fig)


class DeepGlobeImageFileGenerator(DeepGlobeFileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    the original TIFF and JPG files.
    """
    def __init__(self, datasets_path, options):
        self.dataset_path = join(datasets_path, DEEPGLOBE)
        self.name = "deepglobe_image"
        super().__init__(options)

    @staticmethod
    def preprocess(datasets_path):
        # Fix the depth image that is missing a column if it hasn't been
        # fixed already.
        data_path = join(datasets_path, DEEPGLOBE)
        proc_data_path = join(datasets_path, PROCESSED_DEEPGLOBE)
        _makedirs(proc_data_path)

        class Options():
            def __init__(self):
                self.active_input_inds = [0, 1, 2]
                self.train_ratio = 0.8
                self.cross_validation = None

        options = Options()
        DeepGlobeImageFileGenerator(
            datasets_path, options).write_channel_stats(proc_data_path)

    def get_file_size(self, file_ind):
        rgb_file_path = join(
            self.dataset_path,
            '{}_sat.jpg'.format(file_ind))
        nb_rows, nb_cols = get_img_size(rgb_file_path)
        return nb_rows, nb_cols

    def get_img(self, file_ind, window):
        rgb_file_path = join(
            self.dataset_path,
            '{}_sat.jpg'.format(file_ind))
        batch_y_file_path = join(
            self.dataset_path,
            '{}_mask.png'.format(file_ind)) # noqa

        rgb = load_img(rgb_file_path, window)
        channels = [rgb]

        if self.has_y(file_ind):
            batch_y = load_img(batch_y_file_path, window)
            channels.extend([batch_y])

        img = np.concatenate(channels, axis=2)
        return img


    def make_batch(self, img_batch, file_inds):
        batch = DeepGlobeBatch()
        batch.all_x = img_batch[:, :, :, 0:3]
        batch.file_inds = file_inds

        if self.has_y(file_inds[0]):
            batch.y = self.dataset.rgb_to_one_hot_batch(
                img_batch[:, :, :, 3:6])
            batch.y_mask = self.dataset.rgb_to_mask_batch(
                img_batch[:, :, :, 3:6])
        return batch


class DeepGlobeNumpyFileGenerator(DeepGlobeFileGenerator):
    """
    A data generator for the Potsdam dataset that creates batches from
    numpy array files. This is about 20x faster than reading the raw files.
    """
    def __init__(self, datasets_path, options):
        self.raw_dataset_path = join(datasets_path, DEEPGLOBE)
        self.dataset_path = join(datasets_path, PROCESSED_DEEPGLOBE)
        # TODO: figure out download
        # self.download_dataset(['processed_deepglobe.zip'])
        self.name = "deepglobe_numpy"

        super().__init__(options)

    @staticmethod
    def preprocess(datasets_path):
        proc_data_path = join(datasets_path, PROCESSED_DEEPGLOBE)
        _makedirs(proc_data_path)

        class Options():
            def __init__(self):
                self.active_input_inds = [0, 1, 2]
                self.train_ratio = 0.8
                self.cross_validation = None

        options = Options()
        generator = DeepGlobeImageFileGenerator(datasets_path, options)
        dataset = generator.dataset

        def _preprocess(split):
            gen = generator.make_split_generator(
                split, batch_size=1, shuffle=False, augment_methods=None,
                normalize=False, only_xy=False)

            for batch in gen:
                print('.')
                file_ind = batch.file_inds[0]
                x = np.squeeze(batch.x, axis=0)
                channels = [x]

                if batch.y is not None:
                    y = np.squeeze(batch.y, axis=0)
                    y = dataset.one_hot_to_label_batch(y)
                    y_mask = np.squeeze(batch.y_mask, axis=0)
                    channels.extend([y, y_mask])
                channels = np.concatenate(channels, axis=2)

                file_name = '{}'.format(file_ind)
                save_numpy_array(
                    join(proc_data_path, file_name), channels)

                # Free memory
                channels = None
                batch.all_x = None
                batch.x = x = None
                batch.y = y = None
                batch.y_mask = y_mask = None

        _preprocess(TRAIN)
        _preprocess(VALIDATION)
        _preprocess(TEST)

        DeepGlobeNumpyFileGenerator(
            datasets_path, options).write_channel_stats(proc_data_path)

    def get_file_path(self, file_ind):
        return join(self.dataset_path, '{}.npy'.format(file_ind))

    def get_file_size(self, file_ind):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        nb_rows, nb_cols = im.shape[0:2]
        return nb_rows, nb_cols

    def get_img(self, file_ind, window):
        file_path = self.get_file_path(file_ind)
        im = np.load(file_path, mmap_mode='r')
        ((row_begin, row_end), (col_begin, col_end)) = window
        img = im[row_begin:row_end, col_begin:col_end, :]

        return img

    def make_batch(self, img_batch, file_inds):
        batch = DeepGlobeBatch()
        batch.all_x = img_batch[:, :, :, 0:3]
        batch.file_inds = file_inds

        if self.has_y(file_inds[0]):
            batch.y = self.dataset.label_to_one_hot_batch(
                img_batch[:, :, :, 3:4])
            batch.y_mask = img_batch[:, :, :, 4:5]
        return batch
