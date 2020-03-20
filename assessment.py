from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda, UpSampling2D
from tensorflow.keras.models import Model
import os


def process_array(image_array, expand=True):
    """ Process a 3-dimensional array into a scaled, 4 dimensional batch of size 1. """

    image_batch = image_array / 255.0
    if expand:
        image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch


def process_output(output_tensor):
    """ Transforms the 4-dimensional output tensor into a suitable image format. """

    sr_img = output_tensor.clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img


def pad_patch(image_patch, padding_size, channel_last=True):
    """ Pads image_patch with with padding_size edge values. """

    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Splits the image into partially overlapping patches.

    The patches overlap by padding_size pixels.

    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.

    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """

    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size

    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size

    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')

    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)

    xmax, ymax, _ = padded_image.shape
    patches = []

    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)

    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)

    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Reconstruct the image from overlapping patches.

    After scaling, shapes and padding should be scaled too.

    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """

    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size

    complete_image = np.zeros((xmax, ymax, 3))

    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]


class ImageModel:
    """ISR models parent class.

    Contains functions that are common across the super-scaling models.
    """

    def predict(self, input_image_array, by_patch_of_size=None, batch_size=10, padding_size=2):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.

        Args:
            input_image_array: input image array.
            by_patch_of_size: for large image inference. Splits the image into
                patches of the given size.
            padding_size: for large image inference. Padding between the patches.
                Increase the value if there is seamlines.
            batch_size: for large image inferce. Number of patches processed at a time.
                Keep low and increase by_patch_of_size instead.
        Returns:
            sr_img: image output.
        """

        if by_patch_of_size:
            lr_img = process_array(input_image_array, expand=False)
            patches, p_shape = split_image_into_overlapping_patches(
                lr_img, patch_size=by_patch_of_size, padding_size=padding_size
            )
            # return patches
            for i in range(0, len(patches), batch_size):
                batch = self.model.predict(patches[i: i + batch_size])
                if i == 0:
                    collect = batch
                else:
                    collect = np.append(collect, batch, axis=0)

            scale = self.scale
            padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
            scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], scale)) + (3,)
            sr_img = stich_together(
                collect,
                padded_image_shape=padded_size_scaled,
                target_shape=scaled_image_shape,
                padding_size=padding_size * scale,
            )

        else:
            lr_img = process_array(input_image_array)
            sr_img = self.model.predict(lr_img)[0]

        sr_img = process_output(sr_img)
        return sr_img

WEIGHTS_URLS = {
    'psnr-large': {
        'arch_params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5',
        'name': 'rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5'
    },
    'psnr-small': {
        'arch_params': {'C': 3, 'D': 10, 'G': 64, 'G0': 64, 'x': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
        'name': 'rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
    },
    'noise-cancel': {
        'arch_params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5',
        'name': 'rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5',
    }
}


def make_model(arch_params, patch_size):
    """ Returns the model.

    Used to select the model.
    """

    return RDN(arch_params, patch_size)


def get_network(weights):
    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        name = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))
    c_dim = 3
    kernel_size = 3
    upscaling = 'ups'
    return arch_params, c_dim, kernel_size, upscaling, url, name


class RDN(ImageModel):
    """Implementation of the Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1802.08797 (Zhang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, x.
        patch_size: integer or None, determines the input size. Only needed at
            training time, for prediction is set to None.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        init_extreme_val: extreme values for the RandomUniform initializer.
        weights: string, if not empty, download and load pre-trained weights.
            Overrides other parameters.

    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        x: integer, the scaling factor.
        model: Keras model of the RDN.
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """

    def __init__(
            self,
            arch_params={},
            patch_size=None,
            c_dim=3,
            kernel_size=3,
            upscaling='ups',
            init_extreme_val=0.05,
            weights=''
    ):
        if weights:
            arch_params, c_dim, kernel_size, upscaling, url, fname = get_network(weights)

        self.params = arch_params
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.scale = self.params['x']
        self.patch_size = patch_size
        self.c_dim = c_dim
        self.kernel_size = kernel_size
        self.upscaling = upscaling
        self.initializer = RandomUniform(
            minval=-init_extreme_val, maxval=init_extreme_val, seed=None
        )
        self.model = self._build_rdn()
        self.model._name = 'generator'
        self.name = 'rdn'
        if weights:
            weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
            self.model.load_weights(weights_path)

    def _upsampling_block(self, input_layer):
        """ Upsampling block for old weights. """

        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            name='UPN3',
            kernel_initializer=self.initializer,
        )(input_layer)
        return UpSampling2D(size=self.scale, name='UPsample')(x)

    def _pixel_shuffle(self, input_layer):
        """ PixelShuffle implementation of the upscaling layer. """

        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            name='UPN3',
            kernel_initializer=self.initializer,
        )(input_layer)
        return Lambda(
            lambda x: tf.nn.depth_to_space(x, block_size=self.scale, data_format='NHWC'),
            name='PixelShuffle',
        )(x)

    def _UPN(self, input_layer):
        """ Upscaling layers. With old weights use _upsampling_block instead of _pixel_shuffle. """

        x = Conv2D(
            64,
            kernel_size=5,
            strides=1,
            padding='same',
            name='UPN1',
            kernel_initializer=self.initializer,
        )(input_layer)
        x = Activation('relu', name='UPN1_Relu')(x)
        x = Conv2D(
            32, kernel_size=3, padding='same', name='UPN2', kernel_initializer=self.initializer
        )(x)
        x = Activation('relu', name='UPN2_Relu')(x)
        if self.upscaling == 'shuffle':
            return self._pixel_shuffle(x)
        elif self.upscaling == 'ups':
            return self._upsampling_block(x)
        else:
            raise ValueError('Invalid choice of upscaling layer.')

    def _RDBs(self, input_layer):
        """RDBs blocks.

        Args:
            input_layer: input layer to the RDB blocks (e.g. the second convolutional layer F_0).

        Returns:
            concatenation of RDBs output feature maps with G0 feature maps.
        """
        rdb_concat = list()
        rdb_in = input_layer
        for d in range(1, self.D + 1):
            x = rdb_in
            for c in range(1, self.C + 1):
                F_dc = Conv2D(
                    self.G,
                    kernel_size=self.kernel_size,
                    padding='same',
                    kernel_initializer=self.initializer,
                    name='F_%d_%d' % (d, c),
                )(x)
                F_dc = Activation('relu', name='F_%d_%d_Relu' % (d, c))(F_dc)
                # concatenate input and output of ConvRelu block
                # x = [input_layer,F_11(input_layer),F_12([input_layer,F_11(input_layer)]), F_13..]
                x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d' % (d, c))
            # 1x1 convolution (Local Feature Fusion)
            x = Conv2D(
                self.G0, kernel_size=1, kernel_initializer=self.initializer, name='LFF_%d' % (d)
            )(x)
            # Local Residual Learning F_{i,LF} + F_{i-1}
            rdb_in = Add(name='LRL_%d' % (d))([x, rdb_in])
            rdb_concat.append(rdb_in)

        assert len(rdb_concat) == self.D

        return concatenate(rdb_concat, axis=3, name='LRLs_Concat')

    def _build_rdn(self):
        LR_input = Input(shape=(self.patch_size, self.patch_size, 3), name='LR')
        F_m1 = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='F_m1',
        )(LR_input)
        F_0 = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='F_0',
        )(F_m1)
        FD = self._RDBs(F_0)
        # Global Feature Fusion
        # 1x1 Conv of concat RDB layers -> G0 feature maps
        GFF1 = Conv2D(
            self.G0,
            kernel_size=1,
            padding='same',
            kernel_initializer=self.initializer,
            name='GFF_1',
        )(FD)
        GFF2 = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='GFF_2',
        )(GFF1)
        # Global Residual Learning for Dense Features
        FDF = Add(name='FDF')([GFF2, F_m1])
        # Upscaling
        FU = self._UPN(FDF)
        # Compose SR image
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(FU)

        return Model(inputs=LR_input, outputs=SR)
