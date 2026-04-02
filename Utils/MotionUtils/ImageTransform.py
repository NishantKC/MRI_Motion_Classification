import numpy as np
from scipy import ndimage


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))


def deg_to_rad(deg):
    return deg * np.pi / 180.0


class ImageTransformer(object):
    """3D volume transformation class for MRI data with shape (X, Y, Z)"""

    def __init__(self, image):
        self.image = np.squeeze(image)  # Ensure 3D
        if self.image.ndim == 2:
            self.image = self.image[:, :, np.newaxis]
        self.shape = self.image.shape

    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        """Rotate 3D volume along axes with translation."""
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)

        # Build 3x3 rotation matrix (for 3D volume)
        RX = np.array([
            [1, 0, 0],
            [0, np.cos(rtheta), -np.sin(rtheta)],
            [0, np.sin(rtheta), np.cos(rtheta)]
        ])

        RY = np.array([
            [np.cos(rphi), 0, np.sin(rphi)],
            [0, 1, 0],
            [-np.sin(rphi), 0, np.cos(rphi)]
        ])

        RZ = np.array([
            [np.cos(rgamma), -np.sin(rgamma), 0],
            [np.sin(rgamma), np.cos(rgamma), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = RZ @ RY @ RX

        # Center of the volume
        center = np.array(self.shape) / 2.0

        # Offset to rotate around center, then apply translation
        offset = center - R @ center + np.array([dx, dy, dz])

        # Apply affine transformation
        rotated = ndimage.affine_transform(
            self.image,
            R,
            offset=offset,
            output_shape=self.shape,
            order=1,  # Linear interpolation (faster)
            mode='constant',
            cval=0.0
        )

        return rotated
