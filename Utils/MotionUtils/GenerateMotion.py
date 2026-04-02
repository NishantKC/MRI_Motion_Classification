# Hammersmith Hospital using a Philips 3T system
# Guy's Hospital using a Philips 1.5T system
# Institute of Psychiatry using a GE 1.5T system
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image

from Utils.MotionUtils.ImageTransform import ImageTransformer
from Utils.kspace.CartesianSampler import CartesianSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
baseDir = PROJECT_ROOT / "mris" / "disc1"
genDir = PROJECT_ROOT / "generated"

MOTION_PARAMS = {
    0: {"displacement": [0, 0, 0], "rotation": [0, 0, 0]},
    1: {"displacement": [1, 1, 1], "rotation": [1, 1, 1]},
    2: {"displacement": [2, 2, 2], "rotation": [2, 2, 2]},
    3: {"displacement": [3, 3, 3], "rotation": [3, 3, 3]},
    4: {"displacement": [5, 5, 5], "rotation": [5, 5, 5]},
}

fig, axes = plt.subplots(1, 2)


def showSlice(slices):
    for index, slice in enumerate(slices):
        axes[index].imshow(slice, cmap="gray", origin="lower")


def saveSlice(slice, severity, suffix=''):
    slice = np.squeeze(slice)
    if slice.ndim != 2:
        return
    normalized = linearNormalization(slice)
    normalizedImg = (normalized * 255).astype(np.uint8)
    im = Image.fromarray(normalizedImg, mode='L')
    outDir = genDir / f"M{severity}"
    outDir.mkdir(parents=True, exist_ok=True)
    imageName = outDir / f"{suffix}.tiff"
    im.save(str(imageName))


def linearNormalization(values):
    minVal = np.min(values)
    maxVal = np.max(values)
    if maxVal - minVal == 0:
        return np.zeros_like(values)
    return (values - minVal) / (maxVal - minVal)


def findANumberWithMod0(primaryNum):
    firstNum = int(primaryNum / 2)
    while firstNum > 2:
        if primaryNum % firstNum == 0:
            return firstNum
        firstNum -= 1
    return 2


def generateMotion(img, voxelRes, maxDisplacementInMillimeter, maxRotInDegree, primaryAxis=2, severity=0, imageNameSuffix=''):
    voxelRes = np.asarray(voxelRes)
    maxDisplacementInMillimeter = np.asarray(maxDisplacementInMillimeter)
    maxRotInDegree = np.asarray(maxRotInDegree)

    nT = img.shape[primaryAxis]
    axes = (0, 1)
    if primaryAxis == 0:
        axes = (1, 2)
    elif primaryAxis == 1:
        axes = (0, 2)
    elif primaryAxis == 2:
        axes = (0, 1)

    if severity == 0:
        kspaceSamplerWithoutMovement = CartesianSampler(img.shape, axes=axes)
        kspaceSamplerWithoutMovement.distortedImage = img
        for time in range(int(nT / 3), nT - int(nT / 3), 1):
            sliceData = kspaceSamplerWithoutMovement.getSlice(time)
            saveSlice(sliceData, severity, f"{imageNameSuffix}_{time}")
        return

    maxDisplacementInPixel = np.floor(maxDisplacementInMillimeter / voxelRes)
    displacementPixelTrajectory = np.zeros((3, nT))
    rotationDegreeTrajectory = np.zeros((3, nT))
    for i in range(3):
        randomMovement = maxDisplacementInPixel[i] * linearNormalization(generate_perlin_noise_2d((1, nT), [1, findANumberWithMod0(nT)]))
        displacementPixelTrajectory[i, :] = np.round(randomMovement)
        randomRotation = maxRotInDegree[i] * linearNormalization(generate_perlin_noise_2d((1, nT), [1, findANumberWithMod0(nT)]))
        rotationDegreeTrajectory[i, :] = np.round(randomRotation)

    kspaceSampler = CartesianSampler(img.shape, axes=axes)

    imageTransform = ImageTransformer(img)
    for time in range(nT):
        rotatedImage = imageTransform.rotate_along_axis(rotationDegreeTrajectory[0, time], rotationDegreeTrajectory[1, time], rotationDegreeTrajectory[2, time]
                                                        , displacementPixelTrajectory[0, time], displacementPixelTrajectory[1, time], displacementPixelTrajectory[2, time])
        kspaceSampler.sample(rotatedImage, time)
        imageTransform = ImageTransformer(img)
    kspaceSampler.calculateImageAfterSampling()
    for time in range(int(nT / 3), nT - int(nT / 3), 1):
        sliceData = kspaceSampler.getSlice(time)
        saveSlice(sliceData, severity, f"{imageNameSuffix}_{time}")


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def find_analyze_images(base_path):
    """Find all Analyze .hdr files in OASIS dataset structure (use processed images)."""
    hdr_files = []
    for subject_dir in base_path.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith("OAS1_"):
            processed_dir = subject_dir / "PROCESSED" / "MPRAGE" / "T88_111"
            if processed_dir.exists():
                for hdr_file in processed_dir.glob("*_gfc.hdr"):
                    if "masked" not in hdr_file.name:
                        hdr_files.append(hdr_file)
                        break
            if not any(f.parent.parent.parent.parent.name == subject_dir.name for f in hdr_files[-1:] if hdr_files):
                subj_dir = subject_dir / "PROCESSED" / "MPRAGE" / "SUBJ_111"
                if subj_dir.exists():
                    for hdr_file in subj_dir.glob("*.hdr"):
                        hdr_files.append(hdr_file)
                        break
    return hdr_files


if __name__ == "__main__":
    genDir.mkdir(exist_ok=True)
    for severity in range(5):
        (genDir / f"M{severity}").mkdir(exist_ok=True)

    mri_files = find_analyze_images(baseDir)
    print(f"Found {len(mri_files)} MRI images to process")

    for hdr_file in mri_files:
        try:
            imgStructure = nib.load(str(hdr_file))
            voxelSize = imgStructure.header["pixdim"]
            data = np.asarray(imgStructure.get_fdata(), dtype=np.float32)
            data = np.squeeze(data)  # Remove singleton dimensions (176,208,176,1) -> (176,208,176)
            subject_name = hdr_file.parent.parent.parent.parent.name
            if not subject_name.startswith("OAS1_"):
                subject_name = hdr_file.stem.split("_mpr")[0]
            subject_name = subject_name.replace("_MR1", "")

            print(f"Processing: {subject_name} (shape: {data.shape})")

            for severity in range(5):
                params = MOTION_PARAMS[severity]
                generateMotion(
                    data,
                    voxelSize[1:4],
                    maxDisplacementInMillimeter=params["displacement"],
                    maxRotInDegree=params["rotation"],
                    primaryAxis=0,
                    severity=severity,
                    imageNameSuffix=subject_name
                )
            print(f"  Completed {subject_name}")
        except Exception as e:
            print(f"  Error processing {hdr_file}: {e}")

    print("Motion generation complete!")
