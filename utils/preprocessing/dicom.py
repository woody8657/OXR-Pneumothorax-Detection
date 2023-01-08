import pydicom
import numpy as np

class DICOM:
    def __init__(self, dcm_path):
        self.dcm_path = dcm_path

    def get_metadata(self):
        return pydicom.read_file(self.dcm_path)

    def get_img(
        self,
        fix_monochrome=True,
        normalization=True,
        apply_window=True,
        range_correct=True
    ):
        dicom = pydicom.read_file(self.dcm_path)
        # For ignoring the UserWarning: "Bits Stored" value (14-bit)...
        uid = dicom[0x0008, 0x0018].value
        elem = dicom[0x0028, 0x0101]
        elem.value = 16

        # get image
        img = dicom.pixel_array

        median = np.median(img)
        if range_correct:
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                img = np.where(img == 0, median, img)
            else:
                img = np.where(img == 4095, median, img)

        if normalization:
            if apply_window and "WindowCenter" in dicom and "WindowWidth" in dicom:
                window_center = float(dicom.WindowCenter)
                window_width = float(dicom.WindowWidth)
                y_min = (window_center - 0.5 * window_width)
                y_max = (window_center + 0.5 * window_width)
            else:
                y_min = img.min()
                y_max = img.max()
            img = (img - y_min) / (y_max - y_min)
            img = np.clip(img, 0, 1)

        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            img = np.amax(img) - img

        return img*255