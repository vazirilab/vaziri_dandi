"""An imaging extractor for .mat files based on HDF5.

Classes
-------
LbmHdf5ImagingExtractor
    An imaging extractor for HDF5.
"""
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import numpy as np
import re

from roiextractors.extraction_tools import PathType, FloatType, ArrayType
from roiextractors.extraction_tools import (
    get_video_shape,
    write_to_h5_dataset_format,
)
from roiextractors.imagingextractor import ImagingExtractor
#from lazy_ops import DatasetView
from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import BaseImagingExtractorInterface
from neuroconv.utils import FolderPathType

try:
    import h5py

    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False


class LbmHdf5ImagingExtractor(ImagingExtractor):
    """An imaging extractor for LBM-style HDF5-based mat files."""

    extractor_name = "LbmHdf5Imaging"
    installed = HAVE_H5  # check at class level if installed or not
    is_writable = True
    mode = "file"
    installation_mesg = "To use the Hdf5 Extractor run:\n\n pip install h5py\n\n"  # error message when not installed

    def __init__(
        self,
        file_path: PathType,
        mov_field: str = "Y",
        sampling_frequency: FloatType = None,
        start_time: FloatType = None,
        metadata: dict = None,
        channel_names: ArrayType = None,
    ):
        """Create an ImagingExtractor from an HDF5 file.

        Parameters
        ----------
        file_path : str or Path
            Path to the HDF5 file.
        mov_field : str, optional
            Name of the dataset in the HDF5 file that contains the imaging data. The default is "mov".
        sampling_frequency : float, optional
            Sampling frequency of the video. The default is None.
        start_time : float, optional
            Start time of the video. The default is None.
        metadata : dict, optional
            Metadata dictionary. The default is None.
        channel_names : array-like, optional
            List of channel names. The default is None.
        """
        ImagingExtractor.__init__(self)

        self.filepath = Path(file_path)
        self._sampling_frequency = sampling_frequency
        self._mov_field = mov_field
        if self.filepath.suffix not in [".h5", ".hdf5", ".mat"]:
            warn("'file_path' file is not an .hdf5 or .h5 or .mat file")
        self._channel_names = channel_names

        self._file = h5py.File(file_path, "r")
        if mov_field in self._file.keys():
            self._video = self._file[self._mov_field]
            if sampling_frequency is None:
                assert "volumeRate" in self._file.keys(), (
                    "Sampling frequency is unavailable as a dataset attribute! "
                    "Please set the keyword argument 'sampling_frequency'"
                )
                self._sampling_frequency = self._file["volumeRate"][0,0]
            else:
                self._sampling_frequency = sampling_frequency
        else:
            raise Exception(f"{file_path} does not contain the 'Y' dataset")

        if start_time is None:
            if "start_time" in self._video.attrs.keys():
                self._start_time = self._video.attrs["start_time"]
        else:
            self._start_time = start_time

        if metadata is None:
            if "pixelResolution" in self._file.keys():
                self.metadata = {}
                self.metadata["pixel_size_um"] = self._file["pixelResolution"][0,0]
        else:
            self.metadata = metadata

        self._num_channels = 1
        self._num_frames, self._num_cols, self._num_rows = self._video.shape  # returns (time, col, row)
        self._dtype = self._video[0].dtype
        # self._video = self._video.lazy_transpose([2, 0, 1])  # should be: (samples, rows, columns)

        if self._channel_names is not None:
            assert len(self._channel_names) == self._num_channels, (
                "'channel_names' length is different than number " "of channels"
            )
        else:
            self._channel_names = [f"channel_{ch}" for ch in range(self._num_channels)]

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "mov_field": mov_field,
            "sampling_frequency": sampling_frequency,
            "channel_names": channel_names,
        }

    def __del__(self):
        """Close the HDF5 file."""
        self._file.close()

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0):
        # Fancy indexing is non performant for h5.py with long frame lists
        if frame_idxs is not None:
            slice_start = np.min(frame_idxs)
            slice_stop = min(np.max(frame_idxs) + 1, self.get_num_frames())
        else:
            slice_start = 0
            slice_stop = self.get_num_frames()
        
        # ix order in self._video: [t, c, r] 
        # ix order expected by DANDI in the final file: [t, r, c]
        # however, the following happens in imaginextractordatachunkiterator.py, line 140:
        # tranpose_axes = (0, 2, 1) if len(data.shape) == 3 else (0, 2, 1, 3)
        # so it flips the c and r axis there, anyways, we don't have to do that here
        frames = self._video[slice_start:slice_stop, :, :]
        if isinstance(frame_idxs, int):
            frames = frames.squeeze()

        return frames

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        return self._video[start_frame:end_frame, :, :]

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_cols)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        return self._channel_names

    def get_num_channels(self):
        return self._num_channels
    
    def get_dtype(self):
        return self._dtype

    @staticmethod
    def write_imaging(
        imaging: ImagingExtractor,
        save_path,
        overwrite: bool = False,
        mov_field="mov",
        **kwargs,
    ):
        """Write an imaging extractor to an HDF5 file.

        Parameters
        ----------
        imaging : ImagingExtractor
            The imaging extractor object to be saved.
        save_path : str or Path
            Path to save the file.
        overwrite : bool, optional
            If True, overwrite the file if it already exists. The default is False.
        mov_field : str, optional
            Name of the dataset in the HDF5 file that contains the imaging data. The default is "mov".
        **kwargs : dict
            Keyword arguments to be passed to the HDF5 file writer.

        Raises
        ------
        AssertionError
            If the file extension is not .h5 or .hdf5.
        FileExistsError
            If the file already exists and overwrite is False.
        """
        save_path = Path(save_path)
        assert save_path.suffix in [
            ".h5",
            ".hdf5",
        ], "'save_path' file is not an .hdf5 or .h5 file"

        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        with h5py.File(save_path, "w") as f:
            write_to_h5_dataset_format(imaging=imaging, dataset_path=mov_field, file_handle=f, **kwargs)
            dset = f[mov_field]
            dset.attrs["fr"] = imaging.get_sampling_frequency()

from roiextractors.extraction_tools import PathType, FloatType, ArrayType, DtypeType, get_package
from roiextractors.imagingextractor import ImagingExtractor
from roiextractors.volumetricimagingextractor import VolumetricImagingExtractor

class LbmHdf5MultiPlaneImagingExtractor(VolumetricImagingExtractor):
    """Specialized extractor for reading multi-plane (volumetric) .mat files produced by LBM pre-processing."""

    extractor_name = "LbmHdf5MultiPlaneImaging"
    is_writable = True
    mode = "file"

    def __init__(
        self,
        folder_path: PathType,
        channel_name: Optional[str] = None,
    ) -> None:
        self.folder_path = Path(folder_path)
        self.mat_file_paths = list(self.folder_path.glob('*.mat'))
        sort_ixs = [int(re.match('.*_(\d*)\.mat', fp.as_posix()).group(1)) for fp in self.mat_file_paths]
        sort_ixs_sorting = np.argsort(sort_ixs)
        self.mat_file_paths = [self.mat_file_paths[ix] for ix in sort_ixs_sorting]
        self.metadata = None  # extract_extra_metadata(file_path)
        parsed_metadata = None  # parse_metadata(self.metadata)
        num_planes = len(self.mat_file_paths)  # parsed_metadata["num_planes"]
        channel_names = ['green']  # parsed_metadata["channel_names"]
        if channel_name is None:
            channel_name = channel_names[0]
        imaging_extractors = []
        for plane in range(num_planes):
            imaging_extractor = LbmHdf5ImagingExtractor(
                file_path=self.mat_file_paths[plane]
            )
            imaging_extractors.append(imaging_extractor)
        super().__init__(imaging_extractors=imaging_extractors)


class LbmHdf5ImagingInterface(BaseImagingExtractorInterface):
    ExtractorModuleName = "lbmhdf5imagingextractor"
    ExtractorName = "LbmHdf5MultiPlaneImagingExtractor"

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["file_path"]["description"] = "Path to imported pre-processed .mat files."
        return source_schema

    def __init__(
        self,
        folder_path: FolderPathType,
        #fallback_sampling_frequency: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        DataInterface for reading .mat files that are generated by LBM preprocessing

        Parameters
        ----------
        file_path: str
            Path to tiff file.
        fallback_sampling_frequency: float, optional
            The sampling frequency can usually be extracted from the scanimage metadata in
            exif:ImageDescription:state.acq.frameRate. If not, use this.
        """
        '''
        self.image_metadata = extract_extra_metadata(file_path=file_path)

        if "state.acq.frameRate" in self.image_metadata:
            sampling_frequency = float(self.image_metadata["state.acq.frameRate"])
        elif "SI.hRoiManager.scanFrameRate" in self.image_metadata:
            sampling_frequency = float(self.image_metadata["SI.hRoiManager.scanFrameRate"])
        else:
            assert_msg = (
                "sampling frequency not found in image metadata, "
                "input the frequency using the argument `fallback_sampling_frequency`"
            )
            assert fallback_sampling_frequency is not None, assert_msg
        '''
        #sampling_frequency = fallback_sampling_frequency
        super().__init__(folder_path=folder_path, verbose=verbose)
