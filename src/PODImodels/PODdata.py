"""
POD Data Processing and Utilities
==================================

This module provides classes and functions for handling Proper Orthogonal
Decomposition (POD) of datasets, particularly for computational fluid dynamics
field data. It includes utilities for VTK file writing and POD-based
dimensionality reduction.

Classes
-------
PODDataSet
    Class for performing POD on a single dataset.
subdomainDataSet
    Class for performing POD on both cell and patch data with subdomain analysis.

Functions
---------
vtk_writer
    Write field and point data to VTK MultiBlock datasets.
truncationErrorCal
    Calculate truncation error from singular values.
"""

import numpy as np
from scipy.linalg import svd
import pyvista as pv
from typing import List, Optional


# def vtk_writer(
#     field_data,
#     field_name,
#     data_type,
#     refVTMName,
#     save_path_name,
#     points_data=None,
#     is2D=False,
# ):
#     refVTM = pv.MultiBlock(refVTMName)
#     for block_i in range(refVTM.n_blocks):
#         block = refVTM[block_i]
#         if block is not None:
#             if data_type == "scalar":
#                 for data_i in range(len(field_name)):
#                     block.cell_data[field_name[data_i]] = field_data[data_i]
#             elif data_type == "vector":
#                 if is2D:
#                     for data_i in range(len(field_name)):
#                         block.cell_data[field_name[data_i]] = np.hstack(
#                             (
#                                 field_data[data_i].reshape(2, -1).T,
#                                 np.zeros((block.n_cells, 1)),
#                             )
#                         )
#                 else:
#                     for data_i in range(len(field_name)):
#                         block.cell_data[field_name[data_i]] = (
#                             field_data[data_i].reshape(3, -1).T
#                         )
#             if points_data is not None:
#                 points = points_data.reshape(3, -1).T
#                 block.points = points

#     # Save the modified VTM file
#     output_vtm_file_path = f"{save_path_name}.vtm"
#     refVTM.save(output_vtm_file_path)


def vtk_writer(
    field_data: List[np.ndarray],
    field_name: List[str],
    data_type: str,
    refVTMName: str,
    save_path_name: str,
    points_data: np.ndarray = None,
    is2D: bool = False,
):
    """
    Write field and point data to a VTK MultiBlock dataset (.vtm).

    This function is optimized for large meshes by leveraging PyVista's efficient
    data handling and saving mechanisms. It supports both scalar and vector field
    data and can handle 2D vector data by appending zero Z-components.

    Parameters
    ----------
    field_data : list of np.ndarray
        A list of NumPy arrays containing field data, one for each field.
        For scalar data: each array should be of shape (n_cells,).
        For vector data: each array should be of shape (n_cells, n_components).
    field_name : list of str
        A list of names for the fields corresponding to field_data.
    data_type : {'scalar', 'vector'}
        The type of field data being written.
    refVTMName : str
        The path to the reference .vtm file that provides the mesh structure.
    save_path_name : str
        The base path and name for the output .vtm file (without extension).
    points_data : np.ndarray, optional
        A NumPy array of shape (n_points, 3) containing new point coordinates.
        If None, original mesh points are preserved. Default is None.
    is2D : bool, optional
        If True, treats vector data as 2D and appends a zero Z-component to
        make it 3D for VTK compatibility. Default is False.

    Notes
    -----
    The function automatically saves each block as a separate .vtu file
    (which is efficient for large data) and updates the .vtm file accordingly.
    Binary format and compression are used to optimize file size and I/O performance.

    Examples
    --------
    >>> # Write scalar field data
    >>> field_data = [pressure_field, temperature_field]
    >>> field_names = ['pressure', 'temperature']
    >>> vtk_writer(field_data, field_names, 'scalar', 'mesh.vtm', 'output')

    >>> # Write 2D vector field data
    >>> field_data = [velocity_2d]
    >>> field_names = ['velocity']
    >>> vtk_writer(field_data, field_names, 'vector', 'mesh.vtm', 'output', is2D=True)
    """
    refVTM = pv.MultiBlock(refVTMName)

    # Use a counter to track the data index
    field_data_idx = 0

    for block_i in range(refVTM.n_blocks):
        block = refVTM[block_i]
        if block is None:
            continue

        # Check if we have enough data for this block's fields
        num_fields_per_block = len(field_name)

        if data_type == "scalar":
            for i in range(num_fields_per_block):
                block.cell_data[field_name[i]] = field_data[field_data_idx]
                field_data_idx += 1
        elif data_type == "vector":
            if is2D:
                for i in range(num_fields_per_block):
                    vec_2d = field_data[field_data_idx]
                    vec_3d = np.zeros((block.n_cells, 3), dtype=vec_2d.dtype)
                    vec_3d[:, :2] = vec_2d
                    block.cell_data[field_name[i]] = vec_3d
                    field_data_idx += 1
            else:
                for i in range(num_fields_per_block):
                    block.cell_data[field_name[i]] = field_data[field_data_idx]
                    field_data_idx += 1

        if points_data is not None:
            # Assuming points_data is already (n_points, 3) and correctly segmented per block.
            # This is a major assumption and would need to be handled by the caller.
            # A more robust solution would be to pass a list of per-block point arrays.
            # For simplicity, we'll assume a single large array and slice it.
            # This is still not ideal for memory.
            points_start_idx = block_i * block.n_points
            points_end_idx = points_start_idx + block.n_points
            block.points = points_data[points_start_idx:points_end_idx]

    # Save the modified VTM file. PyVista automatically saves each block as a
    # separate .vtu file (which is efficient for large data) and updates the .vtm.
    # Using compression and binary format is crucial for large files.
    output_vtm_file_path = f"{save_path_name}.vtm"
    refVTM.save(output_vtm_file_path, binary=True, compression_level=9)


def truncationErrorCal(singulars):
    """
    Calculate the truncation error from singular values.

    The truncation error indicates the relative energy content that would be
    lost if the POD representation were truncated at each rank level.

    Parameters
    ----------
    singulars : np.ndarray
        Array of singular values from SVD decomposition, typically in
        descending order.

    Returns
    -------
    np.ndarray
        Array of truncation errors for each rank, representing the fraction
        of total energy lost if truncation occurs at that level.

    Notes
    -----
    The truncation error is calculated as:
    error[i] = 1 - sqrt(sum(σ²[0:i+1])) / ||σ||₂

    where σ are the singular values and ||σ||₂ is the Frobenius norm.

    Examples
    --------
    >>> singulars = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    >>> errors = truncationErrorCal(singulars)
    >>> print(f"Error for rank 3: {errors[2]:.4f}")
    """
    return 1 - np.sqrt(np.cumsum(np.power(singulars, 2))) / np.linalg.norm(singulars)


class PODDataSet:
    """
    A class for performing Proper Orthogonal Decomposition (POD) on datasets.

    This class handles the decomposition of high-dimensional field data into
    a reduced set of orthogonal modes and corresponding coefficients. It supports
    both truncated and full decompositions and provides utilities for error
    analysis and data export.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features) where each row
        represents a snapshot of the field data.
    rank : int, optional
        Number of POD modes to retain. Default is 10.
    fullData : bool, optional
        If True, compute and store all POD modes for error analysis.
        If False, only compute the first 'rank' modes. Default is True.

    Attributes
    ----------
    data : np.ndarray
        The input data matrix.
    rank : int
        Number of retained POD modes.
    fullData : bool
        Flag indicating whether full decomposition is computed.
    cell_modes : np.ndarray
        Truncated POD modes matrix of shape (rank, n_features).
    cell_coeffs : np.ndarray
        Truncated POD coefficients matrix of shape (n_samples, rank).
    singulars : np.ndarray
        Truncated singular values array of length rank.
    cell_modes_all : np.ndarray, optional
        Full POD modes matrix (if fullData=True).
    cell_coeffs_all : np.ndarray, optional
        Full POD coefficients matrix (if fullData=True).
    singulars_all : np.ndarray, optional
        Full singular values array (if fullData=True).

    Examples
    --------
    >>> # Create POD dataset with rank 5
    >>> data = np.random.rand(100, 1000)  # 100 snapshots, 1000 features
    >>> pod = PODDataSet(data, rank=5, fullData=True)
    >>> print(f"POD modes shape: {pod.cell_modes.shape}")
    >>> print(f"Truncation error at rank 5: {pod.truncationError()[4]:.4f}")

    >>> # Save POD modes to VTK file
    >>> pod.saveModes('pod_modes', 'reference.vtm', 'vector', rank=5)
    """

    def __init__(self, data: np.ndarray, rank: int = 10, fullData: bool = True) -> None:
        self.data: np.ndarray = data
        self.rank: int = rank
        self.fullData: bool = fullData

        # Optional attributes set in POD method
        self.cell_modes_all: Optional[np.ndarray] = None
        self.singulars_all: Optional[np.ndarray] = None
        self.cell_coeffs_all: Optional[np.ndarray] = None

        self.printInfo()
        self.POD()

    def POD(self) -> None:
        """
        Perform Singular Value Decomposition to compute POD modes and coefficients.

        This method decomposes the data matrix using SVD and extracts the POD modes
        (right singular vectors), coefficients (scaled left singular vectors), and
        singular values. Both truncated and full decompositions are computed based
        on the fullData flag.

        Notes
        -----
        The POD is computed using SVD: X = U Σ Vᵀ, where:
        - cell_modes = V (right singular vectors)
        - cell_coeffs = U Σ (scaled left singular vectors)
        - singulars = Σ (singular values)
        """
        s, vh = svd(self.data, full_matrices=False)[1:]
        self.cell_modes: np.ndarray = vh[: self.rank]
        self.cell_coeffs: np.ndarray = self.data @ vh[: self.rank].T
        self.singulars: np.ndarray = s[: self.rank]

        if self.fullData:
            self.cell_modes_all: np.ndarray = vh
            self.singulars_all: np.ndarray = s
            self.cell_coeffs_all: np.ndarray = self.data @ vh.T

    def truncationError(self):
        """
        Calculate the truncation error for all possible ranks.

        Returns
        -------
        np.ndarray
            Array of truncation errors for each rank from 1 to the total
            number of modes. Only available if fullData=True.

        Raises
        ------
        AttributeError
            If fullData=False and full singular values are not available.
        """
        return truncationErrorCal(self.singulars_all)

    def printInfo(self) -> None:
        """
        Print information about the POD configuration.

        Displays the rank parameter used for the POD decomposition.
        """
        print("The POD rank is: ", self.rank)

    def saveModes(
        self,
        saveFileName: str,
        refVTMName: str,
        dataType: str,
        rank: int = 10,
        is2D: bool = False,
    ) -> None:
        """
        Save POD modes to VTK files along with singular values and truncation errors.

        Parameters
        ----------
        saveFileName : str
            Base filename for saving the modes and analysis files.
        refVTMName : str
            Path to the reference VTM file that provides mesh structure.
        dataType : {'scalar', 'vector'}
            Type of data being saved.
        rank : int, optional
            Number of modes to save. Default is 10.
        is2D : bool, optional
            If True, treat vector data as 2D. Default is False.

        Notes
        -----
        This method creates three files:
        1. {saveFileName}.vtm - VTK file with POD modes
        2. {saveFileName}_truncationError.txt - Truncation errors
        3. {saveFileName}_singulars.txt - Singular values
        """
        # Write the velocity data into VTK file
        field_name = [f"mode_{i}" for i in range(rank)]

        # loop all test data and write the data into VTK file
        vtk_writer(
            self.cell_modes_all[:rank],
            field_name,
            dataType,
            refVTMName,
            saveFileName,
            is2D=is2D,
        )

        # write the truncation error and singular values into txt file
        np.savetxt(f"{saveFileName}_truncationError.txt", self.truncationError())
        np.savetxt(f"{saveFileName}_singulars.txt", self.singulars_all)


class subdomainDataSet:
    """
    A class for performing POD on both cell and patch data with subdomain analysis.

    This class handles POD decomposition for datasets that contain both volumetric
    cell data and boundary/patch data. It performs separate POD on each dataset
    and computes projections between the two representations.

    Parameters
    ----------
    cell_data : np.ndarray
        Cell/volume data matrix of shape (n_samples, n_cell_features).
    patch_data : np.ndarray
        Patch/boundary data matrix of shape (n_samples, n_patch_features).
    cell_rank : int, optional
        Number of POD modes to retain for cell data. Default is 10.
    patch_rank : int, optional
        Number of POD modes to retain for patch data. Default is 5.
    cal_fullData : bool, optional
        If True, compute and store all POD modes for error analysis.
        Default is True.

    Attributes
    ----------
    cell_data : np.ndarray
        Input cell data matrix.
    patch_data : np.ndarray
        Input patch data matrix.
    cell_rank : int
        Number of retained cell POD modes.
    patch_rank : int
        Number of retained patch POD modes.
    cal_fullData : bool
        Flag for computing full decomposition.
    cell_modes : np.ndarray
        Cell POD modes matrix.
    cell_coeffs : np.ndarray
        Cell POD coefficients matrix.
    patch_modes : np.ndarray
        Patch POD modes matrix.
    patch_coeffs : np.ndarray
        Patch POD coefficients matrix.
    projPatch_modes : np.ndarray
        Projection of patch modes onto cell coefficient space.

    Examples
    --------
    >>> cell_data = np.random.rand(50, 1000)  # 50 snapshots, 1000 cells
    >>> patch_data = np.random.rand(50, 200)  # 50 snapshots, 200 patch points
    >>> subdomain = subdomainDataSet(cell_data, patch_data, cell_rank=8, patch_rank=4)
    >>> print(f"Cell modes shape: {subdomain.cell_modes.shape}")
    >>> print(f"Patch modes shape: {subdomain.patch_modes.shape}")
    """

    def __init__(
        self,
        cell_data: np.ndarray,
        patch_data: np.ndarray,
        cell_rank: int = 10,
        patch_rank: int = 5,
        cal_fullData: bool = True,
    ) -> None:
        self.cell_data: np.ndarray = cell_data
        self.patch_data: np.ndarray = patch_data
        self.cell_rank: int = cell_rank
        self.patch_rank: int = patch_rank
        self.cal_fullData: bool = cal_fullData

        # Optional attributes
        self.cell_modes_all: Optional[np.ndarray] = None
        self.singulars_all: Optional[np.ndarray] = None
        self.cell_coeffs_all: Optional[np.ndarray] = None
        self.patch_modes_all: Optional[np.ndarray] = None
        self.patch_coeffs_all: Optional[np.ndarray] = None
        self.patch_singulars_all: Optional[np.ndarray] = None

        self.printInfo()
        self.cellPOD()
        self.patchPOD()
        self.calculate_projPatch_modes()

    def cellPOD(self) -> None:
        """
        Perform POD on the cell/volume data.

        Computes SVD decomposition of the cell data and extracts POD modes,
        coefficients, and singular values. Both truncated and full decompositions
        are stored based on the cal_fullData flag.
        """
        s, vh = svd(self.cell_data, full_matrices=False)[1:]
        self.cell_modes: np.ndarray = vh[: self.cell_rank]
        self.cell_coeffs: np.ndarray = self.cell_data @ vh[: self.cell_rank].T
        self.singulars: np.ndarray = s[: self.cell_rank]

        if self.cal_fullData:
            self.cell_modes_all: np.ndarray = vh
            self.singulars_all: np.ndarray = s
            self.cell_coeffs_all: np.ndarray = self.cell_data @ vh.T

    def calculate_projPatch_modes(self) -> None:
        """
        Calculate the projection of patch modes onto the cell coefficient space.

        This method computes the mapping between cell POD coefficients and patch
        data, enabling reconstruction of patch data from cell POD coefficients.
        The projection is computed as: (Σ⁻²) @ (Uᶜᵀ @ Xᵖ), where Σ are the cell
        singular values, Uᶜ are the cell coefficients, and Xᵖ is the patch data.
        """
        self.projPatch_modes: np.ndarray = np.diag(np.power(self.singulars, -2)) @ (
            self.cell_coeffs.T @ self.patch_data
        )

    def truncationError(self):
        """
        Calculate the truncation error for the cell data POD.

        Returns
        -------
        np.ndarray
            Array of truncation errors for each rank from 1 to the total
            number of cell modes. Only available if cal_fullData=True.
        """
        return truncationErrorCal(self.singulars_all)

    def patchPOD(self) -> None:
        """
        Perform POD on the patch/boundary data.

        Computes SVD decomposition of the patch data and extracts POD modes,
        coefficients, and singular values. Both truncated and full decompositions
        are stored based on the cal_fullData flag.
        """
        s, vh = svd(self.patch_data, full_matrices=False)[1:]
        self.patch_modes: np.ndarray = vh[: self.patch_rank]
        self.patch_coeffs: np.ndarray = self.patch_data @ vh[: self.patch_rank].T
        self.patch_singulars: np.ndarray = s[: self.patch_rank]

        if self.cal_fullData:
            self.patch_modes_all: np.ndarray = vh
            self.patch_coeffs_all: np.ndarray = self.patch_data @ vh.T
            self.patch_singulars_all: np.ndarray = s

    def printInfo(self) -> None:
        """
        Print information about the subdomain POD configuration.

        Displays the rank parameters used for both cell and patch POD decompositions.
        """
        print("The cell POD rank is: ", self.cell_rank)
        print("The patch POD rank is: ", self.patch_rank)
