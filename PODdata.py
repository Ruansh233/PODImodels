import numpy as np
from scipy.linalg import svd
import pyvista as pv


def vtk_writer(
    field_data,
    field_name,
    data_type,
    refVTMName,
    save_path_name,
    points_data=None,
    is2D=False,
):
    refVTM = pv.MultiBlock(refVTMName)
    for block_i in range(refVTM.n_blocks):
        block = refVTM[block_i]
        if block is not None:
            if data_type == "scalar":
                for data_i in range(len(field_name)):
                    block.cell_data[field_name[data_i]] = field_data[data_i]
            elif data_type == "vector":
                if is2D:
                    for data_i in range(len(field_name)):
                        block.cell_data[field_name[data_i]] = np.hstack(
                            (
                                field_data[data_i].reshape(2, -1).T,
                                np.zeros((block.n_cells, 1)),
                            )
                        )
                else:
                    for data_i in range(len(field_name)):
                        block.cell_data[field_name[data_i]] = (
                            field_data[data_i].reshape(3, -1).T
                        )
            if points_data is not None:
                points = points_data.reshape(3, -1).T
                block.points = points

    # Save the modified VTM file
    output_vtm_file_path = f"{save_path_name}.vtm"
    refVTM.save(output_vtm_file_path)


def truncationErrorCal(singulars):
    return 1 - np.sqrt(np.cumsum(np.power(singulars, 2))) / np.linalg.norm(singulars)


class PODDataSet:
    def __init__(self, data, rank=10, fullData=True):
        self.data = data
        self.rank = rank
        self.fullData = fullData

        self.printInfo()
        self.POD()

    def POD(self):
        s, vh = svd(self.data, full_matrices=False)[1:]
        self.cell_modes = vh[: self.rank]
        self.cell_coeffs = self.data @ vh[: self.rank].T
        self.singulars = s[: self.rank]

        if self.fullData:
            self.cell_modes_all = vh
            self.singulars_all = s
            self.cell_coeffs_all = self.data @ vh.T

    def truncationError(self):
        return truncationErrorCal(self.singulars_all)

    def printInfo(self):
        print("The POD rank is: ", self.rank)

    def saveModes(self, saveFileName, refVTMName, dataType, rank=10, is2D=False):
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
    def __init__(
        self, cell_data, patch_data, cell_rank=10, patch_rank=5, cal_fullData=True
    ):
        self.cell_data = cell_data
        self.patch_data = patch_data
        self.cell_rank = cell_rank
        self.patch_rank = patch_rank
        self.cal_fullData = cal_fullData

        self.printInfo()
        self.cellPOD()
        self.patchPOD()
        self.calculate_projPatch_modes()

    def cellPOD(self):
        s, vh = svd(self.cell_data, full_matrices=False)[1:]
        self.cell_modes = vh[: self.cell_rank]
        self.cell_coeffs = self.cell_data @ vh[: self.cell_rank].T
        self.singulars = s[: self.cell_rank]

        if self.cal_fullData:
            self.cell_modes_all = vh
            self.singulars_all = s
            self.cell_coeffs_all = self.cell_data @ vh.T

    def calculate_projPatch_modes(self):
        self.projPatch_modes = np.diag(np.power(self.singulars, -2)) @ (
            self.cell_coeffs.T @ self.patch_data
        )

    def truncationError(self):
        return truncationErrorCal(self.singulars_all)

    def patchPOD(self):
        s, vh = svd(self.patch_data, full_matrices=False)[1:]
        self.patch_modes = vh[: self.patch_rank]
        self.patch_coeffs = self.patch_data @ vh[: self.patch_rank].T
        self.patch_singulars = s[: self.patch_rank]

        if self.cal_fullData:
            self.patch_modes_all = vh
            self.patch_coeffs_all = self.patch_data @ vh.T
            self.patch_singulars_all = s

    def printInfo(self):
        print("The cell POD rank is: ", self.cell_rank)
        print("The patch POD rank is: ", self.patch_rank)
