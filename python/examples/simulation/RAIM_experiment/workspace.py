from raim_torch_hierarchicalfl_step_by_step_exp import raim
from raim_rs_torch_hierarchicalfl_step_by_step_exp import raim_rs
import openpyxl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    esnum = 20
    ednum = 70

    raim_su, raim_cu = raim(True, esnum, ednum, 0.0)
    raimrs_su, raimrs_cu = raim_rs(True, esnum, ednum, 0.0)
    print("[RAIM] sdnum:{} esnum:{} raim_su{} raim_cu{}".format(ednum, esnum, raim_su, raim_cu))
    print("[RAIM-RS] sdnum:{} esnum:{} raimrs_su{} raimrs_cu{}".format(ednum, esnum, raimrs_su, raimrs_cu))