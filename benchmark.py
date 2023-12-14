import os 
import numpy as np 
import tabulate


def txt_parser(path):

    data = np.loadtxt(path, dtype=str, delimiter="\n")

    dsvr = round(float(data[4].split(": ")[1]), 3)
    csvr = round(float(data[5].split(": ")[1]), 3)
    isvr = round(float(data[6].split(": ")[1]), 3)

    
    return [int(path.split("/")[-2][1:]), dsvr, csvr, isvr, np.mean([dsvr, csvr, isvr])]

    
if __name__ == "__main__":

    summarize = True

    target_list = [
        "./jobs/supplementary/Table_C/dim_500/table_3_ablation_0",
        "./jobs/supplementary/Table_C/dim_500/table_3_ablation_1",
        "./jobs/supplementary/Table_C/dim_500/table_3_ablation_2",
        "./jobs/supplementary/Table_C/dim_500/table_3_ablation_3",
        "./jobs/supplementary/Table_C/dim_500/table_3_ablation_4",
        "./jobs/supplementary/Table_C/dim_500/table_3_ablation_5",
        "./jobs/supplementary/Table_C/dim_500/table_3_ablation_6",
    ]

    columns = ["iter", "DSVR", "CSVR", "ISVR", "AVG"]
    
    for target in target_list:
        eval_root = os.path.join(target, "eval")
        eval_folder = sorted(os.listdir(eval_root))
        perform = [txt_parser(os.path.join(eval_root, i, "sim_v.txt")) for i in eval_folder]
        if summarize:
            perform = np.asarray(perform)
            maxind  = np.argmax(perform[:, -1])
            perform = [perform[maxind].tolist()]
        print(target)
        print(tabulate.tabulate(perform, headers=columns, floatfmt=".3f")+"\n\n")
