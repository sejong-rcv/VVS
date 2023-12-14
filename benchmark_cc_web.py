import os 
import numpy as np 
import tabulate
import json

def txt_parser(path):
    data = json.load(open(path))

    mAP = round(data['mAP'], 3)
    mAP_star = round(data['mAP_star'], 3)
    mAP_c = round(data['mAP_c'], 3)
    mAP_c_star = round(data['mAP_c_star'], 3)
    mAP_avg = mAP + mAP_star + mAP_c + mAP_c_star
    
    return [int(path.split("/")[-2][1:]), mAP, mAP_star, mAP_c, mAP_c_star, mAP_avg/4]

    
if __name__ == "__main__":

    summarize = True

    target_list = [
        "jobs/table_benchmark_dim_500",
        "jobs/table_benchmark_dim_512",
        "jobs/table_benchmark_dim_1024",
        "jobs/table_benchmark_dim_3840"    
    ]
    
    columns = ["iter", "mAP", "mAP_star", "mAP_c", "mAP_c_star", "AVG"]
    for target in target_list:
        eval_root = os.path.join(target, "eval")
        eval_folder = sorted(os.listdir(eval_root))
        perform = [txt_parser(os.path.join(eval_root, i, "sim_v.json")) for i in eval_folder]
        if summarize:
            perform = np.asarray(perform)
            maxind  = np.argmax(perform[:, -1])
            perform = [perform[maxind].tolist()]
        print(target)
        print(tabulate.tabulate(perform, headers=columns, floatfmt=".3f")+"\n\n")
