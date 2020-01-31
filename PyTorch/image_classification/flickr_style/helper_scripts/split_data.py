from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import os

def main(csv_path, split_ratio=0.3, sep = ",", root_path="./"):
    df = pd.read_csv(csv_path, names=None, header=None, sep=sep)
    train, test = train_test_split(df, test_size=split_ratio, random_state=4213, shuffle=True, stratify=df[1])
    train.to_csv(os.path.join(root_path, "split_train.csv"), sep = ",", index = False, header = False)
    test.to_csv(os.path.join(root_path, "split_val.csv"), sep = ",", index = False, header = False)
    
    
    
if __name__ == '__main__':
    try:
        csv_path = sys.argv[1].rstrip()
        split_ratio = float(sys.argv[2].rstrip())
        root_path = sys.argv[3].rstrip()
    except Exception as e:
        print("Incorrect usage.")
    sep = ","
    main(csv_path, split_ratio, sep, root_path)
