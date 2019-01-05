import os
import pandas as pd

import sys
import tensorflow as tf
import glob

from pathlib import Path


def extract_write_accuracy_csv(path):
    runlog = pd.DataFrame(columns=['Wall time', 'Step', 'Value'])
    csv_path = "accuracy_csv/{}.csv".format(path.split('/')[-2])
    if Path(csv_path).is_file():
        return None
    try:
        for e in tf.train.summary_iterator(path):
            for v in e.summary.value:
                if v.tag == "valid/accuracy":
                    r = {'Wall time': e.wall_time, 'Step': e.step, 'Value': v.simple_value }
                    runlog = runlog.append(r, ignore_index=True)
    except:
        print('Event file possibly corrupt: {}'.format(path))
        return None


    print("writing csv...".format(csv_path))
    runlog = runlog.set_index('Wall time')
    runlog.to_csv(csv_path)


if __name__ == '__main__':
    logging_dir = sys.argv[1]
    event_paths = glob.glob(os.path.join(logging_dir, "*","event*"))
    # Extraction function
    for path in event_paths:
        print("calling....", path)
        extract_write_accuracy_csv(path)

