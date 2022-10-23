from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.cat import CAT
import pandas as pd
from tqdm import tqdm
import json


# path to pre-trained MedCAT model here. Check https://github.com/CogStack/MedCAT#available-models for free models.
cat = CAT.load_model_pack('/data/users/k1897038/mc_modelpack_phase2_snomed_190k_october_2021.zip')

line_count = 0
collected_ents = []
iters = 0
with open('/data/users/k1897038/mimic_summarisation/hadms_to_dis_course.json') as f, open('/data/users/k1897038/hadms_to_dis_course_processed.jsonl', 'w') as f_out:
    line = f.readline()
    while line != '':
        line = json.loads(line)
        line_count += 1
        line['ents'] = cat.get_entities(line['text'])
        collected_ents.append(json.dumps(line))
        if line_count == 100:
            print(f'writing iter {iters} to disk')
            f_out.writelines('\n'.join(collected_ents))
            f_out.write('\n')
            line_count = 0
            collected_ents = []
            iters += 1
        line = f.readline()
        # write last batch to output
    f_out.writelines('\n'.join(collected_ents))
    