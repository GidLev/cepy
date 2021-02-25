'''
import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/path/to/cepy')
import cepy as ce
parms = {'dimensions': 30, 'walk_length': 20, 'num_walks': 800, 'workers': 1,
         'p': 0.1, 'q': 1.6, 'verbosity': 2, 'window': 3, 'min_count': 0,
         'iter': 1, 'permutations': 5, 'seed': 2}

sc_subject1 = ce.get_example('sc_subject1_matrix')
ce_subject1 = ce.CE(**parms)
ce_subject1.fit(sc_subject1)
ce_subject1.save_model('ce_subject1.json')
ce_subject1.save_model('ce_subject1.json.gz')

sc_subject2 = ce.get_example('sc_subject2_matrix')
ce_subject2 = ce.CE(**parms)
ce_subject2.fit(sc_subject2)
ce_subject2.save_model('ce_subject2.json.gz')

sc_group = ce.get_example('sc_group_matrix')
ce_group = ce.CE(**parms)
ce_group.fit(sc_group)
ce_group.save_model('ce_group.json.gz')

print('Done generating examples.')
'''