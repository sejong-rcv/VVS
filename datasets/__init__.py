import os
import numpy as np
import pickle as pk
import tqdm

from collections import OrderedDict
from sklearn.metrics import average_precision_score

        
class CC_WEB_VIDEO(object):

    def __init__(self):
        with open('data/cc_web/cc_web_video.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'CC_WEB_VIDEO'
        self.database = dataset['index']
        self.queries = dataset['queries']
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        self.queries = [vid.split('/')[-1] for vid in self.queries]
        return self.queries

    def get_database(self):
        self.database = list(map(str, self.database.keys()))
        self.database = [vid.split('/')[-1] for vid in self.database]
        return self.database

    def calculate_mAP(self, similarities, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    video = self.database[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = self.database

        if verbose:
            print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
            not_found = len(set(self.queries) - similarities.keys())
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos'.format(len(all_db)))

        mAP = self.calculate_mAP(similarities, all_videos=False, clean=False)
        mAP_star = self.calculate_mAP(similarities, all_videos=True, clean=False)
        if verbose:
            print('-' * 25)
            print('Original Annotation')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(mAP, mAP_star))

        mAP_c = self.calculate_mAP(similarities, all_videos=False, clean=True)
        mAP_c_star = self.calculate_mAP(similarities, all_videos=True, clean=True)
        if verbose:
            print('Clean Annotation')
            print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(mAP_c, mAP_c_star))
        return {'mAP': mAP, 'mAP_star': mAP_star, 'mAP_c': mAP_c, 'mAP_c_star': mAP_c_star}

class VCDB(object):

    def __init__(self, pool_type="pos", root="data1_bnfreeze/features"):

        self.pool_type = pool_type

        if self.pool_type=="pos":
            with open('data1_bnfreeze/positive.pk', 'rb') as f:
                dataset = pk.load(f)
            all_ind = []
            for k, v in dataset.items():
                all_ind.extend(v)
            self.positive = dataset
            self.all_ind = all_ind
        elif self.pool_type=="same":
            with open('data1_bnfreeze/same.pk', 'rb') as f:
                dataset = pk.load(f)
            self.positive = dataset
            self.all_ind = dataset
        else:
            print("Pool Type check")
            import pdb; pdb.set_trace()


    def get_queries(self):
        return sorted(["{}/0".format(i) for i in self.all_ind])

    def get_database(self):
        if self.pool_type=="pos":
            return sorted(["{}/0".format(i) for i in self.all_ind])
        elif self.pool_type=="same":
            return sorted(["{}/1".format(i) for i in self.all_ind])

    def calculate_mAP(self, query, res, all_db):
        
        qid = int(query.split("/")[0])

        if self.pool_type == "pos":
            q_cluster = [k for k, v in self.positive.items() if qid in v ]
            q_len = len(self.positive[q_cluster[0]])
        elif self.pool_type == "same":
            q_len = 1

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                ri += 1

                dbid = int(video.split("/")[0])
                if self.pool_type == "pos":
                    db_cluster = [k for k, v in self.positive.items() if dbid in v ]
                    if q_cluster[0]==db_cluster[0]:
                        i += 1.0
                        s += i / ri
                elif self.pool_type == "same":
                    if qid==dbid:
                        i += 1.0
                        s += i / ri
        if q_len==0:
            return 0
        else: 
            return s / q_len

    def evaluate(self, similarities, all_db=None, verbose=True):
        if all_db is None:
            all_db = self.database
        APs=[]
        for query, res in similarities.items():
            APs.append(self.calculate_mAP(query, res, all_db))
        print_log = "[{}] - mAP: {:.4f}".format(self.pool_type, np.mean(APs))
        print(print_log)
    
        return [print_log]


class FIVR(object):

    def __init__(self, version='200k'):
        self.version = version
        with open('data/fivr/fivr.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'FIVR'
        self.annotation = dataset['annotation']
        self.queries = dataset[self.version]['queries']
        self.database = dataset[self.version]['database']

    def get_queries(self):
        return self.queries

    def get_database(self, memory_align=False, align_root="data1_eval/features"):
        if memory_align:
            target_list = list(self.database)
            target_size_list = [ 
                [i, os.path.getsize(os.path.join(align_root, i+".pt"))] 
                for i in target_list 
                if os.path.isfile(os.path.join(align_root, i+".pt"))]

            target_size_list = np.array(target_size_list)

            sort_ind = np.argsort(-target_size_list[:, 1].astype(float))
            return target_size_list[sort_ind][:, 0].tolist()
        else:
            return list(self.database)

    def calculate_mAP(self, query, res, all_db, relevant_labels):
        gt_sets = self.annotation[query]
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        if len(query_gt)==0:
            return 0
        else: 
            return s / len(query_gt)

    def _eval(self, similarities, all_db):
        DSVR, CSVR, ISVR = [], [], []
        for query, res in similarities.items():
            if query in self.queries:
                DSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS']))
                CSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS']))
                ISVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS', 'IS']))
        return DSVR, CSVR, ISVR

    def evaluate(self, similarities, all_db=None, verbose=True, use_fg=False, num_dict=None):
        if all_db is None:
            all_db = self.database
        if use_fg:
            if verbose:
                print_log = []
                print_log.append('=' * 5+'FIVR-{} Dataset'.format(self.version.upper())+'=' * 5)
                not_found = len(set(self.queries) - similarities.keys())
                if not_found > 0:
                    print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
                print_log.append('Queries: {} videos'.format(len(similarities["compx"]["bi"][-1.0])))
                print_log.append('Database: {} videos'.format(len(all_db)))
                print_log.append('-' * 16)

            for cox, v in similarities.items():
                for bu, vv in v.items():
                    for th, vvv in vv.items():
                        DSVR, CSVR, ISVR = self._eval(similarities[cox][bu][th], all_db)
                        if verbose:
                            line = "{} | {:3s} | {:3s} | DSVR: {:.4f} | CSVR: {:.4f} | ISVR: {:.4f}".format(
                                "CompO" if cox=="compo" else "CompX",
                                bu, 
                                str(th) if th!=-1.0 else "avg", 
                                np.mean(DSVR), np.mean(CSVR), np.mean(ISVR)
                            )
                            print_log.append(line)
                    if verbose:
                        print_log.append("\n-----------\n")
            if verbose:
                for pl in print_log:
                    print(pl)
            return print_log
        else: 
            DSVR, CSVR, ISVR = self._eval(similarities, all_db)
            if verbose:
                print_log = []
                print_log.append('=' * 5+'FIVR-{} Dataset'.format(self.version.upper())+'=' * 5)
                not_found = len(set(self.queries) - similarities.keys())
                if not_found > 0:
                    print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
                if num_dict is not None:
                    print_log.append("-"*10)
                    print_log.append('Frames:')
                    print_log.append('\t Queries: {}'.format(np.mean(num_dict["frame_q"])))
                    print_log.append('\t Database: {}'.format(np.mean(num_dict["frame_d"])))
                    print_log.append('Segments:')
                    print_log.append('\t Queries: {}'.format(np.mean(num_dict["seg_q"])))
                    print_log.append('\t Database: {}'.format(np.mean(num_dict["seg_d"])))
                    print_log.append("-"*10)

                print_log.append('Queries: {} videos'.format(len(similarities)))
                print_log.append('Database: {} videos'.format(len(all_db)))
                print_log.append('-' * 16)

                print_log.append('DSVR mAP: {:.4f}'.format(np.mean(DSVR)))
                print_log.append('CSVR mAP: {:.4f}'.format(np.mean(CSVR)))
                print_log.append('ISVR mAP: {:.4f}\n'.format(np.mean(ISVR)))

                for pl in print_log:
                    print(pl)

            return {'DSVR': np.mean(DSVR), 'CSVR': np.mean(CSVR), 'ISVR': np.mean(ISVR)}, print_log


class EVVE(object):

    def __init__(self):
        with open('data/evve/evve.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.name = 'EVVE'
        self.events = dataset['annotation']
        self.queries = dataset['queries']
        self.database = dataset['database']

        self.query_to_event = {qname: evname
                               for evname, (queries, _, _) in self.events.items()
                               for qname in queries}

    def get_queries(self):
        return list(self.queries)

    def get_database(self):
        return list(self.database)

    def score_ap_from_ranks_1(self, ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        return ap

    def evaluate(self, similarities, all_db=None, verbose=True):
        results = {e: [] for e in self.events}
        if all_db is None:
            all_db = set(self.database).union(set(self.queries))

        not_found = 0
        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                res = similarities[query]
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                pos_ranks = []

                ri, n_ext = 0.0, 0.0
                for ri, dbname in enumerate(sorted(res.keys(), key=lambda x: res[x], reverse=True)):
                    if dbname in pos:
                        pos_ranks.append(ri - n_ext)
                    if dbname not in all_db:
                        n_ext += 1

                ap = self.score_ap_from_ranks_1(pos_ranks, len(pos))
                results[evname].append(ap)
        if verbose:
            print('=' * 18, 'EVVE Dataset', '=' * 18)
            if not_found > 0:
                print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
            print('Queries: {} videos'.format(len(similarities)))
            print('Database: {} videos\n'.format(len(all_db - set(self.queries))))
            print('-' * 50)
        ap, mAP = [], []
        for evname in sorted(self.events):
            queries, _, _ = self.events[evname]
            nq = len(queries.intersection(all_db))
            ap.extend(results[evname])
            mAP.append(np.sum(results[evname]) / nq)
            if verbose:
                print('{0: <36} '.format(evname), 'mAP = {:.4f}'.format(np.sum(results[evname]) / nq))

        if verbose:
            print('=' * 50)
            print('overall mAP = {:.4f}'.format(np.mean(ap)))
        return {'mAP': np.mean(ap)}