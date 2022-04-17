import numpy as np
from .._model import DMmodel, Generalmodel
from ...util.drawutil import drawRectbin

import math
import pickle
import time, datetime



def recurFindAwakePt(xs, ys, start=0, abptidxs=[]):
    if len(ys)<=1 or len(xs)<=1:
        return []
    maxidx = np.argmax(ys)
    x0,y0,xm,ym = xs[0], ys[0], xs[maxidx], ys[maxidx]
#     sqco = math.sqrt((ym-y0)**2 + (xm-x0)**2) #sqrt of coefficient
    xvec, yvec = xs[:maxidx], ys[:maxidx]
#     dts = ((ym-y0)*xvec - (xm-x0)*yvec + (xm*y0 - ym*x0))/sqco
    dts = ((ym-y0)*xvec - (xm-x0)*yvec + (xm*y0 - ym*x0))
    if len(dts) > 0:
        xaidx = np.argmax(dts)
        abptidxs.append((xaidx+start, maxidx+start))
    else :
        xaidx = 0
    'left'
    recurFindAwakePt(xs[:xaidx], ys[:xaidx], start=start, abptidxs=abptidxs)
    'right'
    diffyincrese = np.argwhere(np.diff(ys[maxidx:]) >0)
    if len(diffyincrese) > 0:
        turningptidx = diffyincrese[0,0]+maxidx
        recurFindAwakePt(xs[turningptidx:], ys[turningptidx:],
                         start = turningptidx + start,
                         abptidxs=abptidxs)
    return abptidxs


def sleepingbeauty(xy):
    xy = sorted(xy, key=lambda x: x[0])
    xs = np.array([x[0] for x in xy])
    ys = np.add.accumulate([x[1] for x in xy])
    abptidxs = []
    abptidxs = recurFindAwakePt(xs, ys, abptidxs=abptidxs)
    if abptidxs:
        delta_x = [xs[b] - xs[a] for a, b in abptidxs]
        delta_y = [ys[b] - ys[a] for a, b in abptidxs]
        maxid = np.argmax(delta_y)
        start_ptr = abptidxs[maxid][0]
        maxid_deltay = delta_y[maxid]
        maxid_deltax = delta_x[maxid]
        return account_id, start_ptr, maxid_deltax, maxid_deltay
    return None

def boundary(x, k=1.5, upper=True):
#     print(x[:100])
    q75, q25 = np.percentile(x, [75, 25])
#     print("75 %s 25 %s" %(q75, q25))
    iqr = q75 - q25
    if upper:
        return np.ceil(q75 + k * iqr)
    else:
        return np.floor(q25 - k * iqr)

import scipy.stats as ss
EPS = 1e-5
def get_pareto_score(objscores, alpha=0.9, p=0.9):
    sortscores = np.sort(objscores)
    sortobjs = np.argsort(objscores)

    tail_fir_score = np.percentile(sortscores, alpha*100)

    # print(f'{(1-alpha)*100}% tail_fir_score:', tail_fir_score)

    try:
        tailidx = np.argwhere(sortscores >= tail_fir_score - EPS)[0][0]
    except Exception as e:
        print(sortscores)
        print(sortobjs)
        print("score", tail_fir_score)
        raise(e)
    tailscores = sortscores[tailidx:]
    tailobjs = sortobjs[tailidx:]
    # print('obj len:', len(tailscores))

    shape, pos, scale = ss.pareto.fit(tailscores)
    cdfs = ss.pareto.cdf(tailscores, shape, pos, scale)

    # print('max cdf:', max(cdfs))
    thre = min(p, max(cdfs))

    levelidxs = np.argwhere(cdfs >= thre)
    levelobjs = tailobjs[levelidxs].T[0]
    if len(objscores[levelobjs]) == 0:
        return -1
    fit_tailscore = min(objscores[levelobjs])
    return fit_tailscore

def get_bound(xs, ys, k = 1.5):
    # 分别按count和countIn求整体的四分位点
    upper_y_bound = boundary(ys, k=k)
    lower_y_bound = boundary(ys, k=k, upper=False)
    print(f'upper_y_bound:{upper_y_bound}, lower_y_bound:{lower_y_bound}')

    upper_x_bound = boundary(xs, k=k)
    lower_x_bound = boundary(xs, k=k, upper=False)
    print(f'upper_x_bound:{upper_x_bound}, lower_x_bound:{lower_x_bound}')

    mid_y = np.percentile(ys, 50)

    print('mid', mid_y)
    bound = (lower_y_bound, upper_y_bound, lower_x_bound, upper_x_bound)
    return bound

def pareto_detection(xs, ys, ids, k = 1.5, alpha = 0.9, p = 0.9, binnum = None):
    lower_y_bound, upper_y_bound, lower_x_bound, upper_x_bound = get_bound(xs, ys, k)
    mid_y = np.percentile(ys, 50)

    INIT_TAILSCORE = upper_x_bound
    STEP = 0.05
    global_tailscore = INIT_TAILSCORE
    is_cross_mid = False

    filter_tmp = (ys > lower_y_bound) & (ys < upper_y_bound)
    xs_tmp = xs[filter_tmp]
    ids_tmp = ids[filter_tmp]
    ys_tmp = ys[filter_tmp].copy()
    
    if binnum is None:
        dict_y_values = dict(Counter(ys_tmp))
        bins = set(dict_y_values.keys())
        ls = dict((x, x) for x in bins)
        mid_y = np.percentile(ys, 50)
    else:
        bins = set()
        ls = np.linspace(np.min(ys_tmp), np.max(ys_tmp), binnum)
        for i, y in enumerate(ys_tmp):
            idx = np.searchsorted(ls, y)
            bins.add(idx)
            ys_tmp[i] = idx
    print("#bins:", len(bins))

    cnt = 0
    list_result_ids = []
    for v in bins:
        selected_v = ys_tmp == v
        x_tmp = xs_tmp[selected_v]
        id_tmp = ids_tmp[selected_v]
        fit_tailscore = get_pareto_score(x_tmp, alpha, p)

        if is_cross_mid == False and v <= mid_y:
            is_cross_mid = True
            global_tailscore = INIT_TAILSCORE

        if len(x_tmp[x_tmp>fit_tailscore]) == 0: 
            # too high, so use global instead
            pass
        else:
            # limit the change.
            if (fit_tailscore > global_tailscore + STEP):
                global_tailscore = global_tailscore + STEP
            elif fit_tailscore > global_tailscore:
                # update 
                global_tailscore = fit_tailscore

        fit_tailscore = global_tailscore
        tail_ids = np.reshape(np.argwhere(x_tmp >= fit_tailscore - EPS), (-1))
        list_result_ids.extend(list(id_tmp[tail_ids]))
        if cnt % 1000 == 0:
            print("Now processed %i", cnt)
        cnt += 1
    return list_result_ids

def get_timebin(year, month, day):
    dt = datetime.datetime(year, month, day)
    t = time.mktime(dt.timetuple())
    return t / 60

def vis_detection_result(list_result, list_result_ids, xs, ys, k = 1.5):
    list_x = []
    list_y = []
    set_result_ids = set(list_result_ids)
    for x in list_result:
        if x[0] in set_result_ids:
            for i in range(10000):
                list_x.append(x[1])
                list_y.append(x[3])
        list_x.append(x[1])
        list_y.append(x[3])
    
    fig, hist = drawRectbin(list_x, list_y)
    plt.show()

def parse_line1(line):
    tokens = line.split("\t")
    account_id = int(tokens[0])
    xy = []
    for token in tokens[1:]:
        timebin, money = token.split(',')
        xy.append((int(timebin), int(money)))
    return account_id, xy


INIT_DATE = get_timebin(2019, 1, 1)
def parse_line2(line):
    '''
    Format: [account_id][create_date][[[trans_date], [trans_amount]]...]
    '''    
    tokens = line.split("\t")
    account_id = tokens[0]
    year, month, day = map(int, tokens[1].split('-'))
    start_date = get_timebin(year, month, day)
    year, month, day, minus = map(int, tokens[2].split(',')[0].split('-'))
    first_date = get_timebin(year, month, day)
    st_timebin = start_date if (start_date <= first_date and start_date > INIT_DATE) else INIT_DATE


    for token in tokens[2:]:
        timebin, money = token.split(',')
        year, month, day, minus = map(int, timebin.split('-'))
        timebin = minus + get_timebin(year, month, day) - st_timebin
        xy.append((int(timebin), float(money)))
        if timebin < 0:
            print("ValueError. Trans time must be positive.")

    return account_id, xy
    

class SleepingBeauty(Generalmodel):
    def __init__(self):
        pass

    def run(self, filepath, output_path1, output_path2, threshold, alpha = 0.9, p = 0.98, binnum = 1000):
        list_result = []

        with open(filepath, "r") as fp:
            for line in fp:
                account_id, xy = parse_line2(line)
                xy = sorted(xy, key=lambda x: x[0])
                xs = np.array([x[0] for x in xy])
                ys = np.add.accumulate([x[1] for x in xy])
                abptidxs = []
                abptidxs = recurFindAwakePt(xs, ys, abptidxs=abptidxs)
                if abptidxs:
                    abptidxs = sorted(abptidxs, key=lambda x:x[0])
                    delta_y = np.array([ys[b] - ys[a] for a, b in abptidxs])
                    delta_x = np.array([xs[b] - xs[a] for a, b in abptidxs])
                    delta_y = np.array(delta_y)
                    maxid = np.argmax(delta_y > threshold)
                    start_ptr = abptidxs[maxid][0]
                    max_delta_y = delta_y[maxid]
                    max_delta_x = delta_x[maxid]
                    if max_delta_y < threshold:
                        continue
                    list_result.append((account_id, start_ptr, max_delta_x, max_delta_y))

        with open(output_path1, "wb") as wfp:
            pickle.dump(list_result, wfp)

        ids = np.array([x[0] for x in list_result])
        ptrs = np.array([x[1] for x in list_result])
        xs = np.array([x[2] for x in list_result])
        ys = np.array([x[3] for x in list_result])

        # 静默期 - delta_y
        drawRectbin(ptrs, ys, "deltay.png")

        # 静默期 - delta_y / delta_x
        drawRectbin(ptrs, ys / xs, "deltay_deltax.png")

        # 静默期 - delta_y^2 / delta_x
        drawRectbin(ptrs, np.square(ys) / xs, "deltay2_deltax.png")


        # with open("cbank_result.pkl", "rb") as fp:
        #     list_result = pickle.load(fp)
        filter1 = [True if x[1] >= 1 and x[3] >= 1 else False for x in list_result]
        ptrs = np.log10(ptrs[filter1])
        ys = np.log10(ys[filter1])
        ids = ids[filter1]

        list_result_ids = pareto_detection(ptrs, ys, ids, k = 1.5, alpha = alpha, p = p, binnum = p)
        
        with open(output_path2, "wb") as wfp:
            pickle.dump(list_result_ids, wfp)
        print(len(list_result_ids))

        vis_detection_result(list_result, list_result_ids, xs, ys, k = 1.5)
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass
